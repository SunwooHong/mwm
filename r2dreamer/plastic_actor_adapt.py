#!/usr/bin/env python3
"""Test-time Fast plastic + Slow actor adaptation on gain-shift phases.

Design constraints (RL fine-tuning pitfalls):
  1. Shift-only replay: actor updates sample only from adapt-phase buffer (default Phase B).
  2. Trace freezing: imagination uses a detached clone of P_t (no writes during rollout).
  3. Adapt ratio: collect adapt_every env steps, then run adapt_epochs actor updates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_env
from envs import wrappers
from plastic import LogitPlasticResidual
from plastic_inference import parse_phases


def load_agent_for_adapt(
    logdir: Path,
    checkpoint: Path,
    device: str,
    *,
    seed: int,
    plastic_enabled: bool,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    actor_adapt: bool,
    actor_lr_scale: float = 1.0,
) -> tuple[Dreamer, object]:
    cfg = OmegaConf.load(logdir / ".hydra/config.yaml")
    cfg.device = device
    cfg.buffer.storage_device = device
    cfg.model.device = device
    cfg.model.compile = False
    cfg.env.env_num = 1
    cfg.env.eval_episode_num = 0
    cfg.seed = seed
    if not hasattr(cfg.model, "plastic"):
        cfg.model.plastic = OmegaConf.create({})
    cfg.model.plastic.enabled = plastic_enabled
    cfg.model.plastic.inference = plastic_enabled
    cfg.model.plastic.eval_only = True
    cfg.model.plastic.lr = plastic_lr
    cfg.model.plastic.gate_mode = gate_mode
    cfg.model.plastic.persist_threshold = persist_threshold
    cfg.model.plastic.max_norm = max_trace_norm

    env = make_env(cfg.env, 0)
    env = wrappers.ActionGain(env, gain=1.0)
    agent = Dreamer(cfg.model, env.observation_space, env.action_space).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if "optims_state_dict" in ckpt:
        tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])

    for name, param in agent._named_params.items():
        param.requires_grad_(actor_adapt and name.startswith("actor."))

    if actor_adapt and actor_lr_scale != 1.0:
        for group in agent._optimizer.param_groups:
            group["lr"] = float(group["lr"]) * actor_lr_scale

    agent.clone_and_freeze()
    agent.set_plastic_inference(plastic_enabled)
    if actor_adapt:
        agent.train()
        agent._slow_value.train(False)
    else:
        agent.eval()
    return agent, env


def make_shift_buffer(device: str, batch_size: int, batch_length: int) -> Buffer:
    cfg = OmegaConf.create(
        {
            "device": device,
            "storage_device": device,
            "batch_size": batch_size,
            "batch_length": batch_length,
            "max_size": max(5000, batch_size * (batch_length + 1) * 4),
        }
    )
    return Buffer(cfg)


def obs_to_batch(obs: dict, device: str) -> dict:
    out = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
    if "is_first" not in out:
        out["is_first"] = torch.zeros(1, dtype=torch.bool, device=device)
    return out


def transition_from_step(
    obs_td: dict,
    action: torch.Tensor,
    state: TensorDict,
    reward: float,
    done: bool,
    episode_id: int,
) -> TensorDict:
    trans = {k: v.detach() for k, v in obs_td.items()}
    trans["action"] = action.detach()
    trans["reward"] = torch.tensor([reward], dtype=torch.float32, device=action.device)
    trans["stoch"] = state["stoch"].detach()
    trans["deter"] = state["deter"].detach()
    trans["episode"] = torch.tensor([episode_id], dtype=torch.int32, device=action.device)
    trans["is_last"] = torch.tensor([done], dtype=torch.bool, device=action.device)
    if "is_terminal" not in trans:
        trans["is_terminal"] = torch.tensor([False], dtype=torch.bool, device=action.device)
    return TensorDict(trans, batch_size=(1,))


def buffer_ready(buf: Buffer, batch_size: int, batch_length: int) -> bool:
    need = batch_size * (batch_length + 1)
    return buf.count() >= need


def run_adaptation(
    *,
    logdir: Path,
    checkpoint: Path,
    outdir: Path,
    device: str,
    phases,
    seed: int,
    plastic_enabled: bool,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    actor_adapt: bool,
    plastic_wm_imagine: bool,
    adapt_phase_names: set[str],
    adapt_every: int,
    adapt_epochs: int,
    buffer_batch_size: int,
    buffer_batch_length: int,
    run_label: str,
    actor_lr_scale: float = 1.0,
) -> pd.DataFrame:
    agent, env = load_agent_for_adapt(
        logdir,
        checkpoint,
        device,
        seed=seed,
        plastic_enabled=plastic_enabled,
        plastic_lr=plastic_lr,
        gate_mode=gate_mode,
        persist_threshold=persist_threshold,
        max_trace_norm=max_trace_norm,
        actor_adapt=actor_adapt,
        actor_lr_scale=actor_lr_scale,
    )
    actor_lr = float(agent._optimizer.param_groups[0]["lr"]) if actor_adapt else 0.0

    shift_buffer = make_shift_buffer(device, buffer_batch_size, buffer_batch_length)
    min_ready = buffer_batch_size * (buffer_batch_length + 1)

    state = agent.get_initial_state(1)
    obs = env.reset()
    episode_id = 0
    rows = []
    adapt_logs = []

    for phase in phases:
        if hasattr(env, "set_gain"):
            env.set_gain(phase.action_gain)
        collect_here = phase.name in adapt_phase_names
        if collect_here:
            shift_buffer = make_shift_buffer(device, buffer_batch_size, buffer_batch_length)

        for local_t in range(phase.steps):
            obs_td = obs_to_batch(obs, device)
            for key in obs_td:
                if obs_td[key].dtype != torch.bool:
                    obs_td[key] = obs_td[key].float()

            with torch.no_grad():
                action, state = agent.act(obs_td, state, eval=True)
            metrics = dict(agent.last_plastic_metrics)

            obs, reward, done, _ = env.step(tools.to_np(action)[0])

            if collect_here and actor_adapt:
                shift_buffer.add_transition(transition_from_step(obs_td, action, state, reward, done, episode_id))

            if (
                collect_here
                and actor_adapt
                and adapt_every > 0
                and local_t > 0
                and (local_t % adapt_every) == 0
                and buffer_ready(shift_buffer, buffer_batch_size, buffer_batch_length)
            ):
                trace_snapshot = None
                if plastic_wm_imagine and plastic_enabled and "plastic_P" in state.keys():
                    trace_snapshot = LogitPlasticResidual.clone_trace(agent._unpack_trace(state))
                for _ in range(adapt_epochs):
                    adapt_mets = agent.update_actor_plastic(
                        shift_buffer,
                        trace_snapshot,
                        use_plastic_wm=plastic_wm_imagine and plastic_enabled,
                    )
                adapt_logs.append(
                    {
                        "phase": phase.name,
                        "local_t": local_t,
                        "buffer_count": shift_buffer.count(),
                        "adapt_every": adapt_every,
                        "adapt_epochs": adapt_epochs,
                        "actor_lr_scale": actor_lr_scale,
                        "actor_lr": actor_lr,
                        **{
                            k: float(v.detach()) if torch.is_tensor(v) else v
                            for k, v in adapt_mets.items()
                        },
                    }
                )

            if done:
                obs = env.reset()
                state = agent.get_initial_state(1)
                episode_id += 1

            rows.append(
                {
                    "run_label": run_label,
                    "seed": seed,
                    "actor_adapt": actor_adapt,
                    "plastic_online": plastic_enabled,
                    "plastic_wm_imagine": plastic_wm_imagine,
                    "adapt_every": adapt_every,
                    "adapt_epochs": adapt_epochs,
                    "actor_lr_scale": actor_lr_scale,
                    "actor_lr": actor_lr,
                    "phase_idx": phase.phase_idx,
                    "phase": phase.name,
                    "action_gain": phase.action_gain,
                    "local_t": local_t,
                    "reward": float(reward),
                    "buffer_count": shift_buffer.count() if collect_here else 0,
                    **metrics,
                }
            )

    step_df = pd.DataFrame(rows)
    step_df.to_csv(outdir / f"step_metrics_{run_label}.csv", index=False)
    if adapt_logs:
        pd.DataFrame(adapt_logs).to_csv(outdir / f"adapt_log_{run_label}.csv", index=False)
    return step_df


def summarize_runs(step_df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "run_label",
        "actor_adapt",
        "plastic_online",
        "plastic_wm_imagine",
        "seed",
        "phase_idx",
        "phase",
        "action_gain",
    ]
    agg = {
        "reward_mean": ("reward", "mean"),
        "reward_sum": ("reward", "sum"),
        "steps": ("reward", "count"),
    }
    if "prior_mse" in step_df.columns:
        agg["prior_mse_mean"] = ("prior_mse", "mean")
        agg["prior_mse_last20"] = ("prior_mse", lambda s: float(s.tail(20).mean()))
    if "trace_norm" in step_df.columns:
        agg["trace_norm_max"] = ("trace_norm", "max")
    out = step_df.groupby(keys, as_index=False).agg(**agg).sort_values(keys)

    rows = []
    for seed in out["seed"].unique():
        sub = out[out["seed"] == seed]
        b_adapt = sub[(sub["actor_adapt"]) & (sub["phase"].str.startswith("B"))]
        b_no_adapt = sub[(~sub["actor_adapt"]) & (sub["phase"].str.startswith("B"))]
        row = {"seed": int(seed)}
        if len(b_adapt) == 1 and len(b_no_adapt) == 1:
            row["B_reward_delta_adapt_vs_no_adapt"] = float(
                b_adapt.iloc[0]["reward_mean"] - b_no_adapt.iloc[0]["reward_mean"]
            )
        b_plastic_wm = sub[
            (sub["actor_adapt"])
            & (sub["plastic_wm_imagine"])
            & (sub["phase"].str.startswith("B"))
        ]
        b_frozen_wm = sub[
            (sub["actor_adapt"])
            & (~sub["plastic_wm_imagine"])
            & (sub["phase"].str.startswith("B"))
        ]
        if len(b_plastic_wm) == 1 and len(b_frozen_wm) == 1:
            row["B_reward_delta_plastic_wm_vs_frozen_wm"] = float(
                b_plastic_wm.iloc[0]["reward_mean"] - b_frozen_wm.iloc[0]["reward_mean"]
            )
            row["B_prior_mse_delta_frozen_minus_plastic_last20"] = float(
                b_frozen_wm.iloc[0]["prior_mse_last20"] - b_plastic_wm.iloc[0]["prior_mse_last20"]
            )
        b_plastic_online = sub[
            (sub["actor_adapt"])
            & (sub["plastic_online"])
            & (~sub["plastic_wm_imagine"])
            & (sub["phase"].str.startswith("B"))
        ]
        b_no_plastic_online = sub[
            (sub["actor_adapt"])
            & (~sub["plastic_online"])
            & (~sub["plastic_wm_imagine"])
            & (sub["phase"].str.startswith("B"))
        ]
        if len(b_plastic_online) == 1 and len(b_no_plastic_online) == 1:
            row["B_reward_delta_plastic_online_vs_off"] = float(
                b_plastic_online.iloc[0]["reward_mean"] - b_no_plastic_online.iloc[0]["reward_mean"]
            )
            row["B_prior_mse_delta_off_minus_online_last20"] = float(
                b_no_plastic_online.iloc[0]["prior_mse_last20"] - b_plastic_online.iloc[0]["prior_mse_last20"]
            )
        if len(row) > 1:
            rows.append(row)
    if rows:
        out = out.merge(pd.DataFrame(rows), on="seed", how="left")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--phases", type=str, default="A1:1.0:200,B:0.7:1000,A2:1.0:300")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plastic-lr", type=float, default=0.05)
    p.add_argument("--gate-mode", type=str, default="always_on")
    p.add_argument("--persist-threshold", type=float, default=0.1)
    p.add_argument("--max-trace-norm", type=float, default=2.0)
    p.add_argument("--actor-adapt", action="store_true")
    p.add_argument("--plastic-enabled", action="store_true", help="Enable online P_t during act()")
    p.add_argument("--no-plastic-enabled", action="store_true", help="Disable online P_t during act()")
    p.add_argument("--plastic-wm-imagine", action="store_true", help="Imagine on F_theta+P_t (default on if actor-adapt)")
    p.add_argument("--no-plastic-wm-imagine", action="store_true")
    p.add_argument("--adapt-phases", type=str, default="B", help="Comma-separated phase names for shift buffer")
    p.add_argument("--adapt-every", type=int, default=50)
    p.add_argument("--adapt-epochs", type=int, default=4)
    p.add_argument("--actor-lr-scale", type=float, default=1.0, help="Multiply optimizer lr for actor adapt")
    p.add_argument("--buffer-batch-size", type=int, default=4)
    p.add_argument("--buffer-batch-length", type=int, default=32)
    p.add_argument("--compare-baseline", action="store_true", help="Also run frozen actor (no adapt) with same plastic online")
    p.add_argument(
        "--ablation-wm-imagine",
        action="store_true",
        help="Run actor adapt on plastic WM vs frozen WM imagination (core MWM ablation)",
    )
    p.add_argument(
        "--ablation-online-plastic",
        action="store_true",
        help="Run B (plastic online + frozen imagine) vs C (no plastic online + frozen imagine)",
    )
    p.add_argument("--run-label", type=str, default=None)
    args = p.parse_args()

    checkpoint = args.checkpoint or (args.logdir / "latest.pt")
    outdir = args.outdir or (args.logdir / "plastic_actor_adapt")
    outdir.mkdir(parents=True, exist_ok=True)
    phases = parse_phases(args.phases)
    adapt_phase_names = {x.strip() for x in args.adapt_phases.split(",") if x.strip()}
    plastic_wm_imagine = not args.no_plastic_wm_imagine
    if args.plastic_wm_imagine:
        plastic_wm_imagine = True
    plastic_enabled = not args.no_plastic_enabled
    if args.plastic_enabled:
        plastic_enabled = True

    common = dict(
        logdir=args.logdir,
        checkpoint=checkpoint,
        outdir=outdir,
        device=args.device,
        phases=phases,
        seed=args.seed,
        plastic_lr=args.plastic_lr,
        gate_mode=args.gate_mode,
        persist_threshold=args.persist_threshold,
        max_trace_norm=args.max_trace_norm,
        adapt_phase_names=adapt_phase_names,
        adapt_every=args.adapt_every,
        adapt_epochs=args.adapt_epochs,
        actor_lr_scale=args.actor_lr_scale,
        buffer_batch_size=args.buffer_batch_size,
        buffer_batch_length=args.buffer_batch_length,
    )

    frames = []
    label = args.run_label or f"adapt_{args.gate_mode}_lr{args.plastic_lr}"

    if args.ablation_online_plastic:
        frames.append(
            run_adaptation(
                **common,
                plastic_enabled=True,
                actor_adapt=True,
                plastic_wm_imagine=False,
                run_label=f"{label}_B_plastic_online",
            )
        )
        frames.append(
            run_adaptation(
                **common,
                plastic_enabled=False,
                actor_adapt=True,
                plastic_wm_imagine=False,
                run_label=f"{label}_C_no_plastic_online",
            )
        )
    elif args.ablation_wm_imagine:
        frames.append(
            run_adaptation(
                **common,
                plastic_enabled=True,
                actor_adapt=True,
                plastic_wm_imagine=True,
                run_label=f"{label}_plastic_wm",
            )
        )
        frames.append(
            run_adaptation(
                **common,
                plastic_enabled=True,
                actor_adapt=True,
                plastic_wm_imagine=False,
                run_label=f"{label}_frozen_wm",
            )
        )
    elif args.actor_adapt:
        frames.append(
            run_adaptation(
                **common,
                plastic_enabled=plastic_enabled,
                actor_adapt=True,
                plastic_wm_imagine=plastic_wm_imagine,
                run_label=label,
            )
        )
    else:
        label = args.run_label or f"plastic_only_{args.gate_mode}"
        frames.append(
            run_adaptation(
                **common,
                plastic_enabled=True,
                actor_adapt=False,
                plastic_wm_imagine=False,
                run_label=label,
            )
        )

    if args.compare_baseline and not args.ablation_wm_imagine and not args.ablation_online_plastic:
        frames.append(
            run_adaptation(
                **common,
                plastic_enabled=True,
                actor_adapt=False,
                plastic_wm_imagine=False,
                run_label=f"{label}_no_adapt",
            )
        )

    all_steps = pd.concat(frames, ignore_index=True)
    summary = summarize_runs(all_steps)
    all_steps.to_csv(outdir / "step_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    meta = {**vars(args), "phases": [asdict(ph) for ph in phases], "adapt_phase_names": sorted(adapt_phase_names)}
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\n=== plastic_actor_adapt seed={args.seed} ===")
    cols = [
        "run_label",
        "actor_adapt",
        "plastic_online",
        "plastic_wm_imagine",
        "phase",
        "reward_mean",
        "prior_mse_last20",
        "trace_norm_max",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    if "B_reward_delta_plastic_online_vs_off" in summary.columns:
        delta = summary["B_reward_delta_plastic_online_vs_off"].dropna().iloc[0]
        print(f"\nPhase B: plastic_online - no_plastic_online reward delta = {delta:+.4f}")
    elif "B_reward_delta_plastic_wm_vs_frozen_wm" in summary.columns:
        delta = summary["B_reward_delta_plastic_wm_vs_frozen_wm"].dropna().iloc[0]
        print(f"\nPhase B: plastic_wm - frozen_wm reward delta = {delta:+.4f}")
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
