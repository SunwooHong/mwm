#!/usr/bin/env python3
"""Test-time consolidation: Slow θ fine-tune + decay-boost flush of Fast P_t.

Conditions (A/B/C/D):
  A  baseline: plastic online + cfg0 actor adapt (no slow θ, no flush)
  B  slow θ only: + background WM fine-tune
  C  consolidate: slow θ + coupled decay-boost flush (target)
  D  trap control: flush without slow θ (oscillation negative control)
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
from torch.utils.tensorboard import SummaryWriter

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_env
from envs import wrappers
from plastic import LogitPlasticResidual
from plastic_inference import parse_phases

CONDITIONS = {
    "A": dict(actor_adapt=True, slow_wm=False, flush_mode="off"),
    "B": dict(actor_adapt=True, slow_wm=True, flush_mode="off"),
    "C": dict(actor_adapt=True, slow_wm=True, flush_mode="coupled"),
    "D": dict(actor_adapt=False, slow_wm=False, flush_mode="error_only"),
}


def load_agent(
    logdir: Path,
    checkpoint: Path,
    device: str,
    *,
    seed: int,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    actor_adapt: bool,
    slow_wm: bool,
    flush_mode: str,
    actor_lr_scale: float,
    wm_lr_scale: float,
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
    cfg.model.plastic.enabled = True
    cfg.model.plastic.inference = True
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
        grad = (actor_adapt and name.startswith("actor.")) or (
            slow_wm and name.startswith(("encoder.", "rssm.", "decoder.", "projector."))
        )
        param.requires_grad_(grad)

    base_lr = float(agent._optimizer.param_groups[0]["lr"])
    agent._opt_base_lr = base_lr
    agent._actor_lr_scale = actor_lr_scale
    agent._wm_lr_scale = wm_lr_scale

    agent.clone_and_freeze()
    agent.set_plastic_inference(True)
    agent.set_consolidation_mode(flush_mode)
    agent.train()
    agent._slow_value.train(False)
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
    return buf.count() >= batch_size * (batch_length + 1)


def run_consolidation(
    *,
    logdir: Path,
    checkpoint: Path,
    outdir: Path,
    device: str,
    phases,
    seed: int,
    condition: str,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    adapt_phase_names: set[str],
    adapt_every: int,
    adapt_epochs: int,
    slow_every: int,
    slow_epochs: int,
    actor_lr_scale: float,
    wm_lr_scale: float,
    buffer_batch_size: int,
    buffer_batch_length: int,
    global_step: int = 0,
) -> tuple[pd.DataFrame, int]:
    spec = CONDITIONS[condition]
    agent, env = load_agent(
        logdir,
        checkpoint,
        device,
        seed=seed,
        plastic_lr=plastic_lr,
        gate_mode=gate_mode,
        persist_threshold=persist_threshold,
        max_trace_norm=max_trace_norm,
        actor_adapt=spec["actor_adapt"],
        slow_wm=spec["slow_wm"],
        flush_mode=spec["flush_mode"],
        actor_lr_scale=actor_lr_scale,
        wm_lr_scale=wm_lr_scale,
    )

    tb_dir = outdir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    shift_buffer = make_shift_buffer(device, buffer_batch_size, buffer_batch_length)
    state = agent.get_initial_state(1)
    obs = env.reset()
    episode_id = 0
    rows = []
    adapt_logs = []
    slow_logs = []

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

            if collect_here and (spec["actor_adapt"] or spec["slow_wm"]):
                shift_buffer.add_transition(
                    transition_from_step(obs_td, action, state, reward, done, episode_id)
                )

            ready = buffer_ready(shift_buffer, buffer_batch_size, buffer_batch_length)

            if (
                collect_here
                and spec["slow_wm"]
                and slow_every > 0
                and local_t > 0
                and (local_t % slow_every) == 0
                and ready
            ):
                for _ in range(slow_epochs):
                    slow_mets = agent.update_slow_wm(shift_buffer)
                slow_logs.append(
                    {
                        "phase": phase.name,
                        "local_t": local_t,
                        **{k: float(v.detach()) if torch.is_tensor(v) else v for k, v in slow_mets.items()},
                    }
                )

            if (
                collect_here
                and spec["actor_adapt"]
                and adapt_every > 0
                and local_t > 0
                and (local_t % adapt_every) == 0
                and ready
            ):
                trace_snapshot = LogitPlasticResidual.clone_trace(agent._unpack_trace(state))
                for _ in range(adapt_epochs):
                    adapt_mets = agent.update_actor_plastic(
                        shift_buffer, trace_snapshot, use_plastic_wm=False
                    )
                adapt_logs.append(
                    {
                        "phase": phase.name,
                        "local_t": local_t,
                        **{k: float(v.detach()) if torch.is_tensor(v) else v for k, v in adapt_mets.items()},
                    }
                )

            if done:
                obs = env.reset()
                state = agent.get_initial_state(1)
                episode_id += 1

            row = {
                "condition": condition,
                "seed": seed,
                "phase_idx": phase.phase_idx,
                "phase": phase.name,
                "action_gain": phase.action_gain,
                "local_t": local_t,
                "global_step": global_step,
                "reward": float(reward),
                "buffer_count": shift_buffer.count() if collect_here else 0,
                **metrics,
            }
            rows.append(row)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"{condition}/{phase.name}/{k}", v, global_step)
            writer.add_scalar(f"{condition}/{phase.name}/reward", float(reward), global_step)
            if slow_logs and phase.name.startswith("B"):
                last_slow = slow_logs[-1]
                if "slow/dyn_loss" in last_slow:
                    writer.add_scalar(f"{condition}/slow/dyn_loss", last_slow["slow/dyn_loss"], global_step)
                if "slow/flush_boost" in last_slow:
                    writer.add_scalar(f"{condition}/slow/flush_boost", last_slow["slow/flush_boost"], global_step)
            global_step += 1

    writer.close()
    step_df = pd.DataFrame(rows)
    label = f"cond_{condition}_seed{seed}"
    step_df.to_csv(outdir / f"step_metrics_{label}.csv", index=False)
    if adapt_logs:
        pd.DataFrame(adapt_logs).to_csv(outdir / f"adapt_log_{label}.csv", index=False)
    if slow_logs:
        pd.DataFrame(slow_logs).to_csv(outdir / f"slow_log_{label}.csv", index=False)
    return step_df, global_step


def summarize(step_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "seed", "phase_idx", "phase", "action_gain"]
    agg = {
        "reward_mean": ("reward", "mean"),
        "steps": ("reward", "count"),
    }
    if "prior_mse" in step_df.columns:
        agg["prior_mse_mean"] = ("prior_mse", "mean")
        agg["prior_mse_last20"] = ("prior_mse", lambda s: float(s.tail(20).mean()))
    if "trace_norm" in step_df.columns:
        agg["trace_norm_mean"] = ("trace_norm", "mean")
        agg["trace_norm_last20"] = ("trace_norm", lambda s: float(s.tail(20).mean()))
        agg["trace_norm_max"] = ("trace_norm", "max")
    if "lambda_eff" in step_df.columns:
        agg["lambda_eff_last20"] = ("lambda_eff", lambda s: float(s.tail(20).mean()))
    return step_df.groupby(keys, as_index=False).agg(**agg).sort_values(keys)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--phases", type=str, default="A1:1.0:200,B:0.7:1000,A2:1.0:300")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", type=str, default="A,B,C,D")
    p.add_argument("--plastic-lr", type=float, default=0.05)
    p.add_argument("--gate-mode", type=str, default="always_on")
    p.add_argument("--persist-threshold", type=float, default=0.1)
    p.add_argument("--max-trace-norm", type=float, default=2.0)
    p.add_argument("--adapt-phases", type=str, default="B")
    p.add_argument("--adapt-every", type=int, default=50)
    p.add_argument("--adapt-epochs", type=int, default=4)
    p.add_argument("--actor-lr-scale", type=float, default=1.0)
    p.add_argument("--slow-every", type=int, default=100)
    p.add_argument("--slow-epochs", type=int, default=1)
    p.add_argument("--wm-lr-scale", type=float, default=0.02)
    p.add_argument("--buffer-batch-size", type=int, default=4)
    p.add_argument("--buffer-batch-length", type=int, default=32)
    args = p.parse_args()

    checkpoint = args.checkpoint or (args.logdir / "latest.pt")
    outdir = args.outdir or (args.logdir / "plastic_consolidation")
    outdir.mkdir(parents=True, exist_ok=True)
    phases = parse_phases(args.phases)
    adapt_phase_names = {x.strip() for x in args.adapt_phases.split(",") if x.strip()}
    conds = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]

    frames = []
    gstep = 0
    for cond in conds:
        if cond not in CONDITIONS:
            raise SystemExit(f"Unknown condition {cond}; use A,B,C,D")
        print(f"\n=== consolidation condition={cond} seed={args.seed} ===")
        df, gstep = run_consolidation(
            logdir=args.logdir,
            checkpoint=checkpoint,
            outdir=outdir,
            device=args.device,
            phases=phases,
            seed=args.seed,
            condition=cond,
            plastic_lr=args.plastic_lr,
            gate_mode=args.gate_mode,
            persist_threshold=args.persist_threshold,
            max_trace_norm=args.max_trace_norm,
            adapt_phase_names=adapt_phase_names,
            adapt_every=args.adapt_every,
            adapt_epochs=args.adapt_epochs,
            slow_every=args.slow_every,
            slow_epochs=args.slow_epochs,
            actor_lr_scale=args.actor_lr_scale,
            wm_lr_scale=args.wm_lr_scale,
            buffer_batch_size=args.buffer_batch_size,
            buffer_batch_length=args.buffer_batch_length,
            global_step=gstep,
        )
        frames.append(df)

    all_steps = pd.concat(frames, ignore_index=True)
    summary = summarize(all_steps)
    all_steps.to_csv(outdir / "step_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    meta = {**vars(args), "phases": [asdict(ph) for ph in phases], "conditions": conds}
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    b = summary[summary["phase"].str.startswith("B")]
    cols = ["condition", "seed", "reward_mean", "trace_norm_last20", "prior_mse_last20"]
    cols = [c for c in cols if c in b.columns]
    print("\n=== Phase B summary ===")
    print(b[cols].to_string(index=False))
    print(f"\nWrote {outdir}")
    print(f"TensorBoard: tensorboard --logdir {outdir / 'tensorboard'}")


if __name__ == "__main__":
    main()
