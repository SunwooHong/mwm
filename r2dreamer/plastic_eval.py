#!/usr/bin/env python3
"""Evaluate Dreamer.act() with optional test-time plastic under action-gain shifts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

import tools
from dreamer import Dreamer
from envs import make_env
from envs import wrappers
from plastic_inference import parse_phases


def load_agent(
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
) -> Dreamer:
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
    agent.eval()
    for param in agent.parameters():
        param.requires_grad = False
    agent.clone_and_freeze()
    agent.set_plastic_inference(plastic_enabled)
    return agent, env


def obs_to_batch(obs: dict, device: str) -> dict:
    out = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
    if "is_first" not in out:
        out["is_first"] = torch.zeros(1, dtype=torch.bool, device=device)
    return out


@torch.no_grad()
def rollout_phase(
    agent: Dreamer,
    env,
    *,
    device: str,
    phase,
    use_plastic: bool,
    run_label: str,
    seed: int,
) -> pd.DataFrame:
    if hasattr(env, "set_gain"):
        env.set_gain(phase.action_gain)
    agent.set_plastic_inference(use_plastic)

    state = agent.get_initial_state(1)
    obs = env.reset()
    rows = []

    for local_t in range(phase.steps):
        obs_td = obs_to_batch(obs, device)
        for key in obs_td:
            if obs_td[key].dtype != torch.bool:
                obs_td[key] = obs_td[key].float()

        action, state = agent.act(obs_td, state, eval=True)
        metrics = dict(agent.last_plastic_metrics)

        obs, reward, done, _ = env.step(tools.to_np(action)[0])
        if done:
            obs = env.reset()
            state = agent.get_initial_state(1)

        rows.append(
            {
                "run_label": run_label,
                "seed": seed,
                "use_plastic": use_plastic,
                "phase_idx": phase.phase_idx,
                "phase": phase.name,
                "action_gain": phase.action_gain,
                "local_t": local_t,
                "reward": float(reward),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_label", "use_plastic", "seed", "phase_idx", "phase", "action_gain"]
    agg = {
        "reward_mean": ("reward", "mean"),
        "reward_sum": ("reward", "sum"),
        "steps": ("reward", "count"),
    }
    if "prior_mse" in df.columns:
        agg["prior_mse_mean"] = ("prior_mse", "mean")
        agg["prior_mse_last20"] = ("prior_mse", lambda s: float(s.tail(20).mean()) if len(s) else float("nan"))
    if "gate" in df.columns:
        agg["gate_mean"] = ("gate", "mean")
    if "trace_norm" in df.columns:
        agg["trace_norm_max"] = ("trace_norm", "max")
    out = df.groupby(keys, as_index=False).agg(**agg).sort_values(keys)

    rows = []
    for seed in out["seed"].unique():
        sub = out[out["seed"] == seed]
        plastic_b = sub[(sub["use_plastic"]) & (sub["phase"].str.startswith("B"))]
        frozen_b = sub[(~sub["use_plastic"]) & (sub["phase"].str.startswith("B"))]
        if len(plastic_b) != 1 or len(frozen_b) != 1:
            continue
        p, f = plastic_b.iloc[0], frozen_b.iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "B_reward_delta": float(p["reward_mean"] - f["reward_mean"]),
                "B_prior_mse_delta_last20": float(f["prior_mse_last20"] - p["prior_mse_last20"]),
            }
        )
    if rows:
        delta = pd.DataFrame(rows)
        out = out.merge(delta, on="seed", how="left")
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
    p.add_argument("--gate-mode", type=str, default="always_on", choices=["selective", "surprise_only", "always_on"])
    p.add_argument("--persist-threshold", type=float, default=0.1)
    p.add_argument("--max-trace-norm", type=float, default=2.0)
    p.add_argument("--compare-frozen", action="store_true")
    p.add_argument("--run-label", type=str, default=None)
    args = p.parse_args()

    checkpoint = args.checkpoint or (args.logdir / "latest.pt")
    outdir = args.outdir or (args.logdir / "plastic_act_eval")
    outdir.mkdir(parents=True, exist_ok=True)
    phases = parse_phases(args.phases)
    run_label = args.run_label or f"{args.gate_mode}_lr{args.plastic_lr}"

    agent, env = load_agent(
        args.logdir,
        checkpoint,
        args.device,
        seed=args.seed,
        plastic_enabled=True,
        plastic_lr=args.plastic_lr,
        gate_mode=args.gate_mode,
        persist_threshold=args.persist_threshold,
        max_trace_norm=args.max_trace_norm,
    )
    frames = [
        rollout_phase(
            agent,
            env,
            device=args.device,
            phase=phase,
            use_plastic=True,
            run_label=run_label,
            seed=args.seed,
        )
        for phase in phases
    ]

    if args.compare_frozen:
        agent2, env2 = load_agent(
            args.logdir,
            checkpoint,
            args.device,
            seed=args.seed,
            plastic_enabled=False,
            plastic_lr=0.0,
            gate_mode="frozen",
            persist_threshold=0.0,
            max_trace_norm=0.0,
        )
        frames.append(
            pd.concat(
                [
                    rollout_phase(
                        agent2,
                        env2,
                        device=args.device,
                        phase=phase,
                        use_plastic=False,
                        run_label=f"{run_label}_with_frozen",
                        seed=args.seed,
                    )
                    for phase in phases
                ],
                ignore_index=True,
            )
        )

    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize(metrics)
    metrics.to_csv(outdir / "step_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    meta = {**vars(args), "phases": [asdict(p) for p in phases], "run_label": run_label}
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\n=== act() eval: {run_label} seed={args.seed} ===")
    cols = ["run_label", "use_plastic", "phase", "reward_mean", "prior_mse_last20"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    if "B_reward_delta" in summary.columns:
        b = summary.drop_duplicates("seed")[["seed", "B_reward_delta", "B_prior_mse_delta_last20"]]
        print("\nPhase B paired deltas (seed-level):")
        print(b.to_string(index=False))
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
