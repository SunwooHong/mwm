#!/usr/bin/env python3
"""CEM planner eval: frozen WM vs plastic WM under action-gain shifts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

import tools
from plastic_eval import load_agent, obs_to_batch, summarize
from plastic_inference import parse_phases


@torch.no_grad()
def _rollout_phase(agent, env, *, device, phase, use_plastic_wm, run_label, seed, cem_kwargs):

    if hasattr(env, "set_gain"):
        env.set_gain(phase.action_gain)

    state = agent.get_initial_state(1)
    obs = env.reset()
    rows = []

    for local_t in range(phase.steps):
        obs_td = obs_to_batch(obs, device)
        for key in obs_td:
            if obs_td[key].dtype != torch.bool:
                obs_td[key] = obs_td[key].float()

        action, state = agent.act_planner(obs_td, state, use_plastic_wm=use_plastic_wm, **cem_kwargs)
        metrics = dict(agent.last_plastic_metrics)

        obs, reward, done, _ = env.step(tools.to_np(action)[0])
        if done:
            obs = env.reset()
            state = agent.get_initial_state(1)

        rows.append(
            {
                "run_label": run_label,
                "seed": seed,
                "use_plastic_wm": use_plastic_wm,
                "phase_idx": phase.phase_idx,
                "phase": phase.name,
                "action_gain": phase.action_gain,
                "local_t": local_t,
                "reward": float(reward),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


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
    p.add_argument("--compare-frozen", action="store_true")
    p.add_argument("--run-label", type=str, default=None)
    p.add_argument("--cem-horizon", type=int, default=12)
    p.add_argument("--cem-samples", type=int, default=256)
    p.add_argument("--cem-elites", type=int, default=32)
    p.add_argument("--cem-iterations", type=int, default=5)
    args = p.parse_args()

    checkpoint = args.checkpoint or (args.logdir / "latest.pt")
    outdir = args.outdir or (args.logdir / "plastic_planner_eval")
    outdir.mkdir(parents=True, exist_ok=True)
    phases = parse_phases(args.phases)
    run_label = args.run_label or f"cem_{args.gate_mode}_lr{args.plastic_lr}"
    cem_kwargs = {
        "cem_horizon": args.cem_horizon,
        "cem_samples": args.cem_samples,
        "cem_elites": args.cem_elites,
        "cem_iterations": args.cem_iterations,
    }

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
        _rollout_phase(
            agent,
            env,
            device=args.device,
            phase=phase,
            use_plastic_wm=True,
            run_label=run_label,
            seed=args.seed,
            cem_kwargs=cem_kwargs,
        )
        for phase in phases
    ]

    if args.compare_frozen:
        agent2, env2 = load_agent(
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
        frames.append(
            pd.concat(
                [
                    _rollout_phase(
                        agent2,
                        env2,
                        device=args.device,
                        phase=phase,
                        use_plastic_wm=False,
                        run_label=f"{run_label}_frozen_wm",
                        seed=args.seed,
                        cem_kwargs=cem_kwargs,
                    )
                    for phase in phases
                ],
                ignore_index=True,
            )
        )

    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize(metrics.rename(columns={"use_plastic_wm": "use_plastic"}))
    metrics.to_csv(outdir / "step_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    meta = {**vars(args), "phases": [asdict(ph) for ph in phases], "run_label": run_label}
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\n=== CEM planner eval: {run_label} seed={args.seed} ===")
    cols = ["run_label", "use_plastic", "phase", "reward_mean", "prior_mse_last20"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
