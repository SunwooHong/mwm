#!/usr/bin/env python3
"""Visualization for plastic gain-shift eval: reward plots, adaptation times, rollout videos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import tools
from plastic_eval import load_agent, obs_to_batch, summarize
from plastic_inference import PhaseSpec, parse_phases


def phase_offsets(phases: list[PhaseSpec]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cum = 0
    for ph in phases:
        offsets[ph.name] = cum
        cum += ph.steps
    return offsets


def add_global_step(df: pd.DataFrame, phases: list[PhaseSpec]) -> pd.DataFrame:
    offsets = phase_offsets(phases)
    out = df.copy()
    out["global_t"] = out["phase"].map(offsets) + out["local_t"]
    return out


def run_rollout_compare(
    logdir: Path,
    checkpoint: Path,
    outdir: Path,
    *,
    device: str,
    phases: list[PhaseSpec],
    seed: int,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    run_label: str,
    compare_frozen: bool,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    agent, env = load_agent(
        logdir,
        checkpoint,
        device,
        seed=seed,
        plastic_enabled=True,
        plastic_lr=plastic_lr,
        gate_mode=gate_mode,
        persist_threshold=persist_threshold,
        max_trace_norm=max_trace_norm,
    )
    frames = []
    for phase in phases:
        frames.append(
            _rollout_phase(
                agent,
                env,
                device=device,
                phase=phase,
                use_plastic=True,
                run_label=run_label,
                seed=seed,
            )
        )
    if compare_frozen:
        agent2, env2 = load_agent(
            logdir,
            checkpoint,
            device,
            seed=seed,
            plastic_enabled=False,
            plastic_lr=0.0,
            gate_mode="frozen",
            persist_threshold=0.0,
            max_trace_norm=0.0,
        )
        for phase in phases:
            frames.append(
                _rollout_phase(
                    agent2,
                    env2,
                    device=device,
                    phase=phase,
                    use_plastic=False,
                    run_label=f"{run_label}_with_frozen",
                    seed=seed,
                )
            )
    metrics = add_global_step(pd.concat(frames, ignore_index=True), phases)
    summary = summarize(metrics.drop(columns=["global_t"], errors="ignore"))
    metrics.to_csv(outdir / "step_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    return metrics


@torch.no_grad()
def _rollout_phase(
    agent,
    env,
    *,
    device: str,
    phase: PhaseSpec,
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


def plot_reward_overlay(
    df: pd.DataFrame,
    outdir: Path,
    *,
    phases: list[PhaseSpec],
    roll_window: int = 20,
) -> Path:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if "global_t" not in df.columns:
        df = add_global_step(df, phases)

    fig, axes = plt.subplots(2 if "prior_mse" in df.columns else 1, 1, figsize=(13, 4.5 if "prior_mse" in df.columns else 4), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    offsets = phase_offsets(phases)
    boundary_ts = [offsets[ph.name] for ph in phases[1:]]

    for use_plastic, group in df.groupby("use_plastic"):
        label = "plastic" if use_plastic else "frozen"
        color = "#2563eb" if use_plastic else "#dc2626"
        roll = group.sort_values("global_t")["reward"].rolling(roll_window, min_periods=1).mean()
        axes[0].plot(group.sort_values("global_t")["global_t"], roll, label=label, color=color, lw=2)

    for t0 in boundary_ts:
        for ax in axes:
            ax.axvline(t0, color="gray", linestyle="--", alpha=0.45)
    for ph in phases:
        t0 = offsets[ph.name]
        axes[0].text(
            t0 + 2,
            axes[0].get_ylim()[1] * 0.97,
            f"{ph.name} (gain={ph.action_gain})",
            fontsize=8,
            va="top",
        )

    axes[0].set_ylabel(f"reward (roll-{roll_window})")
    axes[0].set_title("Gain-shift adaptation: plastic vs frozen")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    if "prior_mse" in df.columns and len(axes) > 1:
        for use_plastic, group in df.groupby("use_plastic"):
            label = "plastic" if use_plastic else "frozen"
            color = "#2563eb" if use_plastic else "#dc2626"
            roll = group.sort_values("global_t")["prior_mse"].rolling(roll_window, min_periods=1).mean()
            axes[1].plot(group.sort_values("global_t")["global_t"], roll, label=label, color=color, lw=2)
        axes[1].set_ylabel(f"prior_mse (roll-{roll_window})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)

    axes[-1].set_xlabel("environment step")
    fig.tight_layout()
    out_path = plot_dir / "reward_overlay.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def compute_adaptation_times(
    df: pd.DataFrame,
    phases: list[PhaseSpec],
    *,
    roll_window: int = 20,
    recover_frac: float = 0.9,
    prior_mse_frac: float = 1.5,
) -> pd.DataFrame:
    if "global_t" not in df.columns:
        df = add_global_step(df, phases)
    offsets = phase_offsets(phases)
    b_phase = next((ph for ph in phases if ph.name.startswith("B")), None)
    a1_phase = next((ph for ph in phases if ph.name.startswith("A1")), phases[0])
    a2_phase = next((ph for ph in phases if ph.name.startswith("A2")), None)

    rows = []
    group_keys = ["run_label", "use_plastic"]
    if "seed" in df.columns:
        group_keys.append("seed")

    for key_vals, sub in df.groupby(group_keys):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(group_keys, key_vals))

        a1 = sub[sub["phase"] == a1_phase.name]
        b = sub[sub["phase"] == b_phase.name] if b_phase else pd.DataFrame()
        a2 = sub[sub["phase"] == a2_phase.name] if a2_phase else pd.DataFrame()

        baseline = float(a1["reward"].tail(roll_window).mean()) if len(a1) else float("nan")
        row["a1_reward_last20"] = baseline
        row["b_reward_mean"] = float(b["reward"].mean()) if len(b) else float("nan")
        row["b_reward_min_roll20"] = (
            float(b["reward"].rolling(roll_window, min_periods=1).mean().min()) if len(b) else float("nan")
        )

        target = recover_frac * baseline
        recover_local = float("nan")
        recover_global = float("nan")
        if len(b):
            roll = b["reward"].rolling(roll_window, min_periods=1).mean()
            hit = b[roll >= target]
            if len(hit):
                recover_local = float(hit["local_t"].iloc[0])
                recover_global = float(hit["global_t"].iloc[0])

        row[f"recover_step_{int(recover_frac * 100)}"] = recover_local
        row[f"recover_global_step_{int(recover_frac * 100)}"] = recover_global
        row["never_recovered"] = bool(len(b) and np.isnan(recover_local))

        if len(a1) and "prior_mse" in sub.columns and len(b):
            mse_baseline = float(a1["prior_mse"].tail(roll_window).mean())
            mse_target = prior_mse_frac * mse_baseline
            mse_roll = b["prior_mse"].rolling(roll_window, min_periods=1).mean()
            mse_hit = b[mse_roll <= mse_target]
            row["prior_mse_recover_step"] = (
                float(mse_hit["local_t"].iloc[0]) if len(mse_hit) else float("nan")
            )
            row["prior_mse_baseline_last20"] = mse_baseline

        if len(a1) and len(a2):
            a1_tail = float(a1["reward"].tail(roll_window).mean())
            a2_tail = float(a2["reward"].tail(roll_window).mean())
            row["a2_recovery_ratio"] = a2_tail / a1_tail if a1_tail > 0 else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def _annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 4
    for line in lines:
        draw.text((6, y), line, fill=(255, 220, 0))
        y += 14
    return np.array(img)


@torch.no_grad()
def record_rollout_video(
    logdir: Path,
    checkpoint: Path,
    outdir: Path,
    *,
    device: str,
    phases: list[PhaseSpec],
    seed: int,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    use_plastic: bool,
    label: str,
    video_phases: set[str] | None,
    max_steps: int | None,
    fps: int,
    size: tuple[int, int] | None,
) -> Path:
    from moviepy.editor import ImageSequenceClip

    outdir.mkdir(parents=True, exist_ok=True)
    agent, env = load_agent(
        logdir,
        checkpoint,
        device,
        seed=seed,
        plastic_enabled=use_plastic,
        plastic_lr=plastic_lr if use_plastic else 0.0,
        gate_mode=gate_mode if use_plastic else "frozen",
        persist_threshold=persist_threshold if use_plastic else 0.0,
        max_trace_norm=max_trace_norm if use_plastic else 0.0,
    )

    if video_phases is None:
        video_phases = {ph.name for ph in phases if ph.name.startswith("B")}

    frames: list[np.ndarray] = []
    state = agent.get_initial_state(1)
    obs = env.reset()
    recorded = 0

    for phase in phases:
        if hasattr(env, "set_gain"):
            env.set_gain(phase.action_gain)
        record_this_phase = phase.name in video_phases

        for local_t in range(phase.steps):
            if max_steps is not None and recorded >= max_steps:
                break

            obs_td = obs_to_batch(obs, device)
            for key in obs_td:
                if obs_td[key].dtype != torch.bool:
                    obs_td[key] = obs_td[key].float()

            action, state = agent.act(obs_td, state, eval=True)
            obs, reward, done, _ = env.step(tools.to_np(action)[0])

            if record_this_phase and "image" in obs:
                frame = np.asarray(obs["image"])
                if size is not None:
                    frame = np.array(Image.fromarray(frame).resize(size, Image.BILINEAR))
                trace_norm = float(agent.last_plastic_metrics.get("trace_norm", 0.0))
                lines = [
                    f"{label} | {phase.name} gain={phase.action_gain}",
                    f"step={local_t} reward={reward:.3f}",
                ]
                if use_plastic:
                    lines.append(f"trace_norm={trace_norm:.3f}")
                frames.append(_annotate_frame(frame, lines))
                recorded += 1

            if done:
                obs = env.reset()
                state = agent.get_initial_state(1)

        if max_steps is not None and recorded >= max_steps:
            break

    if not frames:
        raise RuntimeError("No frames captured; check --video-phases and env image observations.")

    video_path = outdir / f"rollout_{label}.mp4"
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(video_path), codec="libx264", audio=False, logger=None)
    return video_path


def record_comparison_videos(
    logdir: Path,
    checkpoint: Path,
    outdir: Path,
    *,
    device: str,
    phases: list[PhaseSpec],
    seed: int,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    video_phases: set[str] | None,
    max_steps: int | None,
    fps: int,
    size: tuple[int, int] | None,
) -> list[Path]:
    video_dir = outdir / "videos"
    paths = []
    for use_plastic, label in [(True, "plastic"), (False, "frozen")]:
        paths.append(
            record_rollout_video(
                logdir,
                checkpoint,
                video_dir,
                device=device,
                phases=phases,
                seed=seed,
                plastic_lr=plastic_lr,
                gate_mode=gate_mode,
                persist_threshold=persist_threshold,
                max_trace_norm=max_trace_norm,
                use_plastic=use_plastic,
                label=label,
                video_phases=video_phases,
                max_steps=max_steps,
                fps=fps,
                size=size,
            )
        )
    return paths


def _resolve_outdir(args: argparse.Namespace) -> Path:
    if args.outdir is not None:
        return args.outdir
    if args.csv is not None:
        return args.csv.parent
    if args.logdir is not None:
        return args.logdir / "plastic_visualize"
    raise SystemExit("Provide --outdir, --logdir, or --csv")


def _load_csv_or_rollout(args, phases: list[PhaseSpec]) -> tuple[pd.DataFrame, Path]:
    outdir = _resolve_outdir(args)
    outdir.mkdir(parents=True, exist_ok=True)

    run_label = args.run_label or f"{args.gate_mode}_lr{args.plastic_lr}"

    if args.csv is None or getattr(args, "force_rollout", False):
        if args.logdir is None:
            raise SystemExit("--logdir is required to run rollout")
        checkpoint = args.checkpoint or (args.logdir / "latest.pt")
        df = run_rollout_compare(
            args.logdir,
            checkpoint,
            outdir,
            device=args.device,
            phases=phases,
            seed=args.seed,
            plastic_lr=args.plastic_lr,
            gate_mode=args.gate_mode,
            persist_threshold=args.persist_threshold,
            max_trace_norm=args.max_trace_norm,
            run_label=run_label,
            compare_frozen=args.compare_frozen,
        )
    else:
        df = pd.read_csv(args.csv)
        if "global_t" not in df.columns:
            df = add_global_step(df, phases)
    return df, outdir


def _parse_video_phases(text: str | None, phases: list[PhaseSpec]) -> set[str] | None:
    if text is None:
        return {ph.name for ph in phases if ph.name.startswith("B")}
    if text.strip().lower() == "all":
        return {ph.name for ph in phases}
    return {x.strip() for x in text.split(",") if x.strip()}


def _parse_size(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return (256, 256)
    w, h = text.split("x")
    return int(w), int(h)


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--logdir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None, help="Existing step_metrics.csv (skip rollout)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--phases", type=str, default="A1:1.0:200,B:0.7:1000,A2:1.0:300")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plastic-lr", type=float, default=0.05)
    p.add_argument("--gate-mode", type=str, default="always_on", choices=["selective", "surprise_only", "always_on"])
    p.add_argument("--persist-threshold", type=float, default=0.1)
    p.add_argument("--max-trace-norm", type=float, default=2.0)
    p.add_argument("--run-label", type=str, default=None)
    p.add_argument("--compare-frozen", action="store_true", default=True)
    p.add_argument("--no-compare-frozen", action="store_false", dest="compare_frozen")
    p.add_argument("--roll-window", type=int, default=20)
    p.add_argument("--recover-frac", type=float, default=0.9)


def cmd_plot(args: argparse.Namespace) -> None:
    phases = parse_phases(args.phases)
    df, outdir = _load_csv_or_rollout(args, phases)
    path = plot_reward_overlay(df, outdir, phases=phases, roll_window=args.roll_window)
    print(f"Wrote {path}")


def cmd_adapt(args: argparse.Namespace) -> None:
    phases = parse_phases(args.phases)
    df, outdir = _load_csv_or_rollout(args, phases)
    adapt = compute_adaptation_times(
        df, phases, roll_window=args.roll_window, recover_frac=args.recover_frac
    )
    adapt_path = outdir / "adaptation_times.csv"
    adapt.to_csv(adapt_path, index=False)
    print(f"\n=== Adaptation times (recover_frac={args.recover_frac}) ===")
    cols = [c for c in adapt.columns if c in adapt.columns]
    show = [
        "run_label",
        "use_plastic",
        "seed",
        "a1_reward_last20",
        "b_reward_mean",
        f"recover_step_{int(args.recover_frac * 100)}",
        "never_recovered",
        "a2_recovery_ratio",
        "prior_mse_recover_step",
    ]
    show = [c for c in show if c in adapt.columns]
    print(adapt[show].to_string(index=False))
    print(f"\nWrote {adapt_path}")


def cmd_video(args: argparse.Namespace) -> None:
    if args.logdir is None:
        raise SystemExit("--logdir is required for video")
    phases = parse_phases(args.phases)
    checkpoint = args.checkpoint or (args.logdir / "latest.pt")
    outdir = args.outdir or (args.logdir / "plastic_visualize")
    video_phases = _parse_video_phases(args.video_phases, phases)
    paths = record_comparison_videos(
        args.logdir,
        checkpoint,
        outdir,
        device=args.device,
        phases=phases,
        seed=args.seed,
        plastic_lr=args.plastic_lr,
        gate_mode=args.gate_mode,
        persist_threshold=args.persist_threshold,
        max_trace_norm=args.max_trace_norm,
        video_phases=video_phases,
        max_steps=args.video_max_steps,
        fps=args.fps,
        size=_parse_size(args.video_size),
    )
    for p in paths:
        print(f"Wrote {p}")


def cmd_all(args: argparse.Namespace) -> None:
    if args.logdir is None:
        raise SystemExit("--logdir is required")
    args.force_rollout = True
    phases = parse_phases(args.phases)
    df, outdir = _load_csv_or_rollout(args, phases)

    plot_path = plot_reward_overlay(df, outdir, phases=phases, roll_window=args.roll_window)
    adapt = compute_adaptation_times(
        df, phases, roll_window=args.roll_window, recover_frac=args.recover_frac
    )
    adapt_path = outdir / "adaptation_times.csv"
    adapt.to_csv(adapt_path, index=False)

    checkpoint = args.checkpoint or (args.logdir / "latest.pt")
    video_phases = _parse_video_phases(args.video_phases, phases)
    video_paths = record_comparison_videos(
        args.logdir,
        checkpoint,
        outdir,
        device=args.device,
        phases=phases,
        seed=args.seed,
        plastic_lr=args.plastic_lr,
        gate_mode=args.gate_mode,
        persist_threshold=args.persist_threshold,
        max_trace_norm=args.max_trace_norm,
        video_phases=video_phases,
        max_steps=args.video_max_steps,
        fps=args.fps,
        size=_parse_size(args.video_size),
    )

    meta = {**vars(args), "phases": [asdict(ph) for ph in phases]}
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote plot: {plot_path}")
    print(f"Wrote adaptation: {adapt_path}")
    for p in video_paths:
        print(f"Wrote video: {p}")
    print(f"\nDone. Output dir: {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plastic eval visualization toolkit")
    parser.add_argument("--force-rollout", action="store_true", help="Re-run rollout even if --csv is set")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plot = sub.add_parser("plot", help="Reward overlay plot (plastic vs frozen)")
    add_shared_args(p_plot)
    p_plot.set_defaults(func=cmd_plot)

    p_adapt = sub.add_parser("adapt", help="Compute adaptation-time metrics")
    add_shared_args(p_adapt)
    p_adapt.set_defaults(func=cmd_adapt)

    p_video = sub.add_parser("video", help="Record rollout MP4s")
    add_shared_args(p_video)
    p_video.add_argument("--video-phases", type=str, default=None, help="Comma phases or 'all' (default: B*)")
    p_video.add_argument("--video-max-steps", type=int, default=400)
    p_video.add_argument("--fps", type=int, default=30)
    p_video.add_argument("--video-size", type=str, default="256x256")
    p_video.set_defaults(func=cmd_video)

    p_all = sub.add_parser("all", help="Rollout + plot + adapt + video")
    add_shared_args(p_all)
    p_all.add_argument("--video-phases", type=str, default=None)
    p_all.add_argument("--video-max-steps", type=int, default=400)
    p_all.add_argument("--fps", type=int, default=30)
    p_all.add_argument("--video-size", type=str, default="256x256")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
