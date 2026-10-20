#!/usr/bin/env python3
"""Aggregate Condition B ratio sweep (adapt_every × adapt_epochs × actor_lr_scale)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_summaries(outroot: Path) -> pd.DataFrame:
    rows = []
    config_dirs = sorted(outroot.glob("cfg*"))
    no_adapt = outroot / "B_no_adapt"
    if no_adapt.is_dir():
        config_dirs.append(no_adapt)
    for cfg_dir in config_dirs:
        for seed_dir in sorted(cfg_dir.glob("seed_*")):
            summary = seed_dir / "summary.csv"
            if not summary.exists():
                continue
            df = pd.read_csv(summary)
            df["config"] = cfg_dir.name
            df["seed"] = int(seed_dir.name.split("_", 1)[1])
            rows.append(df)
    if not rows:
        raise SystemExit(f"No ratio sweep summaries under {outroot}")
    return pd.concat(rows, ignore_index=True)


def load_run_hparams(outroot: Path, config: str, seed: int) -> dict:
    cfg_path = outroot / config / f"seed_{seed}" / "run_config.json"
    if not cfg_path.exists():
        return {}
    meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
        "adapt_every": meta.get("adapt_every"),
        "adapt_epochs": meta.get("adapt_epochs"),
        "actor_lr_scale": meta.get("actor_lr_scale"),
    }


def load_adapt_stats(outroot: Path, config: str, seed: int) -> dict:
    seed_dir = outroot / config / f"seed_{seed}"
    logs = list(seed_dir.glob("adapt_log_*.csv"))
    if not logs:
        return {}
    df = pd.read_csv(logs[0])
    lp = df["adapt/loss_policy"]
    sign_flips = int((np.sign(lp).diff().fillna(0) != 0).sum())
    return {
        "n_adapt_updates": len(df),
        "loss_policy_std": float(lp.std()),
        "loss_policy_abs_mean": float(lp.abs().mean()),
        "loss_policy_sign_flips": sign_flips,
        "adv_abs_mean": float(df["adapt/adv"].abs().mean()) if "adapt/adv" in df.columns else float("nan"),
    }


def load_act_baselines(logdir: Path) -> pd.DataFrame:
    rows = []
    root = logdir / "plastic_act_seed_sweep"
    for seed_dir in sorted(root.glob("seed_*")):
        summary = seed_dir / "summary.csv"
        if not summary.exists():
            continue
        df = pd.read_csv(summary)
        b = df[df["phase"] == "B"]
        plastic = b[b["use_plastic"]]
        if len(plastic) != 1:
            continue
        rows.append(
            {
                "seed": int(seed_dir.name.split("_", 1)[1]),
                "plastic_act_B_reward": float(plastic.iloc[0]["reward_mean"]),
                "plastic_act_B_prior_mse_last20": float(plastic.iloc[0]["prior_mse_last20"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", type=Path, required=True)
    p.add_argument("--logdir", type=Path, default=None)
    args = p.parse_args()

    all_df = load_summaries(args.outroot)
    all_df.to_csv(args.outroot / "all_summaries.csv", index=False)

    b = all_df[all_df["phase"].str.startswith("B")].copy()
    paired_rows = []
    for (config, seed), g in b.groupby(["config", "seed"]):
        if len(g) != 1:
            continue
        row = g.iloc[0].to_dict()
        row.update(load_run_hparams(args.outroot, config, int(seed)))
        row.update(load_adapt_stats(args.outroot, config, int(seed)))
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    for col in (
        "n_adapt_updates",
        "loss_policy_std",
        "loss_policy_abs_mean",
        "loss_policy_sign_flips",
        "adapt_every",
        "adapt_epochs",
        "actor_lr_scale",
    ):
        if col not in paired.columns:
            paired[col] = float("nan")

    if args.logdir is not None:
        act = load_act_baselines(args.logdir)
        if len(act):
            paired = paired.merge(act, on="seed", how="left")
            paired["delta_vs_plastic_act"] = paired["reward_mean"] - paired["plastic_act_B_reward"]

    no_adapt = paired[paired["config"] == "B_no_adapt"][["seed", "reward_mean"]].rename(
        columns={"reward_mean": "no_adapt_B_reward"}
    )
    if len(no_adapt):
        paired = paired.merge(no_adapt, on="seed", how="left")
        paired["delta_vs_no_adapt"] = paired["reward_mean"] - paired["no_adapt_B_reward"]

    paired.to_csv(args.outroot / "paired_B_by_config_seed.csv", index=False)

    agg = (
        paired.groupby("config", as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            B_reward_mean=("reward_mean", "mean"),
            B_reward_std=("reward_mean", "std"),
            B_reward_max=("reward_mean", "max"),
            B_reward_min=("reward_mean", "min"),
            prior_mse_last20_mean=("prior_mse_last20", "mean"),
            n_adapt_updates_mean=("n_adapt_updates", "mean"),
            loss_policy_std_mean=("loss_policy_std", "mean"),
            loss_policy_abs_mean=("loss_policy_abs_mean", "mean"),
            loss_sign_flips_mean=("loss_policy_sign_flips", "mean"),
            adapt_every=("adapt_every", "first"),
            adapt_epochs=("adapt_epochs", "first"),
            actor_lr_scale=("actor_lr_scale", "first"),
        )
        .sort_values("B_reward_mean", ascending=False)
    )
    if "delta_vs_plastic_act" in paired.columns:
        delta_agg = paired.groupby("config")["delta_vs_plastic_act"].mean().reset_index()
        delta_agg.columns = ["config", "delta_vs_plastic_act_mean"]
        win_agg = (
            paired.groupby("config")["delta_vs_plastic_act"]
            .apply(lambda s: float((s > 0).mean()))
            .reset_index(name="beats_plastic_act_rate")
        )
        agg = agg.merge(delta_agg, on="config", how="left")
        agg = agg.merge(win_agg, on="config", how="left")
    if "delta_vs_no_adapt" in paired.columns:
        adapt_only = paired[paired["config"] != "B_no_adapt"]
        no_delta = adapt_only.groupby("config")["delta_vs_no_adapt"].mean().reset_index()
        no_delta.columns = ["config", "delta_vs_no_adapt_mean"]
        no_win = (
            adapt_only.groupby("config")["delta_vs_no_adapt"]
            .apply(lambda s: float((s > 0).mean()))
            .reset_index(name="beats_no_adapt_rate")
        )
        agg = agg.merge(no_delta, on="config", how="left")
        agg = agg.merge(no_win, on="config", how="left")

    agg.to_csv(args.outroot / "ratio_aggregate.csv", index=False)

    print("\n=== Condition B ratio sweep (Phase B reward, higher is better) ===")
    show = [
        "config",
        "B_reward_mean",
        "B_reward_max",
        "B_reward_min",
        "prior_mse_last20_mean",
        "loss_policy_std_mean",
        "loss_sign_flips_mean",
        "n_adapt_updates_mean",
    ]
    if "delta_vs_plastic_act_mean" in agg.columns:
        show.append("delta_vs_plastic_act_mean")
        show.append("beats_plastic_act_rate")
    if "delta_vs_no_adapt_mean" in agg.columns:
        show.append("delta_vs_no_adapt_mean")
        show.append("beats_no_adapt_rate")
    show = [c for c in show if c in agg.columns]
    print(agg[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Per seed ===")
    cols = ["config", "seed", "reward_mean", "prior_mse_last20", "loss_policy_std", "loss_policy_sign_flips"]
    if "plastic_act_B_reward" in paired.columns:
        cols.append("plastic_act_B_reward")
        cols.append("delta_vs_plastic_act")
    if "no_adapt_B_reward" in paired.columns:
        cols.append("no_adapt_B_reward")
        cols.append("delta_vs_no_adapt")
    cols = [c for c in cols if c in paired.columns]
    print(paired[cols].sort_values(["config", "seed"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nWrote {args.outroot / 'ratio_aggregate.csv'}")


if __name__ == "__main__":
    main()
