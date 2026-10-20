#!/usr/bin/env python3
"""Aggregate multi-seed plastic diagnostic runs into mean/std tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_seed_summaries(outroot: Path) -> pd.DataFrame:
    rows = []
    for config_dir in sorted(p for p in outroot.iterdir() if p.is_dir()):
        for seed_dir in sorted(config_dir.glob("seed_*")):
            summary = seed_dir / "summary.csv"
            if not summary.exists():
                continue
            df = pd.read_csv(summary)
            seed = int(seed_dir.name.split("_", 1)[1])
            df["config"] = config_dir.name
            df["seed"] = seed
            df["seed_dir"] = str(seed_dir)
            rows.append(df)
    if not rows:
        raise SystemExit(f"No seed summaries under {outroot}")
    return pd.concat(rows, ignore_index=True)


def paired_b_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, seed), g in df.groupby(["config", "seed"]):
        plastic = g[(g["use_plastic"]) & (g["phase"].str.startswith("B"))]
        frozen = g[(~g["use_plastic"]) & (g["phase"].str.startswith("B"))]
        if len(plastic) != 1 or len(frozen) != 1:
            continue
        p = plastic.iloc[0]
        f = frozen.iloc[0]
        rows.append(
            {
                "config": config,
                "seed": seed,
                "plastic_B_prior_mse_last20": float(p["prior_mse_last20"]),
                "frozen_B_prior_mse_last20": float(f["prior_mse_last20"]),
                "B_prior_mse_delta_last20": float(f["prior_mse_last20"] - p["prior_mse_last20"]),
                "plastic_B_dyn_kl_last20": float(p["dyn_kl_last20"]),
                "frozen_B_dyn_kl_last20": float(f["dyn_kl_last20"]),
                "B_dyn_kl_delta_last20": float(f["dyn_kl_last20"] - p["dyn_kl_last20"]),
                "plastic_gate_mean_B": float(p["gate_mean"]),
                "plastic_trace_norm_max_B": float(p["trace_norm_max"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_config_stats(paired: pd.DataFrame) -> pd.DataFrame:
    agg = (
        paired.groupby("config", as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            plastic_B_prior_mse_last20_mean=("plastic_B_prior_mse_last20", "mean"),
            plastic_B_prior_mse_last20_std=("plastic_B_prior_mse_last20", "std"),
            frozen_B_prior_mse_last20_mean=("frozen_B_prior_mse_last20", "mean"),
            frozen_B_prior_mse_last20_std=("frozen_B_prior_mse_last20", "std"),
            B_prior_mse_delta_last20_mean=("B_prior_mse_delta_last20", "mean"),
            B_prior_mse_delta_last20_std=("B_prior_mse_delta_last20", "std"),
            B_dyn_kl_delta_last20_mean=("B_dyn_kl_delta_last20", "mean"),
            B_dyn_kl_delta_last20_std=("B_dyn_kl_delta_last20", "std"),
            win_rate=("B_prior_mse_delta_last20", lambda s: float((s > 0).mean())),
        )
        .sort_values("B_prior_mse_delta_last20_mean", ascending=False)
    )
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", type=Path, required=True)
    args = p.parse_args()

    all_df = load_seed_summaries(args.outroot)
    all_df.to_csv(args.outroot / "all_seed_summaries.csv", index=False)

    paired = paired_b_deltas(all_df)
    paired.to_csv(args.outroot / "paired_B_by_seed.csv", index=False)

    agg = aggregate_config_stats(paired)
    agg.to_csv(args.outroot / "seed_aggregate.csv", index=False)

    print("\n=== Phase B paired deltas (frozen - plastic, positive = plastic better) ===")
    cols = [
        "config",
        "n_seeds",
        "plastic_B_prior_mse_last20_mean",
        "frozen_B_prior_mse_last20_mean",
        "B_prior_mse_delta_last20_mean",
        "B_prior_mse_delta_last20_std",
        "win_rate",
    ]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {args.outroot / 'seed_aggregate.csv'}")


if __name__ == "__main__":
    main()
