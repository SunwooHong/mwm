#!/usr/bin/env python3
"""Aggregate multi-seed Dreamer.act() plastic eval runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_summaries(outroot: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(outroot.glob("seed_*")):
        summary = seed_dir / "summary.csv"
        if not summary.exists():
            continue
        df = pd.read_csv(summary)
        df["seed"] = int(seed_dir.name.split("_", 1)[1])
        rows.append(df)
    if not rows:
        raise SystemExit(f"No act-eval summaries under {outroot}")
    return pd.concat(rows, ignore_index=True)


def paired_b(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, g in df.groupby("seed"):
        plastic = g[(g["use_plastic"]) & (g["phase"].str.startswith("B"))]
        frozen = g[(~g["use_plastic"]) & (g["phase"].str.startswith("B"))]
        if len(plastic) != 1 or len(frozen) != 1:
            continue
        p, f = plastic.iloc[0], frozen.iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "plastic_B_reward_mean": float(p["reward_mean"]),
                "frozen_B_reward_mean": float(f["reward_mean"]),
                "B_reward_delta": float(p["reward_mean"] - f["reward_mean"]),
                "plastic_B_prior_mse_last20": float(p["prior_mse_last20"]),
                "frozen_B_prior_mse_last20": float(f["prior_mse_last20"]),
                "B_prior_mse_delta_last20": float(f["prior_mse_last20"] - p["prior_mse_last20"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", type=Path, required=True)
    args = p.parse_args()

    all_df = load_summaries(args.outroot)
    all_df.to_csv(args.outroot / "all_seed_summaries.csv", index=False)

    paired = paired_b(all_df)
    paired.to_csv(args.outroot / "paired_B_by_seed.csv", index=False)

    agg = pd.DataFrame(
        [
            {
                "n_seeds": len(paired),
                "B_reward_delta_mean": paired["B_reward_delta"].mean(),
                "B_reward_delta_std": paired["B_reward_delta"].std(),
                "reward_win_rate": float((paired["B_reward_delta"] > 0).mean()),
                "B_prior_mse_delta_last20_mean": paired["B_prior_mse_delta_last20"].mean(),
                "B_prior_mse_delta_last20_std": paired["B_prior_mse_delta_last20"].std(),
                "prior_mse_win_rate": float((paired["B_prior_mse_delta_last20"] > 0).mean()),
            }
        ]
    )
    agg.to_csv(args.outroot / "seed_aggregate.csv", index=False)

    print("\n=== Phase B act() paired (plastic - frozen reward; frozen - plastic prior_mse) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {args.outroot / 'seed_aggregate.csv'}")


if __name__ == "__main__":
    main()
