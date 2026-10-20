#!/usr/bin/env python3
"""Aggregate B vs C online-plastic ablation (frozen WM actor adapt)."""

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
        raise SystemExit(f"No B vs C summaries under {outroot}")
    return pd.concat(rows, ignore_index=True)


def paired_bc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, g in df.groupby("seed"):
        b_cond = g[
            (g["actor_adapt"])
            & (g["plastic_online"])
            & (~g["plastic_wm_imagine"])
            & (g["phase"].str.startswith("B"))
        ]
        c_cond = g[
            (g["actor_adapt"])
            & (~g["plastic_online"])
            & (~g["plastic_wm_imagine"])
            & (g["phase"].str.startswith("B"))
        ]
        if len(b_cond) != 1 or len(c_cond) != 1:
            continue
        b, c = b_cond.iloc[0], c_cond.iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "B_reward": float(b["reward_mean"]),
                "C_reward": float(c["reward_mean"]),
                "B_minus_C_reward": float(b["reward_mean"] - c["reward_mean"]),
                "B_prior_mse_last20": float(b["prior_mse_last20"]),
                "C_prior_mse_last20": float(c["prior_mse_last20"]),
                "C_minus_B_prior_mse_last20": float(c["prior_mse_last20"] - b["prior_mse_last20"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", type=Path, required=True)
    args = p.parse_args()

    all_df = load_summaries(args.outroot)
    all_df.to_csv(args.outroot / "all_seed_summaries.csv", index=False)

    paired = paired_bc(all_df)
    paired.to_csv(args.outroot / "paired_B_by_seed.csv", index=False)

    agg = pd.DataFrame(
        [
            {
                "n_seeds": len(paired),
                "B_reward_mean": paired["B_reward"].mean(),
                "C_reward_mean": paired["C_reward"].mean(),
                "B_minus_C_reward_mean": paired["B_minus_C_reward"].mean(),
                "B_minus_C_reward_std": paired["B_minus_C_reward"].std(),
                "B_win_rate": float((paired["B_minus_C_reward"] > 0).mean()),
                "C_minus_B_prior_mse_last20_mean": paired["C_minus_B_prior_mse_last20"].mean(),
            }
        ]
    )
    agg.to_csv(args.outroot / "bc_aggregate.csv", index=False)

    print("\n=== B vs C (Phase B): plastic online ON vs OFF, frozen WM imagine ===")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {args.outroot / 'bc_aggregate.csv'}")


if __name__ == "__main__":
    main()
