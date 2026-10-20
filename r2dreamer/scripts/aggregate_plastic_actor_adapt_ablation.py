#!/usr/bin/env python3
"""Aggregate 3+ seed WM-imagination ablation for plastic actor adaptation."""

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
        raise SystemExit(f"No ablation summaries under {outroot}")
    return pd.concat(rows, ignore_index=True)


def paired_wm_imagine(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, g in df.groupby("seed"):
        plastic = g[
            (g["actor_adapt"]) & (g["plastic_wm_imagine"]) & (g["phase"].str.startswith("B"))
        ]
        frozen = g[
            (g["actor_adapt"]) & (~g["plastic_wm_imagine"]) & (g["phase"].str.startswith("B"))
        ]
        if len(plastic) != 1 or len(frozen) != 1:
            continue
        p, f = plastic.iloc[0], frozen.iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "plastic_wm_B_reward": float(p["reward_mean"]),
                "frozen_wm_B_reward": float(f["reward_mean"]),
                "B_reward_delta_plastic_minus_frozen": float(p["reward_mean"] - f["reward_mean"]),
                "plastic_wm_B_prior_mse_last20": float(p["prior_mse_last20"]),
                "frozen_wm_B_prior_mse_last20": float(f["prior_mse_last20"]),
                "B_prior_mse_delta_frozen_minus_plastic": float(
                    f["prior_mse_last20"] - p["prior_mse_last20"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", type=Path, required=True)
    args = p.parse_args()

    all_df = load_summaries(args.outroot)
    all_df.to_csv(args.outroot / "all_seed_summaries.csv", index=False)

    paired = paired_wm_imagine(all_df)
    paired.to_csv(args.outroot / "paired_B_by_seed.csv", index=False)

    agg = pd.DataFrame(
        [
            {
                "n_seeds": len(paired),
                "plastic_wm_B_reward_mean": paired["plastic_wm_B_reward"].mean(),
                "frozen_wm_B_reward_mean": paired["frozen_wm_B_reward"].mean(),
                "B_reward_delta_mean": paired["B_reward_delta_plastic_minus_frozen"].mean(),
                "B_reward_delta_std": paired["B_reward_delta_plastic_minus_frozen"].std(),
                "plastic_wm_win_rate": float(
                    (paired["B_reward_delta_plastic_minus_frozen"] > 0).mean()
                ),
                "B_prior_mse_delta_mean": paired["B_prior_mse_delta_frozen_minus_plastic"].mean(),
            }
        ]
    )
    agg.to_csv(args.outroot / "ablation_aggregate.csv", index=False)

    print("\n=== WM-imagine ablation: plastic_wm - frozen_wm (Phase B reward) ===")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {args.outroot / 'ablation_aggregate.csv'}")


if __name__ == "__main__":
    main()
