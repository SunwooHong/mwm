#!/usr/bin/env python3
"""Aggregate A/B/C/D consolidation runs."""

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
        raise SystemExit(f"No consolidation summaries under {outroot}")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", type=Path, required=True)
    args = p.parse_args()

    all_df = load_summaries(args.outroot)
    all_df.to_csv(args.outroot / "all_summaries.csv", index=False)

    b = all_df[all_df["phase"].str.startswith("B")].copy()
    agg = (
        b.groupby("condition", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            reward_mean=("reward_mean", "mean"),
            reward_std=("reward_mean", "std"),
            trace_norm_last20_mean=("trace_norm_last20", "mean"),
            trace_norm_last20_std=("trace_norm_last20", "std"),
            prior_mse_last20_mean=("prior_mse_last20", "mean"),
        )
        .sort_values("condition")
    )
    agg.to_csv(args.outroot / "consolidation_aggregate.csv", index=False)

    print("\n=== Consolidation Phase B (mean over seeds) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {args.outroot / 'consolidation_aggregate.csv'}")


if __name__ == "__main__":
    main()
