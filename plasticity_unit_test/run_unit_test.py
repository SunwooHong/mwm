#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from plasticity_unit.config import ExperimentConfig
from plasticity_unit.experiment import run_experiment
from plasticity_unit.plotting import plot_summary, plot_timeseries
from plasticity_unit.rules import RULES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic unit test for Hebbian-like plastic residual updates.")
    p.add_argument("--outdir", type=Path, default=Path("runs/plasticity_unit"))
    p.add_argument("--scenario", type=str, default="linear_shift", choices=["linear_shift", "poison", "distractor"])
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dz", type=int, default=16)
    p.add_argument("--da", type=int, default=4)
    p.add_argument("--shift-scale", type=float, default=0.16)
    p.add_argument("--lr-error", type=float, default=0.030)
    p.add_argument("--lr-hebb", type=float, default=0.0015)
    p.add_argument("--max-trace-norm", type=float, default=3.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--rules",
        nargs="*",
        default=RULES,
        help="Subset of rules to run. Default: all.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=args.device,
        dz=args.dz,
        da=args.da,
        scenario=args.scenario,
        shift_scale=args.shift_scale,
        lr_error=args.lr_error,
        lr_hebb=args.lr_hebb,
        max_trace_norm=args.max_trace_norm,
    )
    metrics, summary = run_experiment(cfg, args.outdir, rules=args.rules)
    if not args.no_plots:
        plot_timeseries(metrics, args.outdir / "plots", smooth_window=cfg.smooth_window)
        plot_summary(summary, args.outdir / "plots")

    print("\n=== Summary ===")
    cols = [
        "rule",
        "mean_candidate_delta_val",
        "beneficial_update_rate",
        "mean_applied_delta_val",
        "applied_beneficial_rate",
        "mean_grad_alignment",
        "auc_shift_val_loss",
        "recovery_ratio_vs_oracle",
        "max_trace_norm",
        "rollback_rate",
        "false_write_rate_gate_gt_0_5",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nWrote metrics to: {args.outdir}")


if __name__ == "__main__":
    main()
