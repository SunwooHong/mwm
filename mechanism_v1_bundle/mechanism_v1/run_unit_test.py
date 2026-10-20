#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from plasticity_unit.config import SCENARIOS, ExperimentConfig
from plasticity_unit.experiment import run_experiment
from plasticity_unit.plotting import plot_summary, plot_timeseries
from plasticity_unit.report import write_gate_report
from plasticity_unit.rules import CORE_RULES, PHASE2_RULES, RULES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic mechanism tests for prediction-error-modulated local plasticity.")
    p.add_argument("--outdir", type=Path, default=Path("runs/mechanism_v1/linear_shift"))
    p.add_argument("--scenario", type=str, default="linear_shift", choices=sorted(SCENARIOS))
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dz", type=int, default=16)
    p.add_argument("--da", type=int, default=4)
    p.add_argument("--low-rank", type=int, default=2)
    p.add_argument("--shift-scale", type=float, default=0.16)
    p.add_argument("--action-gain-delta", type=float, default=-0.35)
    p.add_argument("--lr-error", type=float, default=0.030)
    p.add_argument("--lr-hebb", type=float, default=0.0015)
    p.add_argument("--max-trace-norm", type=float, default=3.0)
    p.add_argument("--error-clip-norm", type=float, default=0.25)
    p.add_argument("--anchor-loss-tolerance", type=float, default=0.001)
    p.add_argument("--anchor-add-loss-threshold", type=float, default=0.015)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--core-rules", action="store_true", help="Run the smaller default set for fast paper-figure smoke tests.")
    p.add_argument("--phase2-rules", action="store_true", help="Run Phase 2 rollback/gate comparison rules.")
    p.add_argument(
        "--rules",
        nargs="*",
        default=None,
        help="Subset of rules to run. Default: all, or CORE_RULES when --core-rules is set.",
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
        low_rank=args.low_rank,
        scenario=args.scenario,
        shift_scale=args.shift_scale,
        action_gain_delta=args.action_gain_delta,
        lr_error=args.lr_error,
        lr_hebb=args.lr_hebb,
        max_trace_norm=args.max_trace_norm,
        error_clip_norm=args.error_clip_norm,
        anchor_loss_tolerance=args.anchor_loss_tolerance,
        anchor_add_loss_threshold=args.anchor_add_loss_threshold,
    )
    rules = args.rules
    if rules is None:
        if args.phase2_rules:
            rules = PHASE2_RULES
        elif args.core_rules:
            rules = CORE_RULES
        else:
            rules = RULES
    metrics, summary = run_experiment(cfg, args.outdir, rules=rules)
    if not args.no_plots:
        plot_timeseries(metrics, args.outdir / "plots", smooth_window=cfg.smooth_window)
        plot_summary(summary, args.outdir / "plots")
    report_path = write_gate_report(cfg, summary, args.outdir)

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
        "avg_write_norm_false_context",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f"\nWrote metrics to: {args.outdir}")
    print(f"Wrote gate report to: {report_path}")


if __name__ == "__main__":
    main()
