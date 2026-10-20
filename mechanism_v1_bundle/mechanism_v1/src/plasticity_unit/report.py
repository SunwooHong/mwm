from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import math
import pandas as pd

from .config import ExperimentConfig


MAIN_RULE = "full_selective_v2"
MAIN_RULE_PHASE1 = "full_selective"
NAIVE_RULE = "error_hebb"


def _row(summary: pd.DataFrame, rule: str) -> Optional[pd.Series]:
    sub = summary[summary["rule"] == rule]
    if sub.empty:
        return None
    return sub.iloc[0]


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if abs(x) >= 0.01:
        return f"{x:.4f}"
    return f"{x:.3e}"


def _add(checks: List[Dict[str, str | bool]], name: str, passed: bool, value: str | float) -> None:
    checks.append({"name": name, "pass": bool(passed), "value": value if isinstance(value, str) else _fmt(float(value))})


def make_gate_report(cfg: ExperimentConfig, summary: pd.DataFrame, main_rule: str = MAIN_RULE) -> str:
    main = _row(summary, main_rule)
    if main is None:
        main = _row(summary, MAIN_RULE_PHASE1)
    if main is None:
        main = _row(summary, "persistent_error_hebb")
    if main is None:
        main = _row(summary, NAIVE_RULE)
    frozen = _row(summary, "frozen")
    naive = _row(summary, NAIVE_RULE)
    oracle = _row(summary, "sgd_oracle")

    lines: List[str] = []
    scenario = cfg.canonical_scenario
    lines.append(f"# Mechanism v1 pass/fail report: `{scenario}`")
    lines.append("")
    lines.append("Main rule evaluated: `" + str(main["rule"] if main is not None else "missing") + "`")
    lines.append("")

    if main is None:
        lines.append("No main rule found in summary.")
        return "\n".join(lines)

    checks: List[Dict[str, str | bool]] = []
    _add(checks, "trace norm bounded", main["max_trace_norm"] <= cfg.max_trace_norm * 1.01 + 1e-9, float(main["max_trace_norm"]))

    adaptation_scenarios = {"linear_shift", "low_rank_shift", "action_gain_shift", "back_switch"}
    if scenario in adaptation_scenarios:
        _add(checks, "mean candidate Δ held-out loss < 0", main["mean_candidate_delta_val"] < 0, float(main["mean_candidate_delta_val"]))
        _add(checks, "beneficial update rate > 0.60", main["beneficial_update_rate"] > 0.60, float(main["beneficial_update_rate"]))
        _add(checks, "mean gradient alignment > 0.30", main["mean_grad_alignment"] > 0.30, float(main["mean_grad_alignment"]))
        if frozen is not None:
            _add(
                checks,
                "shift AUC lower than frozen",
                main["auc_shift_val_loss"] < frozen["auc_shift_val_loss"],
                f"main={_fmt(float(main['auc_shift_val_loss']))}, frozen={_fmt(float(frozen['auc_shift_val_loss']))}",
            )
        if oracle is not None and not math.isnan(float(main["recovery_ratio_vs_oracle"])):
            _add(checks, "recovery ratio vs SGD oracle > 0.40", main["recovery_ratio_vs_oracle"] > 0.40, float(main["recovery_ratio_vs_oracle"]))

    elif scenario == "poison_outlier":
        # No true shift. The desired behavior is not to learn corrupted one-off observations.
        _add(checks, "mean applied Δ held-out loss <= small tolerance", main["mean_applied_delta_val"] <= 1e-5, float(main["mean_applied_delta_val"]))
        if frozen is not None:
            tol = max(1e-4, 0.25 * float(frozen["mean_val_loss"]))
            _add(
                checks,
                "clean validation loss preserved near frozen",
                main["mean_val_loss"] <= frozen["mean_val_loss"] + tol,
                f"main={_fmt(float(main['mean_val_loss']))}, frozen={_fmt(float(frozen['mean_val_loss']))}",
            )
        if naive is not None:
            naive_fw = float(naive["avg_write_norm_false_context"])
            main_fw = float(main["avg_write_norm_false_context"])
            if not math.isnan(naive_fw) and not math.isnan(main_fw):
                _add(
                    checks,
                    "false-context applied write norm reduced by >50% vs naive error-Hebb",
                    main_fw <= 0.5 * naive_fw,
                    f"main={_fmt(main_fw)}, naive={_fmt(naive_fw)}",
                )
            _add(
                checks,
                "clean validation loss lower than naive error-Hebb",
                main["mean_val_loss"] < naive["mean_val_loss"],
                f"main={_fmt(float(main['mean_val_loss']))}, naive={_fmt(float(naive['mean_val_loss']))}",
            )

    elif scenario == "nuisance_distractor":
        # Full loss includes nuisance dimensions, so core/control preservation is the target.
        if frozen is not None:
            tol = max(1e-4, 0.25 * float(frozen["auc_shift_core_loss"]))
            _add(
                checks,
                "distractor core loss not much worse than frozen",
                main["auc_shift_core_loss"] <= frozen["auc_shift_core_loss"] + tol,
                f"main={_fmt(float(main['auc_shift_core_loss']))}, frozen={_fmt(float(frozen['auc_shift_core_loss']))}",
            )
        if naive is not None:
            naive_fw = float(naive["avg_write_norm_false_context"])
            main_fw = float(main["avg_write_norm_false_context"])
            if not math.isnan(naive_fw) and not math.isnan(main_fw):
                _add(
                    checks,
                    "false-context applied write norm reduced by >50% vs naive error-Hebb",
                    main_fw <= 0.5 * naive_fw,
                    f"main={_fmt(main_fw)}, naive={_fmt(naive_fw)}",
                )
            _add(
                checks,
                "core loss lower than naive error-Hebb",
                main["auc_shift_core_loss"] < naive["auc_shift_core_loss"],
                f"main={_fmt(float(main['auc_shift_core_loss']))}, naive={_fmt(float(naive['auc_shift_core_loss']))}",
            )
        _add(checks, "mean applied core Δ <= small tolerance", main["mean_applied_delta_core"] <= 1e-5, float(main["mean_applied_delta_core"]))

    passed = sum(bool(c["pass"]) for c in checks)
    lines.append(f"Overall: **{passed}/{len(checks)} checks passed**")
    lines.append("")
    lines.append("| Check | Result | Value |")
    lines.append("|---|---:|---:|")
    for c in checks:
        lines.append(f"| {c['name']} | {'PASS' if c['pass'] else 'FAIL'} | {c['value']} |")

    lines.append("")
    lines.append("## Summary table")
    cols = [
        "rule",
        "mean_candidate_delta_val",
        "mean_applied_delta_val",
        "beneficial_update_rate",
        "mean_grad_alignment",
        "mean_val_loss",
        "auc_shift_val_loss",
        "auc_shift_core_loss",
        "recovery_ratio_vs_oracle",
        "max_trace_norm",
        "rollback_rate",
        "anchor_rollback_rate",
        "consistency_rollback_rate",
        "avg_write_norm_false_context",
    ]
    cols = [c for c in cols if c in summary.columns]
    lines.append(summary[cols].to_markdown(index=False))
    lines.append("")
    lines.append("Interpretation note: this is a research gate, not a final RL result. Synthetic mechanism pass means the local rule is worth integrating into a Dreamer-style world model.")
    return "\n".join(lines)


def write_gate_report(cfg: ExperimentConfig, summary: pd.DataFrame, outdir: Path, main_rule: str = MAIN_RULE) -> Path:
    text = make_gate_report(cfg, summary, main_rule=main_rule)
    path = outdir / "summary_gate.md"
    path.write_text(text, encoding="utf-8")
    return path
