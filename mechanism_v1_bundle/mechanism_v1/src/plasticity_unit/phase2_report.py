from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import ExperimentConfig
from .report import NAIVE_RULE, _fmt, make_gate_report, write_gate_report


PHASE2_SCENARIOS = (
    "linear_shift",
    "low_rank_shift",
    "action_gain_shift",
    "back_switch",
    "poison_outlier",
    "nuisance_distractor",
)

ROBUSTNESS_SCENARIOS = {"poison_outlier", "nuisance_distractor"}


def aggregate_phase2_summaries(run_root: Path, scenarios: Iterable[str] = PHASE2_SCENARIOS) -> pd.DataFrame:
    rows: List[dict] = []
    for scenario in scenarios:
        summary_path = run_root / scenario / "summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            rows.append({"scenario": scenario, **row.to_dict()})
    return pd.DataFrame(rows)


def write_phase2_summary(run_root: Path, scenarios: Iterable[str] = PHASE2_SCENARIOS) -> Path:
    agg = aggregate_phase2_summaries(run_root, scenarios)
    agg_path = run_root / "phase2_aggregate_summary.csv"
    agg.to_csv(agg_path, index=False)

    lines: List[str] = [
        "# Phase 2: Non-oracle selective plasticity summary",
        "",
        "Phase 2 compares rollback/gate proxies without held-out validation rollback.",
        "",
        "## Proxy comparison (robustness scenarios)",
        "",
    ]

    for scenario in scenarios:
        if scenario not in ROBUSTNESS_SCENARIOS:
            continue
        sub = agg[agg["scenario"] == scenario]
        if sub.empty:
            continue
        lines.append(f"### `{scenario}`")
        lines.append("")
        focus_cols = [
            "rule",
            "mean_val_loss",
            "auc_shift_core_loss",
            "avg_write_norm_false_context",
            "rollback_rate",
            "anchor_rollback_rate",
            "consistency_rollback_rate",
        ]
        focus_cols = [c for c in focus_cols if c in sub.columns]
        focus = sub[sub["rule"].isin(
            [
                "frozen",
                NAIVE_RULE,
                "surprise_hebb",
                "persistent_error_hebb",
                "error_hebb_anchor_rollback",
                "error_hebb_consistency_rollback",
                "gated_error_hebb_anchor_rollback",
                "full_selective",
                "full_selective_v2",
                "error_hebb_oracle_rollback",
            ]
        )][focus_cols]
        lines.append(focus.to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `error_hebb`: fast adaptation but catastrophic false learning under poison/distractor.",
            "- `persistent_error_hebb`: direction-aware gate reduces one-off writes.",
            "- `error_hebb_anchor_rollback`: anchor-memory proxy rejects harmful updates.",
            "- `error_hebb_consistency_rollback`: local consistency proxy without anchor memory.",
            "- `full_selective_v2`: persistence + clip + anchor + consistency (Phase 2 main).",
            "- `error_hebb_oracle_rollback`: diagnostic ceiling using held-out validation.",
            "",
        ]
    )

    out = run_root / "PHASE2_SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def refresh_gate_reports(run_root: Path, scenarios: Iterable[str] = PHASE2_SCENARIOS) -> None:
    for scenario in scenarios:
        scenario_dir = run_root / scenario
        config_path = scenario_dir / "config.json"
        summary_path = scenario_dir / "summary.csv"
        if not config_path.exists() or not summary_path.exists():
            continue
        cfg = ExperimentConfig(**pd.read_json(config_path, typ="series").to_dict())
        summary = pd.read_csv(summary_path)
        write_gate_report(cfg, summary, scenario_dir)
