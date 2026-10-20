#!/usr/bin/env bash
set -euo pipefail

OUTROOT=${1:-runs/phase2}
SEEDS=${SEEDS:-10}
PY=${PY:-python}

mkdir -p "${OUTROOT}"

for scenario in linear_shift low_rank_shift action_gain_shift back_switch poison_outlier nuisance_distractor; do
  echo ""
  echo "=== Phase 2: ${scenario} (${SEEDS} seeds) ==="
  PYTHONPATH=src ${PY} run_unit_test.py \
    --outdir "${OUTROOT}/${scenario}" \
    --scenario "${scenario}" \
    --num-seeds "${SEEDS}" \
    --phase2-rules
done

PYTHONPATH=src ${PY} - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "src")
from plasticity_unit.phase2_report import write_phase2_summary, refresh_gate_reports

root = Path("${OUTROOT}")
refresh_gate_reports(root)
summary = write_phase2_summary(root)
print(f"\nWrote aggregate summary: {summary}")
PY

echo ""
echo "Phase 2 sweep complete under ${OUTROOT}"
