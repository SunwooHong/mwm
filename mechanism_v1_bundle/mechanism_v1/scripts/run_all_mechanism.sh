#!/usr/bin/env bash
set -euo pipefail

OUTROOT=${1:-runs/mechanism_v1}
SEEDS=${SEEDS:-20}
PY=${PY:-python}
RULE_FLAG=${RULE_FLAG:---core-rules}

for scenario in linear_shift low_rank_shift action_gain_shift back_switch poison_outlier nuisance_distractor; do
  echo "\n=== Running ${scenario} with ${SEEDS} seeds ==="
  PYTHONPATH=src ${PY} run_unit_test.py \
    --outdir "${OUTROOT}/${scenario}" \
    --scenario "${scenario}" \
    --num-seeds "${SEEDS}" \
    ${RULE_FLAG}
done
