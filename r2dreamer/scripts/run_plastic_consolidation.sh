#!/usr/bin/env bash
# Consolidation eval: A/B/C/D conditions (3-seed pilot by default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source .venv/bin/activate

LOGDIR="${1:-logdir/dreamer_walker_pretrain_5202322}"
OUTROOT="${2:-${LOGDIR}/plastic_consolidation}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
SEEDS="${SEEDS:-0,2,5}"
CONDITIONS="${CONDITIONS:-A,B,C,D}"
ADAPT_EVERY="${ADAPT_EVERY:-50}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-4}"
SLOW_EVERY="${SLOW_EVERY:-100}"
SLOW_EPOCHS="${SLOW_EPOCHS:-1}"
WM_LR_SCALE="${WM_LR_SCALE:-0.02}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "${OUTROOT}"

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

for seed in "${SEED_ARR[@]}"; do
  outdir="${OUTROOT}/seed_${seed}"
  echo "=== consolidation seed=${seed} conditions=${CONDITIONS} ==="
  python plastic_consolidation_eval.py \
    --logdir "${LOGDIR}" \
    --outdir "${outdir}" \
    --device "${DEVICE}" \
    --phases "${PHASES}" \
    --seed "${seed}" \
    --conditions "${CONDITIONS}" \
    --adapt-every "${ADAPT_EVERY}" \
    --adapt-epochs "${ADAPT_EPOCHS}" \
    --slow-every "${SLOW_EVERY}" \
    --slow-epochs "${SLOW_EPOCHS}" \
    --wm-lr-scale "${WM_LR_SCALE}"
done

python scripts/aggregate_plastic_consolidation.py --outroot "${OUTROOT}"
