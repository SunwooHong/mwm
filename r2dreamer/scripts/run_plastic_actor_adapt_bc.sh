#!/usr/bin/env bash
# B vs C ablation (10-seed): does online P_t improve replay for frozen-WM actor adapt?
#
#   B: plastic online ON  + frozen WM imagination + actor adapt
#   C: plastic online OFF + frozen WM imagination + actor adapt
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
  echo "Run: uv venv .venv --python 3.11 && uv pip install -e '.[dmc]'"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

LOGDIR="${1:-logdir/dreamer_walker_pretrain_5202322}"
OUTROOT="${2:-${LOGDIR}/plastic_actor_adapt_bc}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
GATE_MODE="${GATE_MODE:-always_on}"
PLASTIC_LR="${PLASTIC_LR:-0.05}"
ADAPT_EVERY="${ADAPT_EVERY:-50}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-4}"
ADAPT_PHASES="${ADAPT_PHASES:-B}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-4}"
BUFFER_BATCH_LENGTH="${BUFFER_BATCH_LENGTH:-32}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUTROOT}"

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

for seed in "${SEED_ARR[@]}"; do
  outdir="${OUTROOT}/seed_${seed}"
  echo "=== B vs C online-plastic ablation seed=${seed} ==="
  python plastic_actor_adapt.py \
    --logdir "${LOGDIR}" \
    --outdir "${outdir}" \
    --device "${DEVICE}" \
    --phases "${PHASES}" \
    --seed "${seed}" \
    --gate-mode "${GATE_MODE}" \
    --plastic-lr "${PLASTIC_LR}" \
    --adapt-phases "${ADAPT_PHASES}" \
    --adapt-every "${ADAPT_EVERY}" \
    --adapt-epochs "${ADAPT_EPOCHS}" \
    --buffer-batch-size "${BUFFER_BATCH_SIZE}" \
    --buffer-batch-length "${BUFFER_BATCH_LENGTH}" \
    --ablation-online-plastic
done

python scripts/aggregate_plastic_actor_adapt_bc.py --outroot "${OUTROOT}"
