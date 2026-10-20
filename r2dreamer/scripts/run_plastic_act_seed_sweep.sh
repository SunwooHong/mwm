#!/usr/bin/env bash
# Multi-seed Dreamer.act() eval: always_on lr=0.05 vs frozen under gain-shift phases.
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
OUTROOT="${2:-${LOGDIR}/plastic_act_seed_sweep}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
GATE_MODE="${GATE_MODE:-always_on}"
PLASTIC_LR="${PLASTIC_LR:-0.05}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUTROOT}"

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

for seed in "${SEED_ARR[@]}"; do
  outdir="${OUTROOT}/seed_${seed}"
  echo "=== act() eval seed=${seed} ==="
  python plastic_eval.py \
    --logdir "${LOGDIR}" \
    --outdir "${outdir}" \
    --device "${DEVICE}" \
    --phases "${PHASES}" \
    --gate-mode "${GATE_MODE}" \
    --plastic-lr "${PLASTIC_LR}" \
    --seed "${seed}" \
    --compare-frozen
done

python scripts/aggregate_plastic_act_seeds.py --outroot "${OUTROOT}"
