#!/usr/bin/env bash
# CEM planner eval: plastic WM vs frozen WM (no frozen actor).
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
OUTDIR="${2:-${LOGDIR}/plastic_planner_eval}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
SEED="${SEED:-0}"
GATE_MODE="${GATE_MODE:-always_on}"
PLASTIC_LR="${PLASTIC_LR:-0.05}"
CEM_SAMPLES="${CEM_SAMPLES:-256}"
CEM_HORIZON="${CEM_HORIZON:-12}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

python plastic_planner_eval.py \
  --logdir "${LOGDIR}" \
  --outdir "${OUTDIR}" \
  --device "${DEVICE}" \
  --phases "${PHASES}" \
  --gate-mode "${GATE_MODE}" \
  --plastic-lr "${PLASTIC_LR}" \
  --seed "${SEED}" \
  --cem-samples "${CEM_SAMPLES}" \
  --cem-horizon "${CEM_HORIZON}" \
  --compare-frozen
