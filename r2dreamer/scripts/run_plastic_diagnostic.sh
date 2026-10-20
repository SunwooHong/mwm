#!/usr/bin/env bash
# Phase 3 inference diagnostic (frozen checkpoint + test-time plastic only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
  echo "Run: uv venv .venv --python 3.11 && uv pip install -e '.[dmc]'"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

LOGDIR="${1:-logdir/smoke_gpu_baseline}"
OUTDIR="${2:-${LOGDIR}/plastic_diagnostic_v2}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
GATE_MODE="${GATE_MODE:-always_on}"
PLASTIC_LR="${PLASTIC_LR:-0.05}"
PERSIST_THRESHOLD="${PERSIST_THRESHOLD:-0.1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

python plastic_inference.py \
  --logdir "${LOGDIR}" \
  --outdir "${OUTDIR}" \
  --device "${DEVICE}" \
  --phases "${PHASES}" \
  --gate-mode "${GATE_MODE}" \
  --plastic-lr "${PLASTIC_LR}" \
  --persist-threshold "${PERSIST_THRESHOLD}" \
  --compare-frozen
