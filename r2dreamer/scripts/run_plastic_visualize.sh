#!/usr/bin/env bash
# Rollout + reward plot + adaptation times + comparison videos for gain-shift eval.
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
OUTDIR="${2:-${LOGDIR}/plastic_visualize}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
SEED="${SEED:-0}"
GATE_MODE="${GATE_MODE:-always_on}"
PLASTIC_LR="${PLASTIC_LR:-0.05}"
VIDEO_MAX_STEPS="${VIDEO_MAX_STEPS:-400}"
VIDEO_PHASES="${VIDEO_PHASES:-B}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

CMD="${PLASTIC_VIZ_CMD:-all}"

python plastic_visualize.py "${CMD}" \
  --logdir "${LOGDIR}" \
  --outdir "${OUTDIR}" \
  --device "${DEVICE}" \
  --phases "${PHASES}" \
  --seed "${SEED}" \
  --gate-mode "${GATE_MODE}" \
  --plastic-lr "${PLASTIC_LR}" \
  --compare-frozen \
  --video-phases "${VIDEO_PHASES}" \
  --video-max-steps "${VIDEO_MAX_STEPS}"

echo "Finished. See ${OUTDIR}/plots/reward_overlay.png and ${OUTDIR}/adaptation_times.csv"
