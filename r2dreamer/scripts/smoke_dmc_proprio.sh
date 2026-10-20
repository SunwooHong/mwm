#!/usr/bin/env bash
# Minimal R2-Dreamer smoke test for Phase 3 baseline verification.
# Uses Dreamer (not R2-Dreamer) rep_loss so plastic residual is the only novelty later.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: uv venv .venv --python 3.11 && uv pip install -e '.[dmc]'"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
LOGDIR="${1:-${ROOT}/logdir/smoke_dmc_proprio_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-800}"
ENV_NUM="${ENV_NUM:-2}"

echo "Logdir: ${LOGDIR}"
echo "Device: ${DEVICE} | Steps: ${STEPS} | env_num: ${ENV_NUM} | MUJOCO_GL=${MUJOCO_GL}"

python train.py \
  logdir="${LOGDIR}" \
  env=dmc_proprio \
  env.task=dmc_walker_walk \
  env.env_num="${ENV_NUM}" \
  env.eval_episode_num=0 \
  model.rep_loss=dreamer \
  model.compile=False \
  device="${DEVICE}" \
  buffer.storage_device="${DEVICE}" \
  trainer.steps="${STEPS}" \
  trainer.eval_every=400 \
  trainer.train_ratio=32 \
  trainer.update_log_every=200 \
  batch_size=4 \
  batch_length=16 \
  buffer.max_size=5000 \
  seed=0

echo ""
echo "Smoke run finished. Check ${LOGDIR}/console.log and tensorboard logs."
