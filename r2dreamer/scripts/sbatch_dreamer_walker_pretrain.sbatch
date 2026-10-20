#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH -J r2d_walker_pretrain
#SBATCH -t 6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=logs/r2d_walker_pretrain_%j.out
#SBATCH --error=logs/r2d_walker_pretrain_%j.err
#
# Submit from r2dreamer root:
#   mkdir -p logs && sbatch scripts/sbatch_dreamer_walker_pretrain.sbatch

set -euo pipefail

R2_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${R2_ROOT}"
mkdir -p logs

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# shellcheck disable=SC1091
source .venv/bin/activate

LOGDIR="${LOGDIR:-logdir/dreamer_walker_pretrain_${SLURM_JOB_ID:-local}}"

python train.py \
  logdir="${LOGDIR}" \
  env=dmc_proprio \
  env.task=dmc_walker_walk \
  model.rep_loss=dreamer \
  model.compile=True \
  device=cuda:0 \
  buffer.storage_device=cuda:0 \
  seed=0

echo "Finished. Checkpoint: ${LOGDIR}/latest.pt"
