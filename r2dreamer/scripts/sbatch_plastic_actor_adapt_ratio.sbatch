#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH -J r2d_adapt_ratio
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/r2d_adapt_ratio_%j.out
#SBATCH --error=logs/r2d_adapt_ratio_%j.err
#
# Pilot (3 seeds × 4 configs):
#   mkdir -p logs
#   LOGDIR=logdir/dreamer_walker_pretrain_5202322 sbatch scripts/sbatch_plastic_actor_adapt_ratio.sbatch
#
# Full 10-seed:
#   SEEDS=0,1,2,3,4,5,6,7,8,9 sbatch scripts/sbatch_plastic_actor_adapt_ratio.sbatch

set -euo pipefail

R2_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${R2_ROOT}"
mkdir -p logs

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# shellcheck disable=SC1091
source .venv/bin/activate

LOGDIR="${LOGDIR:-logdir/dreamer_walker_pretrain_5202322}"
OUTROOT="${OUTROOT:-${LOGDIR}/plastic_actor_adapt_ratio}"
SEEDS="${SEEDS:-0,2,5}"
INCLUDE_DEFAULT="${INCLUDE_DEFAULT:-1}"

./scripts/run_plastic_actor_adapt_ratio.sh "${LOGDIR}" "${OUTROOT}"

echo "Finished. Aggregate: ${OUTROOT}/ratio_aggregate.csv"
