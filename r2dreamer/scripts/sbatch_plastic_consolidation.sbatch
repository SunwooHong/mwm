#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH -J r2d_consolid
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/r2d_consolid_%j.out
#SBATCH --error=logs/r2d_consolid_%j.err
#
#   mkdir -p logs
#   LOGDIR=logdir/dreamer_walker_pretrain_5202322 sbatch scripts/sbatch_plastic_consolidation.sbatch

set -euo pipefail
R2_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${R2_ROOT}"
mkdir -p logs
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
source .venv/bin/activate
LOGDIR="${LOGDIR:-logdir/dreamer_walker_pretrain_5202322}"
OUTROOT="${OUTROOT:-${LOGDIR}/plastic_consolidation}"
SEEDS="${SEEDS:-0,2,5}"
./scripts/run_plastic_consolidation.sh "${LOGDIR}" "${OUTROOT}"
echo "Finished. Aggregate: ${OUTROOT}/consolidation_aggregate.csv"
