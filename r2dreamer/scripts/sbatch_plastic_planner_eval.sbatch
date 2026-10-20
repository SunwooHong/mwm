#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH -J r2d_cem_plastic
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/r2d_cem_plastic_%j.out
#SBATCH --error=logs/r2d_cem_plastic_%j.err
#
# Submit from r2dreamer root:
#   mkdir -p logs
#   LOGDIR=logdir/dreamer_walker_pretrain_5202322 sbatch scripts/sbatch_plastic_planner_eval.sbatch

set -euo pipefail

R2_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${R2_ROOT}"
mkdir -p logs

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# shellcheck disable=SC1091
source .venv/bin/activate

LOGDIR="${LOGDIR:-logdir/dreamer_walker_pretrain_5202322}"
OUTDIR="${OUTDIR:-${LOGDIR}/plastic_planner_eval}"
SEED="${SEED:-0}"

./scripts/run_plastic_planner_eval.sh "${LOGDIR}" "${OUTDIR}"

echo "Finished. Results: ${OUTDIR}"
