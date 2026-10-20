#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH -J mwm_s1_diag
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --output=logs/mwm_s1_diag_%j.out
#SBATCH --error=logs/mwm_s1_diag_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
source .venv/bin/activate
export PYTHONPATH=.
python -u scripts/v2_system1_diagnosis.py
