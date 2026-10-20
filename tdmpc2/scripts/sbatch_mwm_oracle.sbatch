#!/bin/bash
#SBATCH -A aip-rahulgk
#SBATCH -J mwm_oracle
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --output=logs/mwm_oracle_%j.out
#SBATCH --error=logs/mwm_oracle_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/tdmpc2"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
source /home/hongsunw/projects/aip-rahulgk/hongsunw/mwm/r2dreamer/.venv/bin/activate
export PYTHONPATH=.
python -u mwm_oracle.py
