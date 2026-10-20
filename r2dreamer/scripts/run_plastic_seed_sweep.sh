#!/usr/bin/env bash
# Multi-seed evaluation for top-3 Phase 3 plastic configs vs frozen baseline.
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
OUTROOT="${2:-${LOGDIR}/plastic_seed_sweep}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUTROOT}"

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

run_case() {
  local config_name="$1"
  local gate="$2"
  local lr="$3"
  local persist="${4:-0.1}"
  local seed="$5"
  local outdir="${OUTROOT}/${config_name}/seed_${seed}"
  echo "=== ${config_name} seed=${seed} ==="
  python plastic_inference.py \
    --logdir "${LOGDIR}" \
    --outdir "${outdir}" \
    --device "${DEVICE}" \
    --phases "${PHASES}" \
    --gate-mode "${gate}" \
    --plastic-lr "${lr}" \
    --persist-threshold "${persist}" \
    --seed "${seed}" \
    --run-label "${config_name}" \
    --compare-frozen
}

for seed in "${SEED_ARR[@]}"; do
  run_case selective_lr0.05 selective 0.05 0.1 "${seed}"
  run_case surprise_only_lr0.1 surprise_only 0.1 0.1 "${seed}"
  run_case always_on_lr0.05 always_on 0.05 0.1 "${seed}"
done

python scripts/aggregate_plastic_seeds.py --outroot "${OUTROOT}"
