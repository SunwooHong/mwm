#!/usr/bin/env bash
# Condition B ratio sweep: plastic online + frozen WM imagine + actor adapt.
#
# CONFIGS:
#   top2   — cfg0_default + cfg1_conservative (recommended after pilot)
#   top3   — top2 + cfg1_5_hybrid (50/2/0.3×)
#   pilot  — top2 + cfg2_diversity + cfg3_micro (legacy 4-config pilot)
#   all    — same as pilot
#
# COMPARE_NO_ADAPT=1 — also run B_no_adapt (plastic online, frozen actor) per seed.
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
OUTROOT="${2:-${LOGDIR}/plastic_actor_adapt_ratio}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
SEEDS="${SEEDS:-0,2,5}"
GATE_MODE="${GATE_MODE:-always_on}"
PLASTIC_LR="${PLASTIC_LR:-0.05}"
ADAPT_PHASES="${ADAPT_PHASES:-B}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-4}"
BUFFER_BATCH_LENGTH="${BUFFER_BATCH_LENGTH:-32}"
CONFIGS="${CONFIGS:-top2}"
COMPARE_NO_ADAPT="${COMPARE_NO_ADAPT:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUTROOT}"

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

run_cfg() {
  local cfg_name="$1"
  local adapt_every="$2"
  local adapt_epochs="$3"
  local actor_lr_scale="$4"
  local seed="$5"
  local outdir="${OUTROOT}/${cfg_name}/seed_${seed}"
  echo "=== ${cfg_name} seed=${seed} every=${adapt_every} epochs=${adapt_epochs} lr_scale=${actor_lr_scale} ==="
  python plastic_actor_adapt.py \
    --logdir "${LOGDIR}" \
    --outdir "${outdir}" \
    --device "${DEVICE}" \
    --phases "${PHASES}" \
    --seed "${seed}" \
    --gate-mode "${GATE_MODE}" \
    --plastic-lr "${PLASTIC_LR}" \
    --adapt-phases "${ADAPT_PHASES}" \
    --adapt-every "${adapt_every}" \
    --adapt-epochs "${adapt_epochs}" \
    --actor-lr-scale "${actor_lr_scale}" \
    --buffer-batch-size "${BUFFER_BATCH_SIZE}" \
    --buffer-batch-length "${BUFFER_BATCH_LENGTH}" \
    --plastic-enabled \
    --no-plastic-wm-imagine \
    --actor-adapt \
    --run-label "${cfg_name}"
}

run_no_adapt() {
  local seed="$1"
  local outdir="${OUTROOT}/B_no_adapt/seed_${seed}"
  echo "=== B_no_adapt seed=${seed} (plastic online, frozen actor) ==="
  python plastic_actor_adapt.py \
    --logdir "${LOGDIR}" \
    --outdir "${outdir}" \
    --device "${DEVICE}" \
    --phases "${PHASES}" \
    --seed "${seed}" \
    --gate-mode "${GATE_MODE}" \
    --plastic-lr "${PLASTIC_LR}" \
    --plastic-enabled \
    --no-plastic-wm-imagine \
    --run-label "B_no_adapt"
}

should_run_cfg() {
  local name="$1"
  case "${CONFIGS}" in
    top2)
      [[ "${name}" == cfg0_default || "${name}" == cfg1_conservative ]]
      ;;
    top3)
      [[ "${name}" == cfg0_default || "${name}" == cfg1_conservative || "${name}" == cfg1_5_hybrid ]]
      ;;
    pilot|all)
      [[ "${name}" == cfg0_default || "${name}" == cfg1_conservative || "${name}" == cfg2_diversity || "${name}" == cfg3_micro ]]
      ;;
    *)
      echo "Unknown CONFIGS=${CONFIGS} (use top2, top3, pilot, all)" >&2
      exit 1
      ;;
  esac
}

for seed in "${SEED_ARR[@]}"; do
  if [[ "${COMPARE_NO_ADAPT}" == "1" ]]; then
    run_no_adapt "${seed}"
  fi
  if should_run_cfg cfg0_default; then
    run_cfg cfg0_default 50 4 1.0 "${seed}"
  fi
  if should_run_cfg cfg1_conservative; then
    run_cfg cfg1_conservative 100 2 0.2 "${seed}"
  fi
  if should_run_cfg cfg1_5_hybrid; then
    run_cfg cfg1_5_hybrid 50 2 0.3 "${seed}"
  fi
  if should_run_cfg cfg2_diversity; then
    run_cfg cfg2_diversity 200 4 0.5 "${seed}"
  fi
  if should_run_cfg cfg3_micro; then
    run_cfg cfg3_micro 25 1 0.1 "${seed}"
  fi
done

python scripts/aggregate_plastic_actor_adapt_ratio.py --outroot "${OUTROOT}" --logdir "${LOGDIR}"
