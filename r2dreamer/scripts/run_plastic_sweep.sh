#!/usr/bin/env bash
# Sweep gate modes and plastic learning rates for Phase 3 diagnostic.
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
OUTROOT="${2:-${LOGDIR}/plastic_sweep_v2}"
DEVICE="${DEVICE:-cuda:0}"
PHASES="${PHASES:-A1:1.0:200,B:0.7:1000,A2:1.0:300}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUTROOT}"

run_case() {
  local gate="$1"
  local lr="$2"
  local persist="${3:-0.1}"
  local label="${gate}_lr${lr}_p${persist}"
  local outdir="${OUTROOT}/${label}"
  echo "=== ${label} ==="
  python plastic_inference.py \
    --logdir "${LOGDIR}" \
    --outdir "${outdir}" \
    --device "${DEVICE}" \
    --phases "${PHASES}" \
    --gate-mode "${gate}" \
    --plastic-lr "${lr}" \
    --persist-threshold "${persist}" \
    --run-label "${label}" \
    --compare-frozen
}

# gate ablation + lr sweep
for gate in always_on surprise_only selective; do
  for lr in 0.01 0.05 0.1; do
    run_case "${gate}" "${lr}" 0.1
  done
done

# lower persistence threshold for selective (more writing)
run_case selective 0.05 0.02
run_case selective 0.1 0.02

OUTROOT="${OUTROOT}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

outroot = Path(os.environ["OUTROOT"])
frames = []
for d in sorted(outroot.iterdir()):
    summary = d / "summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        df["sweep_dir"] = d.name
        frames.append(df)
if not frames:
    raise SystemExit("No sweep results found")
all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv(outroot / "sweep_summary.csv", index=False)

# best on Phase B last20 prior_mse (plastic only)
b = all_df[(all_df["use_plastic"]) & (all_df["phase"].str.startswith("B"))]
if len(b):
    best = b.sort_values("prior_mse_last20").head(10)
    best.to_csv(outroot / "sweep_best_B_last20.csv", index=False)
    print("\nTop B-phase plastic (prior_mse_last20):")
    cols = ["sweep_dir", "gate_mode", "plastic_lr", "prior_mse_last20", "dyn_kl_last20", "gate_mean", "trace_norm_max"]
    print(best[cols].to_string(index=False))
print(f"\nWrote {outroot / 'sweep_summary.csv'}")
PY
