# Mechanism v1: Selective Local Plasticity Unit Tests

This package implements **Phase 0–2** for the Malleable World Models research direction.

The goal is **not** to prove full DreamerV3 performance yet. The goal is to verify the mechanism before full-agent experiments:

> Can a frozen transition model's systematic error be reduced by a bounded, backprop-free, signed local plastic update? Can the same system avoid false writes under outliers and nuisance-only shifts?

## Phase 0 claim

See [`docs/PHASE0_CLAIM.md`](docs/PHASE0_CLAIM.md).

The locked mechanism is:

```math
\Delta P_t = \eta\, g_t\, e_t x_t^\top - \lambda P_t
```

where `e_t` is a signed prediction error. Pure Hebbian/Oja rules are retained as ablations, not as the main claim.

## Install

```bash
pip install -r requirements.txt
```

## Single scenario

```bash
PYTHONPATH=src python run_unit_test.py \
  --outdir runs/mechanism_v1/linear_shift \
  --scenario linear_shift \
  --num-seeds 5 \
  --core-rules
```

## Full Phase-1 sweep

```bash
SEEDS=20 ./scripts/run_all_mechanism.sh runs/mechanism_v1
```

## Phase 2: non-oracle rollback proxies

See [`docs/PHASE2_EXPERIMENT_PLAN.md`](docs/PHASE2_EXPERIMENT_PLAN.md).

```bash
SEEDS=10 ./scripts/run_phase2.sh runs/phase2
```

Phase 2 main rule: `full_selective_v2` = signed error + direction persistence gate + clip + anchor rollback + consistency rollback.

Outputs include `PHASE2_SUMMARY.md` and `phase2_aggregate_summary.csv`.

Scenarios:

```text
linear_shift
low_rank_shift
action_gain_shift
back_switch
poison_outlier
nuisance_distractor
```

## Outputs

Each scenario writes:

```text
metrics.csv
summary.csv
summary_gate.md
config.json
plots/*.png
```

Key metrics:

- `candidate_delta_val`: held-out validation loss after candidate update minus before update. Negative is good.
- `beneficial_update_rate`: fraction of candidate updates with negative held-out Δ loss.
- `grad_alignment`: cosine similarity between local update and true negative per-sample gradient.
- `p_error_to_true_delta`: Frobenius distance between `P_t` and the known regime residual `ΔM`.
- `trace_norm`: Frobenius norm of `P_t`.
- `avg_write_norm_false_context`: write magnitude during poison/distractor contexts.
- `summary_gate.md`: automatic research pass/fail report.

## Current main rule

**Phase 2:** `full_selective_v2` (non-oracle selective plasticity)

**Phase 1:** `full_selective`

```text
signed prediction-error update
+ direction-aware persistence gate (surprise + κ_t)
+ robust error clipping
+ anchor-memory rollback
+ consistency rollback (Phase 2)
+ optional task-relevance mask for nuisance tests
+ trace norm projection
```

`error_hebb_oracle_rollback` is diagnostic only and should not be treated as a deployable method.

## R2-Dreamer integration note

The `r2dreamer_patch/` directory contains a prior-logit residual skeleton for the next phase. Do not treat it as a final patch yet. The synthetic mechanism gates should pass first.
