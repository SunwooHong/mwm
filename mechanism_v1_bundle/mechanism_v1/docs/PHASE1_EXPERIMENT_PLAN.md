# Phase 1: Synthetic Mechanism Package

## Scenarios implemented

| Scenario | Purpose |
|---|---|
| `linear_shift` | Dense systematic transition residual in regime B. |
| `low_rank_shift` | Low-rank transition residual; verifies that the plastic trace can identify a compact systematic shift. |
| `action_gain_shift` | Action-column-only residual; mimics actuator gain changes. |
| `back_switch` | A→B→A→B sequence; measures recovery and leakage. |
| `poison_outlier` | True low-rank shift plus occasional corrupted next-state observations. |
| `nuisance_distractor` | Core/control dynamics unchanged; nuisance dimensions and observations shift. |

## Rules implemented

| Rule | Purpose |
|---|---|
| `frozen` | Lower bound: no plasticity. |
| `random` | Trace perturbation control. |
| `pure_hebb` | Co-activation without signed prediction error. |
| `oja` | Pure Hebbian plus Oja-style stabilization. |
| `surprise_hebb` | Scalar surprise gates co-activation; direction remains non-error-corrective. |
| `error_hebb` | Signed error-Hebbian local delta update. |
| `persistent_error_hebb` | Signed error update with persistence gate. |
| `error_hebb_oja` | Signed error update plus Oja-style stabilization. |
| `error_hebb_anchor_rollback` | Naive signed error update with non-oracle anchor rollback. |
| `gated_error_hebb_anchor_rollback` | Persistence-gated signed update with anchor rollback. |
| `full_selective` | Main Phase-1 selective rule: signed update + persistence gate + robust clipping + anchor rollback hooks + optional nuisance task mask. |
| `error_hebb_oracle_rollback` | Diagnostic upper-bound rollback using held-out validation. |
| `sgd_oracle` | Mini-batch supervised residual update upper bound. |

## Key outputs

Each run writes:

- `metrics.csv`: per-step metrics.
- `summary.csv`: rule-level summary.
- `summary_gate.md`: pass/fail report.
- `plots/*.png`: mechanism plots.

## Main pass/fail checks

- `mean_candidate_delta_val < 0`
- `beneficial_update_rate > 0.60`
- `mean_grad_alignment > 0.30`
- `auc_shift_val_loss < frozen`
- `recovery_ratio_vs_oracle > 0.40`
- bounded `max_trace_norm`
- poison/distractor false-write norm reduced vs naive `error_hebb`

## Recommended run

```bash
PYTHONPATH=src python run_unit_test.py \
  --outdir runs/mechanism_v1/linear_shift \
  --scenario linear_shift \
  --num-seeds 20 \
  --core-rules
```

Full Phase-1 sweep:

```bash
SEEDS=20 ./scripts/run_all_mechanism.sh runs/mechanism_v1
```
