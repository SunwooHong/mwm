# Phase 2: Non-Oracle Selective Plasticity

## Goal

Replace oracle held-out validation rollback with **deployment-realistic proxies** and show:

1. Naive signed error update (`error_hebb`) adapts fast but **false-learns** poison/distractor shifts.
2. **Persistence gate** (signed-error direction κ_t) reduces one-off writes.
3. **Anchor-memory rollback** rejects updates that harm recent low-loss transitions.
4. **Consistency rollback** rejects harmful updates using local trace/residual/error proxies only.
5. **`full_selective_v2`** combines the above without held-out validation rollback.

## Main rule (Phase 2)

`full_selective_v2`:

```text
signed prediction-error update
+ direction-aware persistence gate (surprise + κ_t)
+ robust error clipping
+ anchor-memory rollback
+ consistency rollback
+ optional synthetic task mask (nuisance test only)
```

## Rollback proxy comparison rules

| Rule | Write gate | Anchor rollback | Consistency rollback | Oracle rollback |
|---|---|---|---|---|
| `error_hebb` | none | no | no | no |
| `persistent_error_hebb` | direction + surprise | no | no | no |
| `error_hebb_anchor_rollback` | none | yes | no | no |
| `error_hebb_consistency_rollback` | none | no | yes | no |
| `gated_error_hebb_anchor_rollback` | direction + surprise | yes | no | no |
| `full_selective_v2` | direction + surprise | yes | yes | no |
| `error_hebb_oracle_rollback` | none | no | no | yes (diagnostic) |

## Consistency rollback signals (no clean labels)

- low gate + large write norm
- trace step norm jump
- residual output jump on current x
- signed-error direction flip under elevated surprise
- observed-loss spike vs EMA when gate is low

## Pass criteria (robustness)

On `poison_outlier` and `nuisance_distractor`:

- `full_selective_v2` false-context write norm ≤ 50% of naive `error_hebb`
- validation/core loss near frozen (within tolerance)
- clean/core loss lower than naive `error_hebb`

On adaptation scenarios (`linear_shift`, etc.):

- `full_selective_v2` still passes Phase 1 mechanism checks (alignment, ΔL_val, recovery)

## Run

```bash
SEEDS=10 ./scripts/run_phase2.sh runs/phase2
```

Paper-scale:

```bash
SEEDS=20 ./scripts/run_phase2.sh runs/phase2_paper
```

Outputs:

- per-scenario `summary_gate.md`
- `phase2_aggregate_summary.csv`
- `PHASE2_SUMMARY.md`

## Next step after Phase 2 pass

R2-Dreamer prior-logit residual patch with the same non-oracle proxies (no held-out val rollback at deployment).
