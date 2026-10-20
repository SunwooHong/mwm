# Plasticity Unit Test

Synthetic mechanism test for **Hebbian-like / error-modulated local plasticity** before integrating into DreamerV3 or R2-Dreamer.

The test freezes a nominal transition model `M0` and gives each rule only a plastic residual matrix `P_t`:

```text
z_{t+1} = (M0 + ΔM_regime) [z_t; a_t] + noise
z_hat    = M0 [z_t; a_t] + P_t [z_t; a_t]
```

The central question is whether a local update `ΔP_t` reduces held-out prediction loss without backpropagating through the frozen model.

## Install

```bash
pip install -r requirements.txt
```

## Run the main linear-shift unit test

```bash
python run_unit_test.py --outdir runs/linear_shift --scenario linear_shift --num-seeds 5
```

## Run poison/outlier and distractor tests

```bash
python run_unit_test.py --outdir runs/poison --scenario poison --num-seeds 5
python run_unit_test.py --outdir runs/distractor --scenario distractor --num-seeds 5
```

## Outputs

Each run writes:

```text
metrics.csv                 # per-step metrics
summary.csv                 # rule-level summary
config.json                 # config used for the run
plots/*.png                 # diagnostic plots
```

Key metrics:

- `candidate_delta_val`: held-out validation loss after candidate update minus before update. Negative is good.
- `beneficial_update_rate`: fraction of candidate updates with negative held-out Δ loss.
- `mean_applied_delta_val`: held-out Δ loss after rollback/projection is applied.
- `applied_beneficial_rate`: fraction of actually applied updates with negative held-out Δ loss.
- `grad_alignment`: cosine similarity between local update and true negative per-sample gradient.
- `p_error_to_true_delta`: Frobenius distance between `P_t` and the known regime residual `ΔM`.
- `trace_norm`: Frobenius norm of `P_t`.
- `false_write_rate_gate_gt_0_5`: in poison/distractor contexts, how often the gate opens above 0.5.

## Implemented rules

- `frozen`: no plasticity.
- `random`: random trace perturbation control.
- `pure_hebb`: co-activation update using observed post activity, no signed prediction error.
- `oja`: pure Hebbian plus row-wise Oja-style stabilization.
- `surprise_hebb`: scalar surprise opens a gate, but direction is still co-activation.
- `error_hebb`: signed prediction error times pre-synaptic activity.
- `gated_error_hebb`: persistence-gated signed error-Hebbian update.
- `error_hebb_oja`: error-Hebbian plus Oja-style stabilization.
- `error_hebb_rollback`: error-Hebbian candidate update with held-out rollback for mechanism debugging.
- `sgd_oracle`: mini-batch supervised residual update used as an upper bound.

## What counts as a pass?

A rule is mechanism-valid only if it shows:

```text
mean(candidate_delta_val) < 0
beneficial_update_rate > 0.6
mean_grad_alignment > 0
bounded trace_norm
low false_write_rate in poison/distractor tests
```

For the first paper figure, compare:

```text
pure_hebb / oja / surprise_hebb / error_hebb / sgd_oracle
```

If `error_hebb` is clearly positive while pure co-activation is weak or unstable, the paper should frame the mechanism as **bounded prediction-error-modulated local plasticity**, not plain Hebbian learning.
