# Method contract — System 1 v2 / System 2 (not yet the paper result)

Legacy `LogitPlasticResidual` stays as the audited baseline. It is not this contract.

## System 1 (`plastic/causal_s1.py`)

Per environment, `P ∈ R^{K×d}` with optional fixed projection to 64 features, seeded and not fit on evaluation data.

At step t, using only the previous latent and previous action:

1. Build `x` and the frozen prior logits.
2. Read `logit = prior + P x` if System 1 is on, else `logit = prior`.
3. Record KL / raw logit MSE against the posterior **before** writing.
4. Posterior probabilities come from the observation at this step.
5. `P ← Proj_{||P||_F≤μ}((1-λ)P + η (q − p) xᵀ / (||x||²+ε))` under `no_grad`.
6. Later decisions may read the updated `P`. Shift boundaries do not reset `P`. Episode `is_first` may reset it.

Action path, when used: 4 candidate actions from the current actor, one-step prior each, same selector for ON and OFF, frozen reward/value score, no multi-step plastic imagination. Actor-only control remains a separate reference. A write-only condition updates `P` and does not read it.

## System 2

Unchanged starting point: replay actor updates every 50 environment steps, 4 gradient steps, world model frozen, learning rate `4e-5` times the configured scale (default scale 1).

## Not claimed

Replay-data quality, positive interaction, and consolidation into `θ_wm` are not part of this contract.
