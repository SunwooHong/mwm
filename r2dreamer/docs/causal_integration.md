# Causal integration

`Dreamer.act` is unchanged. MWM v2 is `Dreamer.act_causal`, enabled with `enable_causal_s1(feature_dim=64)`.

Order inside one call:

1. Posterior from the current observation.
2. If the previous call stored a pending prior, score KL against this posterior, then write `P`. That pending prior was computed before this observation existed.
3. Four candidates: actor mode plus three matched Gaussian perturbations (`cand_seed`). Candidate 0 is the mode.
4. One-step RSSM prior per candidate. Residual is added only when `read_plastic=True`.
5. Score `reward.mode + cont.mean * value.mode` with frozen heads.
6. Store the chosen prior and features for the next observation.

`write_plastic=False` keeps `P` fixed (no-write / stale-memory control).

`clone_and_freeze` sets `_frozen_actor` parameter `.data` to the same storage as `actor`. A CPU load of this checkpoint reports `shared True`, so actor-adapt updates are visible to the selector without a separate copy. The selector reads `self.actor`.

CPU smoke, two environment steps, projection 64:

- Plastic matrix: 32,768 floats, 128 KiB per environment.
- First-step score span: 13.14 (not constant). The selector did not pick the actor mode.
- Second-step pre-update KL: 8.76, trace norm 0.67, score span 2.21.

That smoke checks wiring and action sensitivity. It is not a gain-shift result, and a changed action is not an improved action.
