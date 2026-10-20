# Phase 0 audit — legacy plastic prior vs control

Date: 2026-10-20. Checkpoint files (`*.pt`) are not in this tree, so numerical rollout checks use the code path and deterministic unit tests (`plastic/test_causal_audit.py`).

## Gate

**The current prior residual does not change the online actor input.** Previous return gaps (draft Table 4, +0.068) must not be read as System 1 changing actions. Plastic ON/OFF can still change logged `prior_mse` and the stored matrix `P`.

## Execution path

Online control is `Dreamer.act` → `_observe_step` → `_frozen_actor(get_feat(stoch, deter))`.

1. Encoder embeds the current observation.
2. `_obs_step_core` updates `deter` from the previous stochastic state, previous deter, and previous action. That step does not use the current embedding.
3. Posterior logits are `obs_net([deter, embed])`. The current observation enters here.
4. Prior logits are `img_net(deter)` on that same new deter. They do not see the current embedding.
5. If plastic is on, `x = [previous stoch, previous deter, previous action]` (the `stoch` returned by `_obs_step_core` is the reset-masked previous sample, not the new posterior sample). The residual is added only to a logit copy used for the error.
6. `P` is written from `post_flat - corrected_prior`.
7. The actor feature is `get_feat` of a **posterior sample** and `deter`. The corrected prior is discarded.

`plastic_inference.plastic_step_metrics` does the same thing, then calls `_frozen_actor` on the posterior feature (`plastic_inference.py` around the action line after `plastic_step_metrics`).

`_img_step_plan` can sample a plastic-corrected prior, but `act()` does not call it. It is used by the optional CEM planner, not by the actor-adapt loop that produced Table 4.

## Answers

1. **Does plastic ON/OFF change the actor input under a matched state?** No. Both branches sample `stoch` from the same posterior logits. Any action difference is posterior sampling noise, not the residual.
2. **Does the prior see the current observation?** The prior network does not. `x` uses the previous stochastic sample, not the posterior sample of this observation. The error target is the current posterior, which is the usual one-step filter discrepancy, not a prediction of a future observation scored before that observation is written.
3. **Is the logged error prequential?** The logged MSE is computed before `write` on that step, so it is not fit-after-update. It is still same-step posterior matching, not “predict o_{t+1}, then observe it.”
4. **Size.** Default proprio config is `size12M`: `stoch=32`, `discrete=16`, `deter=2048`. Walker action dim is 6. `K=512`, `d=512+2048+6=2566`, `P` has **1,313,792** floats, **5.0 MiB per environment**, before gate state. That is a full matrix, not a small adapter. Update is one outer product inside `no_grad`.
5. **System 2.** `plastic_actor_adapt` sets `requires_grad` only for names starting with `actor.`. World model and reward/value stay frozen unless a different script turns them on. `actor_lr_scale=1` multiplies the existing optimizer group. Config `model.lr` is **4e-5**, so the table’s `α_actor=1` is a multiplier, not an absolute learning rate of 1.

## MSE definition

`LogitPlasticResidual.write` rescales the signed error so its L2 norm is at most `error_clip_norm=0.5`, then uses that vector in the outer product. The logged `prior_mse` is the mean square of the **unclipped** logit residual. Values above 4 in the draft (Table 6, about 5–6) match unclipped logit MSE, not `||clip(e, 0.5)||^2 / K`, which cannot be large.

## What is allowed next

Do not interpret legacy control returns as a System 1 effect. `plastic/causal_s1.py` is a separate v2: score the prior, then write, and a one-step selector that reads `P` only when explicitly enabled. The legacy modules are unchanged.
