# Phase 0: Mechanism Claim Lock

## Locked claim

The project should not claim that a pure Hebbian/Oja co-activation rule is sufficient for world-model correction.

The Phase-0 mechanism claim is:

> A frozen transition/world-model prior can be recalibrated at test time by a small bounded residual trace updated with a local, signed prediction-error rule. Selective gates and rollback proxies decide when the residual is written, preserved, or rejected.

## Main update rule

For a frozen nominal transition model

```math
\hat z_{t+1} = M_0 x_t + P_t x_t,
```

where `x_t = [z_t; a_t]`, only the residual trace `P_t` is plastic.

The main local update is:

```math
\Delta P_t = \eta\, g_t\, e_t x_t^\top - \lambda P_t,
```

where:

- `e_t = z_{t+1}^{obs} - (M_0 + P_t)x_t` is a signed prediction error.
- `g_t` is a selective write gate.
- `lambda` is decay.
- norm projection bounds the trace.

This is Hebbian-like only in the sense that it is local and outer-product based. It is better described as **prediction-error-modulated local delta plasticity** or **three-factor local plasticity**.

## Role of Oja/homeostasis

Oja-like terms, decay, norm projection, and homeostatic shutoff are stability devices. They are not the primary error-correction direction.

The key empirical question is whether the update direction aligns with the true negative prediction-loss gradient:

```math
\cos(\Delta P_t, -\nabla_{P_t} L_t) > 0.
```

## Phase-1 pass gate

Before touching Dreamer/RL, synthetic experiments must show:

1. Signed error-plasticity lowers held-out transition loss.
2. Pure Hebbian/Oja and surprise-only Hebbian do not explain the result.
3. Persistence gate / anchor rollback reduce harmful writes under poison or nuisance-only shifts.
4. Trace norm remains bounded.

If these fail, DreamerV3 integration is not diagnostic.
