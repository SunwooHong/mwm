"""Cross-Entropy Method planner on frozen Dreamer reward + RSSM dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from plastic.logit_plastic_trace import PlasticTrace


def _dist_mode(dist: object) -> torch.Tensor:
    mode = dist.mode  # type: ignore[attr-defined]
    return mode() if callable(mode) else mode


@dataclass
class CEMConfig:
    horizon: int = 12
    num_samples: int = 256
    num_elites: int = 32
    iterations: int = 5
    init_std: float = 0.5
    min_std: float = 0.05
    discount: float = 0.997
    warm_start_actor: bool = True


def _repeat_batch(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.expand(n, *x.shape[1:]) if x.shape[0] == 1 else x.repeat(n, *([1] * (x.ndim - 1)))


def _repeat_trace(trace: PlasticTrace, n: int) -> PlasticTrace:
    return PlasticTrace(
        P=_repeat_batch(trace.P, n),
        ema=_repeat_batch(trace.ema, n),
        var=_repeat_batch(trace.var, n),
        persist=_repeat_batch(trace.persist, n),
        prev_err=None if trace.prev_err is None else _repeat_batch(trace.prev_err, n),
    )


@torch.no_grad()
def score_action_sequences(
    agent,
    stoch: torch.Tensor,
    deter: torch.Tensor,
    trace: PlasticTrace | None,
    actions: torch.Tensor,
    *,
    use_plastic_wm: bool,
    discount: float,
) -> torch.Tensor:
    """Score candidate action sequences with frozen reward head.

    actions: (N, H, A)
    returns: (N,)
    """
    n, horizon, _ = actions.shape
    s = _repeat_batch(stoch, n)
    d = _repeat_batch(deter, n)
    tr = _repeat_trace(trace, n) if trace is not None else None
    total = torch.zeros(n, device=actions.device, dtype=torch.float32)
    factor = 1.0
    for t in range(horizon):
        feat = agent._frozen_rssm.get_feat(s, d)
        rew = _dist_mode(agent._frozen_reward(feat))
        if rew.ndim > 1:
            rew = rew.squeeze(-1)
        total = total + factor * rew
        factor *= discount
        s, d = agent._img_step_plan(s, d, actions[:, t], tr, use_plastic_read=use_plastic_wm)
    return total


@torch.no_grad()
def plan_cem(
    agent,
    state: TensorDict,
    *,
    use_plastic_wm: bool,
    config: CEMConfig,
) -> torch.Tensor:
    """Return planned action (B, A) from current latent state."""
    stoch, deter = state["stoch"], state["deter"]
    trace = agent._unpack_trace(state) if use_plastic_wm and "plastic_P" in state.keys() else None
    b = stoch.shape[0]
    device = stoch.device
    act_dim = agent.act_dim

    mean = torch.zeros(config.horizon, act_dim, device=device)
    std = torch.full((config.horizon, act_dim), config.init_std, device=device)
    if config.warm_start_actor:
        feat = agent._frozen_rssm.get_feat(stoch, deter)
        a0 = _dist_mode(agent._frozen_actor(feat))
        mean[0] = a0[0]

    for _ in range(config.iterations):
        noise = torch.randn(config.num_samples, config.horizon, act_dim, device=device)
        candidates = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1.0, 1.0)
        values = score_action_sequences(
            agent,
            stoch,
            deter,
            trace,
            candidates,
            use_plastic_wm=use_plastic_wm,
            discount=config.discount,
        )
        k = min(config.num_elites, values.shape[0])
        elite_idx = torch.topk(values, k, dim=0).indices
        elite = candidates[elite_idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp(min=config.min_std)

    action = mean[0].unsqueeze(0).expand(b, -1)
    return action
