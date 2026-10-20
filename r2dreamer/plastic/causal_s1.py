"""Versioned System 1 (v2): causal prior residual and one-step action selector.

This module does not replace ``LogitPlasticResidual`` or ``Dreamer.act``.
The legacy online path corrects prior logits after the posterior is known and
then feeds the posterior sample, not the corrected prior, to the actor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CausalTrace:
    P: torch.Tensor


class CausalPlasticPrior(nn.Module):
    """Per-environment matrix P in R^{K x d}. Updates never call backward."""

    def __init__(
        self,
        *,
        out_dim: int,
        in_dim: int,
        eta: float = 0.05,
        decay: float = 0.001,
        max_norm: float = 2.0,
        eps: float = 1.0e-6,
        feature_dim: int | None = None,
        proj_seed: int = 0,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.raw_in_dim = int(in_dim)
        self.feature_dim = None if feature_dim is None else int(feature_dim)
        self.eta = float(eta)
        self.decay = float(decay)
        self.max_norm = float(max_norm)
        self.eps = float(eps)
        if self.feature_dim is not None:
            g = torch.Generator()
            g.manual_seed(int(proj_seed))
            proj = torch.randn(self.raw_in_dim, self.feature_dim, generator=g)
            proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-6)
            self.register_buffer("proj", proj)
            self.in_dim = self.feature_dim
        else:
            self.register_buffer("proj", torch.empty(0))
            self.in_dim = self.raw_in_dim

    @property
    def parameter_count(self) -> int:
        return self.out_dim * self.in_dim

    def memory_bytes(self, batch: int) -> int:
        return int(batch * self.parameter_count * 4)

    def initial(self, batch: int, device) -> CausalTrace:
        return CausalTrace(P=torch.zeros(batch, self.out_dim, self.in_dim, device=device))

    def featurize(self, raw: torch.Tensor) -> torch.Tensor:
        raw = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
        if self.feature_dim is None:
            return raw
        return raw @ self.proj.to(device=raw.device, dtype=raw.dtype)

    def read_logits(self, trace: CausalTrace, x: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        corr = torch.einsum("boi,bi->bo", trace.P, x)
        return base_logit + corr

    @staticmethod
    def probs(logit: torch.Tensor, n_classes: int) -> torch.Tensor:
        return torch.softmax(logit.reshape(logit.shape[0], -1, n_classes), dim=-1).reshape(logit.shape[0], -1)

    @torch.no_grad()
    def write(self, trace: CausalTrace, x: torch.Tensor, q_post: torch.Tensor, p_plastic: torch.Tensor) -> CausalTrace:
        assert not x.requires_grad and not q_post.requires_grad
        err = (q_post - p_plastic).detach()
        scale = 1.0 / (x.pow(2).sum(-1, keepdim=True) + self.eps)
        delta = self.eta * torch.einsum("bo,bi->boi", err, x * scale)
        P = (1.0 - self.decay) * trace.P + delta
        flat = torch.linalg.vector_norm(P.reshape(P.shape[0], -1), dim=-1).clamp(min=1e-12)
        P = P * torch.clamp(self.max_norm / flat, max=1.0)[:, None, None]
        return CausalTrace(P=P)

    @staticmethod
    def reset_where(trace: CausalTrace, reset: torch.Tensor) -> CausalTrace:
        mask = reset.reshape(-1).bool()
        if not mask.any():
            return trace
        P = trace.P.clone()
        P[mask] = 0
        return CausalTrace(P=P)


@torch.no_grad()
def prequential_step(
    plastic: CausalPlasticPrior,
    trace: CausalTrace,
    *,
    raw_x: torch.Tensor,
    base_logit: torch.Tensor,
    post_logit: torch.Tensor,
    n_classes: int,
    reset: torch.Tensor | None = None,
) -> tuple[CausalTrace, dict]:
    """Score the prior before the observation is written into P."""
    if reset is not None:
        trace = plastic.reset_where(trace, reset)
    x = plastic.featurize(raw_x)
    pred = plastic.read_logits(trace, x, base_logit)
    q = plastic.probs(post_logit, n_classes)
    p = plastic.probs(pred, n_classes)
    p_base = plastic.probs(base_logit, n_classes)
    kl = (q * (q.clamp(min=1e-8).log() - p.clamp(min=1e-8).log())).sum(-1)
    kl_base = (q * (q.clamp(min=1e-8).log() - p_base.clamp(min=1e-8).log())).sum(-1)
    raw_mse = (post_logit - pred).pow(2).mean(-1)
    metrics = {
        "kl": kl,
        "kl_base": kl_base,
        "raw_mse": raw_mse,
        "correction": torch.linalg.vector_norm(pred - base_logit, dim=-1),
        "trace_norm": torch.linalg.vector_norm(trace.P.reshape(trace.P.shape[0], -1), dim=-1),
    }
    trace = plastic.write(trace, x, q, p)
    return trace, metrics


@torch.no_grad()
def select_action(
    plastic: CausalPlasticPrior,
    trace: CausalTrace,
    *,
    candidates: torch.Tensor,
    score_fn,
    prior_fn,
    read: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score candidate actions. ``read=False`` is the matched no-plastic selector.

    ``prior_fn(action) -> (raw_x, base_logit)`` must not consume the future observation.
    ``score_fn(corrected_logit) -> (B,)`` uses frozen reward/value features.
    """
    scores = []
    for i in range(candidates.shape[1]):
        raw_x, base = prior_fn(candidates[:, i])
        logit = plastic.read_logits(trace, plastic.featurize(raw_x), base) if read else base
        scores.append(score_fn(logit))
    stacked = torch.stack(scores, dim=-1)
    idx = stacked.argmax(dim=-1)
    chosen = candidates[torch.arange(candidates.shape[0], device=candidates.device), idx]
    return chosen, stacked
