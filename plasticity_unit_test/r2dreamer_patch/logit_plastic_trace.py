"""First-pass plastic prior-logit residual for R2-Dreamer / DreamerV3-style RSSM.

This is intentionally a patch skeleton, not a drop-in replacement. It avoids
changing train-time gradients. The trace is test-time state, not an nn.Parameter.

Recommended first integration target:
  - Add a residual to RSSM prior logits, not to deterministic state.
  - In obs_step, posterior logits are available as a signed local target.
  - Update P_t with: ΔP = η * gate * outer(post_logit - prior_logit, x) - decay * P.

Why prior-logit residual first?
  Standard Dreamer/R2-Dreamer RSSMs do not produce a separate posterior deter
  target; posterior information enters through stochastic-state logits. Prior
  logits therefore give a cleaner signed prediction-error target for the first
  mechanism test inside Dreamer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class PlasticTrace:
    """Per-environment plastic state.

    P:       [B, O, I] residual matrix, where O = stoch * discrete and
             I = flat_stoch + deter + action_dim.
    ema:     [B] running surprise mean.
    var:     [B] running surprise variance.
    persist: [B] persistence accumulator for rejecting one-off spikes.
    """

    P: torch.Tensor
    ema: torch.Tensor
    var: torch.Tensor
    persist: torch.Tensor


class LogitPlasticResidual(nn.Module):
    """Stateful local plastic residual for RSSM prior logits.

    This module has no trainable parameters by default. It only provides methods
    to initialize, read from, and write to a PlasticTrace object.
    """

    def __init__(
        self,
        *,
        flat_stoch: int,
        deter: int,
        act_dim: int,
        out_dim: int,
        lr: float = 0.01,
        decay: float = 0.001,
        max_norm: float = 2.0,
        gate_ema_beta: float = 0.99,
        gate_var_beta: float = 0.99,
        gate_z_threshold: float = 1.5,
        gate_temperature: float = 0.5,
        persist_beta: float = 0.94,
        persist_threshold: float = 0.35,
        persist_temperature: float = 0.07,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.flat_stoch = int(flat_stoch)
        self.deter = int(deter)
        self.act_dim = int(act_dim)
        self.in_dim = self.flat_stoch + self.deter + self.act_dim
        self.out_dim = int(out_dim)
        self.lr = float(lr)
        self.decay = float(decay)
        self.max_norm = float(max_norm)
        self.gate_ema_beta = float(gate_ema_beta)
        self.gate_var_beta = float(gate_var_beta)
        self.gate_z_threshold = float(gate_z_threshold)
        self.gate_temperature = float(gate_temperature)
        self.persist_beta = float(persist_beta)
        self.persist_threshold = float(persist_threshold)
        self.persist_temperature = float(persist_temperature)
        self.residual_scale = float(residual_scale)

    def initial(self, batch_size: int, device: torch.device | str) -> PlasticTrace:
        P = torch.zeros(batch_size, self.out_dim, self.in_dim, device=device, dtype=torch.float32)
        ema = torch.zeros(batch_size, device=device, dtype=torch.float32)
        var = torch.ones(batch_size, device=device, dtype=torch.float32) * 1.0e-4
        persist = torch.zeros(batch_size, device=device, dtype=torch.float32)
        return PlasticTrace(P=P, ema=ema, var=var, persist=persist)

    def make_input(self, stoch: torch.Tensor, deter: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Build local pre-synaptic vector x = [flat_stoch, deter, action]."""
        B = action.shape[0]
        flat_stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        return torch.cat([flat_stoch, deter, action], dim=-1).detach()

    def read(self, trace: PlasticTrace, x: torch.Tensor) -> torch.Tensor:
        """Return residual prior-logit vector [B, O]."""
        residual = torch.einsum("boi,bi->bo", trace.P, x)
        return self.residual_scale * residual

    @torch.no_grad()
    def write(self, trace: PlasticTrace, x: torch.Tensor, signed_error: torch.Tensor) -> Tuple[PlasticTrace, torch.Tensor]:
        """Apply prediction-error-modulated local plasticity.

        signed_error should be stopgrad(post_logit - prior_logit), flattened to [B, O].
        Returns updated trace and the gate [B].
        """
        err = signed_error.detach()
        surprise = err.pow(2).mean(dim=-1)

        std = torch.sqrt(torch.clamp(trace.var, min=1.0e-8))
        z = (surprise - trace.ema) / std
        raw_gate = torch.sigmoid((z - self.gate_z_threshold) / self.gate_temperature)
        persist = self.persist_beta * trace.persist + (1.0 - self.persist_beta) * raw_gate
        gate = torch.sigmoid((persist - self.persist_threshold) / self.persist_temperature)

        dP = self.lr * gate[:, None, None] * torch.einsum("bo,bi->boi", err, x)
        dP = dP - self.decay * trace.P
        P = trace.P + dP

        # Per-batch Frobenius projection.
        norm = torch.linalg.vector_norm(P.reshape(P.shape[0], -1), dim=-1).clamp(min=1.0e-12)
        scale = torch.clamp(self.max_norm / norm, max=1.0)
        P = P * scale[:, None, None]

        diff = surprise - trace.ema
        ema = self.gate_ema_beta * trace.ema + (1.0 - self.gate_ema_beta) * surprise
        var = self.gate_var_beta * trace.var + (1.0 - self.gate_var_beta) * diff.pow(2)

        return PlasticTrace(P=P.detach(), ema=ema.detach(), var=var.detach(), persist=persist.detach()), gate.detach()

    @staticmethod
    def reset_where(trace: PlasticTrace, reset: torch.Tensor) -> PlasticTrace:
        """Zero trace state for envs where reset=True."""
        mask = reset.reshape(-1).bool()
        if not mask.any():
            return trace
        P = trace.P.clone()
        ema = trace.ema.clone()
        var = trace.var.clone()
        persist = trace.persist.clone()
        P[mask] = 0.0
        ema[mask] = 0.0
        var[mask] = 1.0e-4
        persist[mask] = 0.0
        return PlasticTrace(P=P, ema=ema, var=var, persist=persist)
