"""Test-time plastic prior-logit residual for Dreamer/R2-Dreamer RSSM.

No trainable parameters. Updates only the plastic trace P at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class PlasticTrace:
    P: torch.Tensor
    ema: torch.Tensor
    var: torch.Tensor
    persist: torch.Tensor
    prev_err: torch.Tensor | None = None


class LogitPlasticResidual(nn.Module):
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
        dir_persist_threshold: float = 0.25,
        dir_persist_temperature: float = 0.08,
        error_min_norm: float = 0.02,
        error_clip_norm: float = 0.5,
        residual_scale: float = 1.0,
        gate_mode: str = "selective",
        flush_k: float = 0.0,
        flush_err_scale: float = 0.15,
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
        self.dir_persist_threshold = float(dir_persist_threshold)
        self.dir_persist_temperature = float(dir_persist_temperature)
        self.error_min_norm = float(error_min_norm)
        self.error_clip_norm = float(error_clip_norm)
        self.residual_scale = float(residual_scale)
        self.gate_mode = str(gate_mode)
        self.flush_k = float(flush_k)
        self.flush_err_scale = float(flush_err_scale)
        self.dir_persist_ema = None

    def initial(self, batch_size: int, device: torch.device | str) -> PlasticTrace:
        P = torch.zeros(batch_size, self.out_dim, self.in_dim, device=device, dtype=torch.float32)
        ema = torch.zeros(batch_size, device=device, dtype=torch.float32)
        var = torch.ones(batch_size, device=device, dtype=torch.float32) * 1.0e-4
        persist = torch.zeros(batch_size, device=device, dtype=torch.float32)
        return PlasticTrace(P=P, ema=ema, var=var, persist=persist, prev_err=None)

    def make_input(self, stoch: torch.Tensor, deter: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = action.shape[0]
        flat_stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        return torch.cat([flat_stoch, deter, action], dim=-1).detach()

    def read(self, trace: PlasticTrace, x: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * torch.einsum("boi,bi->bo", trace.P, x)

    def _direction_gate(self, trace: PlasticTrace, err: torch.Tensor) -> torch.Tensor:
        B = err.shape[0]
        if self.dir_persist_ema is None:
            self.dir_persist_ema = torch.zeros(B, device=err.device, dtype=err.dtype)
        kappa = torch.zeros(B, device=err.device, dtype=err.dtype)
        err_norm = torch.linalg.vector_norm(err, dim=-1)
        if trace.prev_err is not None:
            prev_norm = torch.linalg.vector_norm(trace.prev_err, dim=-1)
            valid = (err_norm >= self.error_min_norm) & (prev_norm >= self.error_min_norm)
            if valid.any():
                cos = (err * trace.prev_err).sum(-1) / (err_norm * prev_norm).clamp(min=1.0e-12)
                kappa = torch.clamp(cos, min=0.0) * valid.float()
        self.dir_persist_ema = self.persist_beta * self.dir_persist_ema + (1.0 - self.persist_beta) * kappa
        return torch.sigmoid((self.dir_persist_ema - self.dir_persist_threshold) / self.dir_persist_temperature)

    @torch.no_grad()
    def write(
        self,
        trace: PlasticTrace,
        x: torch.Tensor,
        signed_error: torch.Tensor,
        *,
        flush_boost: float = 0.0,
    ) -> Tuple[PlasticTrace, torch.Tensor, torch.Tensor]:
        err = signed_error.detach()
        err_norm = torch.linalg.vector_norm(err, dim=-1, keepdim=True)
        err = err * torch.clamp(self.error_clip_norm / err_norm.clamp(min=1.0e-12), max=1.0)

        surprise = err.pow(2).mean(dim=-1)
        std = torch.sqrt(torch.clamp(trace.var, min=1.0e-8))
        z = (surprise - trace.ema) / std
        raw_gate = torch.sigmoid((z - self.gate_z_threshold) / self.gate_temperature)
        persist = self.persist_beta * trace.persist + (1.0 - self.persist_beta) * raw_gate
        gate_surprise = torch.sigmoid((persist - self.persist_threshold) / self.persist_temperature)
        if self.gate_mode == "always_on":
            gate = torch.ones_like(surprise)
        elif self.gate_mode == "surprise_only":
            gate = gate_surprise
        else:
            gate_dir = self._direction_gate(trace, err)
            gate = gate_surprise * gate_dir

        dP = self.lr * gate[:, None, None] * torch.einsum("bo,bi->boi", err, x)
        per_sample_err = torch.linalg.vector_norm(err, dim=-1)
        decay_boost = self.flush_k * torch.sigmoid(-per_sample_err / self.flush_err_scale)
        if flush_boost > 0.0:
            decay_boost = decay_boost + flush_boost
        decay_eff = self.decay + decay_boost
        dP = dP - decay_eff[:, None, None] * trace.P
        P = trace.P + dP

        flat_norm = torch.linalg.vector_norm(P.reshape(P.shape[0], -1), dim=-1).clamp(min=1.0e-12)
        P = P * torch.clamp(self.max_norm / flat_norm, max=1.0)[:, None, None]

        diff = surprise - trace.ema
        ema = self.gate_ema_beta * trace.ema + (1.0 - self.gate_ema_beta) * surprise
        var = self.gate_var_beta * trace.var + (1.0 - self.gate_var_beta) * diff.pow(2)

        prev_err = err.clone() if err_norm.min().item() >= self.error_min_norm else trace.prev_err
        return (
            PlasticTrace(P=P.detach(), ema=ema.detach(), var=var.detach(), persist=persist.detach(), prev_err=prev_err),
            gate.detach(),
            decay_eff.detach(),
        )

    def reset_dir_persist_where(self, reset: torch.Tensor) -> None:
        if self.dir_persist_ema is None:
            return
        mask = reset.reshape(-1).bool()
        if mask.any():
            if self.dir_persist_ema.shape[0] != mask.shape[0]:
                self.dir_persist_ema = torch.zeros(mask.shape[0], device=mask.device, dtype=torch.float32)
            self.dir_persist_ema[mask] = 0.0

    @staticmethod
    def clone_trace(trace: PlasticTrace) -> PlasticTrace:
        """Detached copy for static use during imagination (no online writes)."""
        prev_err = None if trace.prev_err is None else trace.prev_err.detach().clone()
        return PlasticTrace(
            P=trace.P.detach().clone(),
            ema=trace.ema.detach().clone(),
            var=trace.var.detach().clone(),
            persist=trace.persist.detach().clone(),
            prev_err=prev_err,
        )

    @staticmethod
    def reset_where(trace: PlasticTrace, reset: torch.Tensor) -> PlasticTrace:
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
        prev_err = None if trace.prev_err is None else trace.prev_err.clone()
        if prev_err is not None:
            prev_err[mask] = 0.0
        return PlasticTrace(P=P, ema=ema, var=var, persist=persist, prev_err=prev_err)
