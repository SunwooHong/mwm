from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .config import ExperimentConfig
from .dynamics import Batch, base_predict


def fro_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(x)


def project_frobenius(P: torch.Tensor, max_norm: float) -> torch.Tensor:
    nrm = torch.linalg.matrix_norm(P)
    if nrm <= max_norm:
        return P
    return P * (max_norm / torch.clamp(nrm, min=1.0e-12))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    denom = torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv)
    if denom.item() <= 1.0e-12:
        return 0.0
    return float((av @ bv / denom).clamp(-1.0, 1.0).item())


@dataclass
class GateState:
    """Online gate state for surprise and persistence."""

    ema: Optional[torch.Tensor] = None
    var: Optional[torch.Tensor] = None
    persist: float = 0.0

    def raw_gate(self, surprise: torch.Tensor, cfg: ExperimentConfig) -> float:
        # Initialize with the first surprise, which usually occurs in the nominal phase.
        if self.ema is None or self.var is None:
            self.ema = surprise.detach().clone()
            self.var = torch.ones_like(surprise) * 1.0e-4

        std = torch.sqrt(torch.clamp(self.var, min=1.0e-8))
        z = (surprise - self.ema) / std
        raw = torch.sigmoid((z - cfg.gate_z_threshold) / cfg.gate_temperature)

        # Update statistics after computing the gate.
        diff = surprise.detach() - self.ema
        self.ema = cfg.gate_ema_beta * self.ema + (1.0 - cfg.gate_ema_beta) * surprise.detach()
        self.var = cfg.gate_var_beta * self.var + (1.0 - cfg.gate_var_beta) * diff.detach().pow(2)
        return float(raw.clamp(0.0, 1.0).item())

    def persistent_gate(self, raw_gate: float, cfg: ExperimentConfig) -> float:
        self.persist = cfg.persist_beta * self.persist + (1.0 - cfg.persist_beta) * raw_gate
        gate = torch.sigmoid(torch.tensor((self.persist - cfg.persist_threshold) / cfg.persist_temperature))
        return float(gate.clamp(0.0, 1.0).item())


@dataclass
class RuleState:
    P: torch.Tensor
    gate: GateState


class PlasticRule:
    """One local plasticity rule.

    All rules update only P_t, the residual matrix. M0 is frozen.
    """

    def __init__(self, name: str, cfg: ExperimentConfig, seed: int):
        self.name = name
        self.cfg = cfg
        self.rng = torch.Generator(device=cfg.device).manual_seed(seed + 424242)

    def init_state(self) -> RuleState:
        return RuleState(
            P=torch.zeros(self.cfg.dz, self.cfg.dx, device=self.cfg.device),
            gate=GateState(),
        )

    def update(
        self,
        state: RuleState,
        *,
        M0: torch.Tensor,
        x: torch.Tensor,
        obs_next: torch.Tensor,
        oracle_batch: Optional[Batch] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Return dP and gate value.

        This method does not mutate P. GateState is updated online, because the gate
        represents an observable test-time controller.
        """
        cfg = self.cfg
        P = state.P
        base = base_predict(M0, x)
        pred_residual = P @ x
        residual_target = obs_next - base
        err = residual_target - pred_residual
        surprise = err.pow(2).mean().detach()
        raw_gate = state.gate.raw_gate(surprise, cfg)
        persistent_gate = state.gate.persistent_gate(raw_gate, cfg)

        if self.name == "frozen":
            return torch.zeros_like(P), 0.0

        if self.name == "random":
            dP = cfg.lr_random * torch.randn(P.shape, generator=self.rng, device=cfg.device) - cfg.decay * P
            return dP, 1.0

        if self.name == "pure_hebb":
            # Co-activation only. It observes post activity but not signed prediction error.
            post = obs_next.detach()
            dP = cfg.lr_hebb * torch.outer(post, x) - cfg.decay * P
            return dP, 1.0

        if self.name == "oja":
            post = obs_next.detach()
            hebb = torch.outer(post, x)
            oja = post.pow(2).unsqueeze(1) * P
            dP = cfg.lr_hebb * (hebb - cfg.oja_alpha * oja) - cfg.decay * P
            return dP, 1.0

        if self.name == "surprise_hebb":
            post = obs_next.detach()
            hebb = torch.outer(post, x)
            dP = cfg.lr_hebb * raw_gate * hebb - cfg.decay * P
            return dP, raw_gate

        if self.name == "error_hebb":
            dP = cfg.lr_error * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, 1.0

        if self.name == "gated_error_hebb":
            dP = cfg.lr_error * persistent_gate * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, persistent_gate

        if self.name == "error_hebb_oja":
            # Error-corrective update plus row-wise Oja-style stabilization.
            corr = torch.outer(err.detach(), x)
            oja = err.detach().pow(2).unsqueeze(1) * P
            dP = cfg.lr_error * (corr - cfg.oja_alpha * oja) - cfg.decay * P
            return dP, 1.0

        if self.name == "error_hebb_rollback":
            # Rollback is applied by the experiment loop after evaluating held-out loss.
            dP = cfg.lr_error * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, 1.0

        if self.name == "sgd_oracle":
            if oracle_batch is None:
                raise ValueError("sgd_oracle requires oracle_batch.")
            base_b = base_predict(M0, oracle_batch.x)
            pred_res_b = oracle_batch.x @ P.T
            target_res_b = oracle_batch.clean_next - base_b
            err_b = target_res_b - pred_res_b
            # Negative gradient of 0.5 * mean ||err||^2 wrt P, up to constant scaling.
            grad_like = err_b.T @ oracle_batch.x / oracle_batch.x.shape[0]
            dP = cfg.lr_oracle * grad_like - cfg.decay * P
            return dP, 1.0

        raise ValueError(f"Unknown rule: {self.name}")


RULES = [
    "frozen",
    "random",
    "pure_hebb",
    "oja",
    "surprise_hebb",
    "error_hebb",
    "gated_error_hebb",
    "error_hebb_oja",
    "error_hebb_rollback",
    "sgd_oracle",
]
