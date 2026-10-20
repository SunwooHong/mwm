from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    """Online gate state for surprise and temporal persistence."""

    ema: Optional[torch.Tensor] = None
    var: Optional[torch.Tensor] = None
    persist: float = 0.0
    dir_persist_ema: float = 0.0
    prev_err: Optional[torch.Tensor] = None
    ema_obs_loss: float = 0.0
    ema_surprise: float = 0.0
    step_count: int = 0

    def raw_gate(self, surprise: torch.Tensor, cfg: ExperimentConfig) -> float:
        # Initialize with first surprise. This usually occurs in nominal phase.
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
        surprise_val = float(surprise.item())
        self.ema_surprise = cfg.consistency_ema_beta * self.ema_surprise + (1.0 - cfg.consistency_ema_beta) * surprise_val
        self.step_count += 1
        return float(raw.clamp(0.0, 1.0).item())

    def direction_persistence(self, err: torch.Tensor, cfg: ExperimentConfig) -> float:
        """kappa_t: repeated signed-error direction, not scalar surprise alone."""
        err_norm = torch.linalg.vector_norm(err)
        if err_norm.item() < cfg.error_min_norm or self.prev_err is None:
            kappa = 0.0
        else:
            prev_norm = torch.linalg.vector_norm(self.prev_err)
            if prev_norm.item() < cfg.error_min_norm:
                kappa = 0.0
            else:
                kappa = max(0.0, float((err @ self.prev_err / (err_norm * prev_norm)).clamp(-1.0, 1.0).item()))
        self.dir_persist_ema = cfg.persist_beta * self.dir_persist_ema + (1.0 - cfg.persist_beta) * kappa
        return kappa

    def commit_err(self, err: torch.Tensor, cfg: ExperimentConfig) -> None:
        if torch.linalg.vector_norm(err).item() >= cfg.error_min_norm:
            self.prev_err = err.detach().clone()

    def persistent_gate(self, raw_gate: float, kappa: float, cfg: ExperimentConfig) -> float:
        """Combine surprise persistence with signed-error direction persistence."""
        self.persist = cfg.persist_beta * self.persist + (1.0 - cfg.persist_beta) * raw_gate
        gate_surprise = torch.sigmoid(torch.tensor((self.persist - cfg.persist_threshold) / cfg.persist_temperature))
        gate_direction = torch.sigmoid(
            torch.tensor((self.dir_persist_ema - cfg.dir_persist_threshold) / cfg.dir_persist_temperature)
        )
        gate = gate_surprise * gate_direction
        return float(gate.clamp(0.0, 1.0).item())

    def update_obs_loss_ema(self, obs_loss: float, cfg: ExperimentConfig) -> None:
        self.ema_obs_loss = cfg.consistency_ema_beta * self.ema_obs_loss + (1.0 - cfg.consistency_ema_beta) * obs_loss


@dataclass
class AnchorMemory:
    """Small FIFO memory of low-observed-loss transitions.

    This is a non-oracle rollback proxy: it stores x and the observed next latent
    for transitions that the current model already predicts well. It does not use
    clean labels, regime labels, or future held-out batches.
    """

    xs: List[torch.Tensor]
    ys: List[torch.Tensor]

    def maybe_add(self, x: torch.Tensor, obs_next: torch.Tensor, obs_loss: float, cfg: ExperimentConfig) -> None:
        if obs_loss > cfg.anchor_add_loss_threshold:
            return
        self.xs.append(x.detach().clone())
        self.ys.append(obs_next.detach().clone())
        if len(self.xs) > cfg.anchor_size:
            self.xs = self.xs[-cfg.anchor_size:]
            self.ys = self.ys[-cfg.anchor_size:]

    def __len__(self) -> int:
        return len(self.xs)

    def loss(self, M0: torch.Tensor, P: torch.Tensor) -> Optional[float]:
        if not self.xs:
            return None
        x = torch.stack(self.xs, dim=0)
        y = torch.stack(self.ys, dim=0)
        pred = x @ (M0 + P).T
        return float((pred - y).pow(2).mean().item())


@dataclass
class RuleState:
    P: torch.Tensor
    gate: GateState
    anchor: AnchorMemory


class PlasticRule:
    """One local plasticity rule.

    All rules update only P_t, the residual matrix. M0 is frozen. The main
    mechanism is the signed prediction-error local delta update:

        ΔP_t = η g_t e_t x_t^T - λP_t.

    Pure Hebbian/Oja variants are kept as mechanistic ablations.
    """

    def __init__(self, name: str, cfg: ExperimentConfig, seed: int):
        self.name = name
        self.cfg = cfg
        self.rng = torch.Generator(device=cfg.device).manual_seed(seed + 424242)

    def init_state(self) -> RuleState:
        return RuleState(
            P=torch.zeros(self.cfg.dz, self.cfg.dx, device=self.cfg.device),
            gate=GateState(),
            anchor=AnchorMemory(xs=[], ys=[]),
        )

    @staticmethod
    def uses_anchor_rollback(name: str) -> bool:
        return name in {
            "error_hebb_anchor_rollback",
            "gated_error_hebb_anchor_rollback",
            "full_selective",
            "full_selective_v2",
        }

    @staticmethod
    def uses_consistency_rollback(name: str) -> bool:
        return name in {"error_hebb_consistency_rollback", "full_selective_v2"}

    @staticmethod
    def uses_oracle_rollback(name: str) -> bool:
        return name in {"error_hebb_oracle_rollback", "error_hebb_rollback"}

    def update(
        self,
        state: RuleState,
        *,
        M0: torch.Tensor,
        x: torch.Tensor,
        obs_next: torch.Tensor,
        oracle_batch: Optional[Batch] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """Return dP, raw gate, applied gate.

        This method does not mutate P. GateState is updated online, because the
        gate is an observable test-time controller.
        """
        cfg = self.cfg
        P = state.P
        base = base_predict(M0, x)
        pred_residual = P @ x
        residual_target = obs_next - base
        err = residual_target - pred_residual
        surprise = err.pow(2).mean().detach()
        raw_gate = state.gate.raw_gate(surprise, cfg)
        kappa = state.gate.direction_persistence(err.detach(), cfg)
        persistent_gate = state.gate.persistent_gate(raw_gate, kappa, cfg)

        # Synthetic task-relevance mask for the nuisance_distractor unit test.
        # This is intentionally explicit: Phase 1 tests whether a mask would solve
        # false nuisance learning; later Dreamer experiments must learn/estimate it.
        err_task = err
        if self.name == "full_selective" and cfg.canonical_scenario == "nuisance_distractor":
            err_task = err.clone()
            err_task[cfg.core_dims:] = 0.0
        if self.name == "full_selective_v2" and cfg.canonical_scenario == "nuisance_distractor":
            err_task = err.clone()
            err_task[cfg.core_dims:] = 0.0

        if self.name == "frozen":
            return torch.zeros_like(P), raw_gate, 0.0

        if self.name == "random":
            dP = cfg.lr_random * torch.randn(P.shape, generator=self.rng, device=cfg.device) - cfg.decay * P
            return dP, raw_gate, 1.0

        if self.name == "pure_hebb":
            # Co-activation only. It observes post activity but not signed prediction error.
            post = obs_next.detach()
            dP = cfg.lr_hebb * torch.outer(post, x) - cfg.decay * P
            return dP, raw_gate, 1.0

        if self.name == "oja":
            post = obs_next.detach()
            hebb = torch.outer(post, x)
            oja = post.pow(2).unsqueeze(1) * P
            dP = cfg.lr_hebb * (hebb - cfg.oja_alpha * oja) - cfg.decay * P
            return dP, raw_gate, 1.0

        if self.name == "surprise_hebb":
            post = obs_next.detach()
            hebb = torch.outer(post, x)
            dP = cfg.lr_hebb * raw_gate * hebb - cfg.decay * P
            return dP, raw_gate, raw_gate

        if self.name == "error_hebb":
            dP = cfg.lr_error * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, raw_gate, 1.0

        if self.name in {"gated_error_hebb", "persistent_error_hebb"}:
            dP = cfg.lr_error * persistent_gate * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, raw_gate, persistent_gate

        if self.name == "error_hebb_oja":
            # Error-corrective update plus row-wise Oja-style stabilization.
            corr = torch.outer(err.detach(), x)
            oja = err.detach().pow(2).unsqueeze(1) * P
            dP = cfg.lr_error * (corr - cfg.oja_alpha * oja) - cfg.decay * P
            return dP, raw_gate, 1.0

        if self.name in {"error_hebb_oracle_rollback", "error_hebb_rollback", "error_hebb_anchor_rollback"}:
            dP = cfg.lr_error * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, raw_gate, 1.0

        if self.name == "error_hebb_consistency_rollback":
            dP = cfg.lr_error * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, raw_gate, 1.0

        if self.name == "gated_error_hebb_anchor_rollback":
            dP = cfg.lr_error * persistent_gate * torch.outer(err.detach(), x) - cfg.decay * P
            return dP, raw_gate, persistent_gate

        if self.name in {"full_selective", "full_selective_v2"}:
            # Robust local update: clip the signed error vector so one-off poisoned
            # observations cannot dominate the trace. This is still local and
            # backprop-free; it is a stability/selectivity device, not an oracle.
            err_eff = err_task.detach()
            err_norm = torch.linalg.vector_norm(err_eff)
            if err_norm > cfg.error_clip_norm:
                err_eff = err_eff * (cfg.error_clip_norm / torch.clamp(err_norm, min=1.0e-12))
            dP = cfg.lr_error * persistent_gate * torch.outer(err_eff, x) - cfg.decay * P
            return dP, raw_gate, persistent_gate

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
            return dP, raw_gate, 1.0

        raise ValueError(f"Unknown rule: {self.name}")


def should_consistency_rollback(
    cfg: ExperimentConfig,
    state: RuleState,
    *,
    P_before: torch.Tensor,
    candidate: torch.Tensor,
    dP: torch.Tensor,
    x: torch.Tensor,
    err: torch.Tensor,
    applied_gate: float,
    obs_loss: float,
) -> Tuple[bool, str]:
    """Non-oracle rollback using local consistency proxies only."""
    if state.gate.step_count < cfg.consistency_warmup_steps:
        return False, "none"

    write_norm = float(torch.linalg.matrix_norm(dP).item())
    trace_step = float(torch.linalg.matrix_norm(candidate - P_before).item())
    residual_jump = float(torch.linalg.vector_norm((candidate - P_before) @ x).item())
    surprise = float(err.pow(2).mean().item())

    if applied_gate < cfg.consistency_min_gate and write_norm > cfg.consistency_min_write_norm:
        return True, "low_gate_write"
    if trace_step > cfg.consistency_max_trace_step:
        return True, "trace_step"
    if residual_jump > cfg.consistency_max_residual_jump:
        return True, "residual_jump"

    if state.gate.prev_err is not None and state.gate.ema_surprise > 1.0e-8:
        prev = state.gate.prev_err
        err_norm = torch.linalg.vector_norm(err)
        prev_norm = torch.linalg.vector_norm(prev)
        if err_norm.item() >= cfg.error_min_norm and prev_norm.item() >= cfg.error_min_norm:
            cos = float((err @ prev / (err_norm * prev_norm)).item())
            if cos < cfg.consistency_min_direction_cos and surprise > cfg.consistency_surprise_mult * state.gate.ema_surprise:
                return True, "direction_flip"

    if state.gate.ema_obs_loss > 1.0e-8:
        if obs_loss > state.gate.ema_obs_loss * (1.0 + cfg.consistency_loss_spike) and applied_gate < 0.5:
            return True, "obs_loss_spike"

    return False, "none"


RULES = [
    "frozen",
    "random",
    "pure_hebb",
    "oja",
    "surprise_hebb",
    "error_hebb",
    "persistent_error_hebb",
    "error_hebb_oja",
    "error_hebb_anchor_rollback",
    "error_hebb_consistency_rollback",
    "gated_error_hebb_anchor_rollback",
    "full_selective",
    "full_selective_v2",
    "error_hebb_oracle_rollback",
    "sgd_oracle",
]

CORE_RULES = [
    "frozen",
    "pure_hebb",
    "oja",
    "surprise_hebb",
    "error_hebb",
    "persistent_error_hebb",
    "gated_error_hebb_anchor_rollback",
    "full_selective",
    "sgd_oracle",
]

PHASE2_RULES = [
    "frozen",
    "error_hebb",
    "surprise_hebb",
    "persistent_error_hebb",
    "error_hebb_anchor_rollback",
    "error_hebb_consistency_rollback",
    "gated_error_hebb_anchor_rollback",
    "full_selective",
    "full_selective_v2",
    "error_hebb_oracle_rollback",
]
