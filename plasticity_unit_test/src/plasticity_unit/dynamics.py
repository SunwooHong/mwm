from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

from .config import ExperimentConfig


@dataclass
class Transition:
    """One transition from the pre-generated episode."""

    t: int
    phase: str
    x: torch.Tensor          # [dx]
    clean_next: torch.Tensor # [dz]
    obs_next: torch.Tensor   # [dz], possibly corrupted for poison/distractor tests
    corrupted: bool


@dataclass
class Batch:
    """A batch sampled independently from a regime."""

    x: torch.Tensor          # [B, dx]
    clean_next: torch.Tensor # [B, dz]
    obs_next: torch.Tensor   # [B, dz]


def _randn(shape: Tuple[int, ...], generator: torch.Generator, device: str) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, device=device)


def make_stable_matrix(dz: int, generator: torch.Generator, device: str, spectral_radius: float) -> torch.Tensor:
    """Create a stable-ish recurrent matrix A with target spectral radius."""
    raw = _randn((dz, dz), generator, device) / (dz ** 0.5)
    eigvals = torch.linalg.eigvals(raw).abs()
    radius = torch.clamp(eigvals.max().real, min=1.0e-6)
    return raw * (spectral_radius / radius)


def make_system(cfg: ExperimentConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return nominal transition M0 and a low-rank shift DeltaB.

    z_{t+1} = (M0 + Delta) [z_t; a_t] + noise
    """
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 12345)
    A0 = make_stable_matrix(cfg.dz, g, cfg.device, cfg.spectral_radius)
    B0 = cfg.action_scale * _randn((cfg.dz, cfg.da), g, cfg.device) / (cfg.da ** 0.5)
    M0 = torch.cat([A0, B0], dim=1)  # [dz, dz + da]

    # Low-rank systematic residual. This gives us a known target for P_t.
    U = _randn((cfg.dz, cfg.low_rank), g, cfg.device) / (cfg.low_rank ** 0.5)
    V = _randn((cfg.dx, cfg.low_rank), g, cfg.device) / (cfg.dx ** 0.5)
    Delta = U @ V.T
    Delta = cfg.shift_scale * Delta / torch.clamp(torch.linalg.matrix_norm(Delta), min=1.0e-6)

    # For distractor scenario, make the systematic shift nuisance-only.
    if cfg.scenario == "distractor":
        core = cfg.core_dims
        Delta[:core, :] = 0.0
        # Rescale remaining rows to keep the nuisance signal visible.
        nrm = torch.linalg.matrix_norm(Delta)
        if nrm > 1.0e-6:
            Delta = cfg.shift_scale * Delta / nrm

    return M0, Delta


def phase_at(t: int, cfg: ExperimentConfig) -> str:
    start = 0
    for name, length in zip(cfg.phase_names, cfg.phase_lengths):
        if start <= t < start + length:
            return name
        start += length
    raise IndexError(f"Step {t} is outside total_steps={cfg.total_steps}.")


def delta_for_phase(phase: str, DeltaB: torch.Tensor) -> torch.Tensor:
    return DeltaB if phase == "B" else torch.zeros_like(DeltaB)


def predict(M0: torch.Tensor, P: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Predict next latent for single x [dx] or batch x [B, dx]."""
    if x.ndim == 1:
        return (M0 + P) @ x
    if x.ndim == 2:
        return x @ (M0 + P).T
    raise ValueError("x must be rank 1 or rank 2.")


def base_predict(M0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return M0 @ x
    return x @ M0.T


def sample_batch(
    cfg: ExperimentConfig,
    M0: torch.Tensor,
    DeltaB: torch.Tensor,
    phase: str,
    batch_size: int,
    generator: torch.Generator,
    corrupt: bool = False,
) -> Batch:
    """Sample an independent validation/adaptation batch from a phase.

    Validation uses corrupt=False. The oracle can use corrupt=False as an upper bound.
    """
    z = _randn((batch_size, cfg.dz), generator, cfg.device)
    a = _randn((batch_size, cfg.da), generator, cfg.device).clamp(-2.0, 2.0)
    x = torch.cat([z, a], dim=1)
    Delta = delta_for_phase(phase, DeltaB)
    clean = x @ (M0 + Delta).T
    clean = clean + cfg.transition_noise * _randn((batch_size, cfg.dz), generator, cfg.device)
    obs = clean.clone()
    if corrupt:
        if cfg.scenario == "poison":
            mask = torch.rand(batch_size, generator=generator, device=cfg.device) < cfg.poison_prob
            if mask.any():
                obs[mask] += cfg.poison_scale * _randn((int(mask.sum().item()), cfg.dz), generator, cfg.device)
        elif cfg.scenario == "distractor" and phase == "B":
            core = cfg.core_dims
            obs[:, core:] += cfg.distractor_scale * _randn((batch_size, cfg.dz - core), generator, cfg.device)
    return Batch(x=x, clean_next=clean, obs_next=obs)


def generate_episode(cfg: ExperimentConfig, M0: torch.Tensor, DeltaB: torch.Tensor, seed: int) -> List[Transition]:
    """Generate a fixed trajectory independent of the plastic rule.

    This makes rule comparisons fair: every rule sees the same transitions.
    """
    g = torch.Generator(device=cfg.device).manual_seed(seed + 999)
    z = _randn((cfg.dz,), g, cfg.device)
    transitions: List[Transition] = []

    for t in range(cfg.total_steps):
        phase = phase_at(t, cfg)
        a = _randn((cfg.da,), g, cfg.device).clamp(-2.0, 2.0)
        x = torch.cat([z, a], dim=0)
        Delta = delta_for_phase(phase, DeltaB)
        clean = (M0 + Delta) @ x + cfg.transition_noise * _randn((cfg.dz,), g, cfg.device)
        obs = clean.clone()
        corrupted = False

        if cfg.scenario == "poison":
            if torch.rand((), generator=g, device=cfg.device).item() < cfg.poison_prob:
                obs = obs + cfg.poison_scale * _randn((cfg.dz,), g, cfg.device)
                corrupted = True
        elif cfg.scenario == "distractor" and phase == "B":
            core = cfg.core_dims
            obs[core:] = obs[core:] + cfg.distractor_scale * _randn((cfg.dz - core,), g, cfg.device)
            corrupted = True

        transitions.append(Transition(t=t, phase=phase, x=x, clean_next=clean, obs_next=obs, corrupted=corrupted))
        z = clean.clamp(-cfg.state_clip, cfg.state_clip)

    return transitions
