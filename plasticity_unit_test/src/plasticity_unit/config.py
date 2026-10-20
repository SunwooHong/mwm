from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the synthetic plasticity unit test.

    The default config is intentionally small enough to run on CPU in seconds,
    while still exposing the three core questions:
      1. Does a local update reduce held-out prediction loss?
      2. Does it align with the negative prediction-loss gradient?
      3. Does boundedness / gating prevent harmful writes?
    """

    # Randomness / device.
    seed: int = 0
    num_seeds: int = 5
    device: str = "cpu"

    # Synthetic latent dynamics dimensions.
    dz: int = 16
    da: int = 4
    low_rank: int = 2
    transition_noise: float = 0.01
    shift_scale: float = 0.16
    spectral_radius: float = 0.85
    action_scale: float = 0.25
    state_clip: float = 8.0

    # Episode phases: nominal A, shifted B, nominal A, shifted B.
    phase_names: Tuple[str, ...] = ("A", "B", "A", "B")
    phase_lengths: Tuple[int, ...] = (100, 240, 160, 240)

    # Validation/adaptation batches sampled from the current regime.
    val_batch: int = 256
    oracle_batch: int = 64

    # Scenario options: linear_shift | poison | distractor.
    scenario: str = "linear_shift"
    poison_prob: float = 0.08
    poison_scale: float = 3.0
    distractor_scale: float = 2.5
    core_dims: int = 8

    # Plasticity hyperparameters.
    lr_error: float = 0.030
    lr_hebb: float = 0.0015
    lr_random: float = 0.003
    lr_oracle: float = 0.060
    decay: float = 0.001
    oja_alpha: float = 0.10
    max_trace_norm: float = 3.0
    rollback_tolerance: float = 1.0e-6

    # Gate hyperparameters.
    gate_ema_beta: float = 0.990
    gate_var_beta: float = 0.990
    gate_z_threshold: float = 1.5
    gate_temperature: float = 0.5
    persist_beta: float = 0.94
    persist_threshold: float = 0.35
    persist_temperature: float = 0.07

    # Plotting/logging.
    smooth_window: int = 15

    @property
    def dx(self) -> int:
        return self.dz + self.da

    @property
    def total_steps(self) -> int:
        return int(sum(self.phase_lengths))

    def validate(self) -> None:
        if len(self.phase_names) != len(self.phase_lengths):
            raise ValueError("phase_names and phase_lengths must have the same length.")
        if self.scenario not in {"linear_shift", "poison", "distractor"}:
            raise ValueError("scenario must be one of: linear_shift, poison, distractor.")
        if not 0 < self.core_dims <= self.dz:
            raise ValueError("core_dims must be in [1, dz].")
