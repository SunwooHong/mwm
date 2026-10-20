from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENARIOS = {
    "linear_shift",
    "low_rank_shift",
    "action_gain_shift",
    "back_switch",
    "poison",
    "poison_outlier",
    "distractor",
    "nuisance_distractor",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the synthetic plasticity mechanism tests.

    Phase 0/1 purpose:
      1. Verify that signed local prediction-error plasticity moves the residual
         trace in a loss-reducing direction.
      2. Verify that selective control reduces false writes under outliers or
         nuisance-only shifts.
      3. Keep the test independent of Dreamer/RL so that mechanism failures are
         diagnosable before full-agent experiments.
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
    action_gain_delta: float = -0.35
    state_clip: float = 8.0

    # Episode phases: nominal A, shifted B, nominal A, shifted B.
    phase_names: Tuple[str, ...] = ("A", "B", "A", "B")
    phase_lengths: Tuple[int, ...] = (100, 240, 160, 240)

    # Validation/adaptation batches sampled from the current regime.
    val_batch: int = 256
    oracle_batch: int = 64

    # Scenario options.
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
    error_clip_norm: float = 0.25
    rollback_tolerance: float = 1.0e-6

    # Gate hyperparameters.
    gate_ema_beta: float = 0.990
    gate_var_beta: float = 0.990
    gate_z_threshold: float = 0.75
    gate_temperature: float = 0.5
    persist_beta: float = 0.94
    persist_threshold: float = 0.12
    persist_temperature: float = 0.05

    # Non-oracle anchor rollback.  Anchors are low-observed-loss transitions
    # kept in a small FIFO memory; no clean labels or held-out regime labels are
    # used by this controller.
    anchor_size: int = 64
    anchor_add_loss_threshold: float = 0.015
    anchor_loss_tolerance: float = 0.001
    anchor_min_size: int = 8

    # Phase 2: signed-error direction persistence (kappa_t).
    error_min_norm: float = 0.02
    dir_persist_threshold: float = 0.25
    dir_persist_temperature: float = 0.08

    # Phase 2: non-oracle consistency rollback proxies (no held-out val batch).
    consistency_min_gate: float = 0.20
    consistency_min_write_norm: float = 0.04
    consistency_max_trace_step: float = 0.20
    consistency_max_residual_jump: float = 0.35
    consistency_min_direction_cos: float = -0.15
    consistency_surprise_mult: float = 2.0
    consistency_loss_spike: float = 0.35
    consistency_ema_beta: float = 0.985
    consistency_warmup_steps: int = 20

    # Plotting/logging.
    smooth_window: int = 15

    @property
    def dx(self) -> int:
        return self.dz + self.da

    @property
    def total_steps(self) -> int:
        return int(sum(self.phase_lengths))

    @property
    def canonical_scenario(self) -> str:
        if self.scenario == "poison":
            return "poison_outlier"
        if self.scenario == "distractor":
            return "nuisance_distractor"
        return self.scenario

    def validate(self) -> None:
        if len(self.phase_names) != len(self.phase_lengths):
            raise ValueError("phase_names and phase_lengths must have the same length.")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of: {sorted(SCENARIOS)}")
        if not 0 < self.core_dims <= self.dz:
            raise ValueError("core_dims must be in [1, dz].")
        if self.anchor_size <= 0:
            raise ValueError("anchor_size must be positive.")
