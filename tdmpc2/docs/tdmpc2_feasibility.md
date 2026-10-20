# TD-MPC2 feasibility

Job `1865155` on one L40S. Checkpoint `checkpoints/walker-walk-1.pt` (official 5M single-task `walker-walk-1`). Raw numbers: `logs/mwm_feasibility/feasibility.json`. R2-Dreamer code was not changed.

## Reproduction

Three eval episodes, native MPC, gain 1.0, 500 steps: returns **993.7, 995.5, 995.8**. The published controller is running.

`WorldModel.next` is what MPC calls. A nonzero plastic matrix changes the planned action (L2 0.082 under a fixed torch seed and a small random `P`). The correction is inside the planner, not a side selector.

## Integration

`z_hat` is already SimNorm output. The residual is added in the inverse-SimNorm (log-simplex) space and mapped back with the same SimNorm, so `P=0` reproduces `z_hat`. An earlier run added the residual after SimNorm and re-normalized; that made `P=0` worse than the frozen dynamics and was discarded.

Memory is `512 x 64` floats, 128 KiB. Update is the normalized outer product of `(z_obs - z_pred)` with `phi(z, a)`, `eta=0.05`, decay `0.001`, Frobenius cap 10. The cap did not bind. Prediction error is recorded before the write.

## Prediction

Same actions, 80 steps. Mean latent MSE.

| Seed | Gain 1.0 | Gain 0.7 frozen | Gain 0.7 Hebbian |
| --- | --- | --- | --- |
| 0 | 0.00168 | 0.00252 | 0.00247 |
| 1 | 0.00014 | 0.00540 | 0.00495 |
| 2 | 0.00018 | 0.00466 | 0.00431 |

The gain shift is visible, especially on seeds 1 and 2. Hebbian removes roughly a tenth of that excess, on every seed. A `P` learned on seed 0 at gain 0.7 and applied read-only to seeds 1 and 2 is within 0.00005 of the zero matrix. A memory learned at gain 1.0 matches zero. This is not a gain-specific memory that transfers.

Offline Adam on the same 128 KiB matrix, first 40 shifted transitions, next 40 held out:

| Seed | Train frozen → fit | Held-out frozen → fit |
| --- | --- | --- |
| 0 | 0.00270 → 0.00221 | 0.00235 → 0.00225 |
| 1 | 0.00322 → 0.00243 | 0.00758 → 0.00699 |
| 2 | 0.00208 → 0.00154 | 0.00723 → 0.00740 |

The residual can move in-sample error. Held-out error barely moves, and on seed 2 it gets worse. Hebbian is not failing relative to that ceiling.

Q-value argmax among 4 random actions flips on 3–5 of 12 states. That shows the latent change can reach a value head. It does not show a better action.

## Decision

Do not start the 3-seed closed-loop System 1 pilot, and do not start System 2.

Nominal TD-MPC2 control is reproduced, and the gain shift shows up in next-latent error. The 128 KiB action-conditioned residual does not carry a useful out-of-sample correction. The next change is the adaptation target or the feature, not another learning rate on this rule.

## Command

From `tdmpc2/`:

```bash
sbatch scripts/sbatch_mwm_feasibility.sbatch
```

GPU `l40s:1`, account `aip-rahulgk`, 1 hour, 16G. The script is `tdmpc2/mwm_feasibility.py`.
