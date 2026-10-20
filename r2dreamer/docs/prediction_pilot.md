# Prediction pilot

Fixed frozen-actor trajectories on `walker-walk`, gain `1.0` for 10 steps then `0.7` for 100 steps. Seeds 0, 1, 2. The same observations and actions are scored for every method. KL is categorical posterior versus the prior stored before that observation updates `P`. `eta=0.05`, norm cap 2, no hyperparameter search. Checkpoint `logdir/dreamer_walker_pretrain_5202322/latest.pt`. Raw rows: `logdir/v2_prediction_pilot/summary.json`.

Post-shift mean KL, online minus frozen. Negative means the Hebbian prior is closer to the posterior.

| Memory | Seed | ΔKL first 10 | ΔKL first 100 | Frozen KL first 100 |
| --- | --- | --- | --- | --- |
| 512×64 | 0 | −0.052 | −0.123 | 21.99 |
| 512×64 | 1 | −0.041 | −0.121 | 51.90 |
| 512×64 | 2 | −0.010 | −0.110 | 31.98 |
| 512×2566 | 0 | −0.061 | −0.669 | 21.99 |
| 512×2566 | 1 | −0.059 | −0.504 | 51.90 |
| 512×2566 | 2 | −0.004 | −0.436 | 31.98 |

The sign is consistent: 6/6 online runs reduce KL. The size is not. A drop of about 0.1–0.7 against a baseline of 22–52 is under 3 percent. No-write matches frozen exactly. A `P` learned on seed 0 and applied read-only to seeds 1–2 moves KL by a similar small amount, so this is not evidence of a state-dependent shift model.

Plastic-walk time, not the RSSM step: about 174 ms/step for the 128 KiB matrix and about 375 ms/step for the full matrix, on CPU. The environment rollout itself was several seconds per step on this machine.

H1, meaningful fast predictive adaptation, is not supported at this configuration.
