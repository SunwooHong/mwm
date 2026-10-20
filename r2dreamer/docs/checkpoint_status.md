# Checkpoint status

Found and loaded:

`r2dreamer/logdir/dreamer_walker_pretrain_5202322/latest.pt` (115 MB, 2026-06-10).

Hydra config in that directory: `dmc_walker_walk`, `rep_loss=dreamer`, `steps: 510000`, `size12M` (`stoch=32`, `discrete=16`, `deter=2048`). Logged train step in `metrics.jsonl` reaches 507136. Last logged `episode/score` is 922.9 at step 496015, episode length 500.

`load_state_dict(..., strict=False)` succeeds on CPU. Parameter counts: RSSM 5.78M, actor 0.79M, value 0.85M, reward 0.72M, cont 0.66M, encoder 0.14M, decoder 0.79M.

Other logdirs exist (`dreamer_walker`, `dreamer_walker_pretrain_5202300`, smoke baselines). This file is the one the sbatch scripts name as the default.

No retraining was started.
