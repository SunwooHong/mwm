# MWM

Malleable World Models working tree: R2-Dreamer negative results, TD-MPC2 feasibility, and the action-oracle script.

Heavy artifacts are not in git.

- R2-Dreamer checkpoint: `r2dreamer/logdir/dreamer_walker_pretrain_5202322/latest.pt` on the original machine (115 MB, over GitHub's file limit).
- TD-MPC2 checkpoint: download `dmcontrol/walker-walk-1.pt` from `https://huggingface.co/nicklashansen/tdmpc2` into `tdmpc2/checkpoints/`.
- Virtualenvs are local. The Dreamer runs used `r2dreamer/.venv`.

TD-MPC2 feasibility: `cd tdmpc2 && sbatch scripts/sbatch_mwm_feasibility.sbatch`

Action oracle: `cd tdmpc2 && sbatch scripts/sbatch_mwm_oracle.sbatch`
