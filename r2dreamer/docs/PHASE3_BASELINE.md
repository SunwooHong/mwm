# Phase 3 baseline: R2-Dreamer environment setup

Before integrating plastic residual (prior-logit patch), verify the upstream baseline runs.

## 1. Create uv virtualenv

```bash
cd /home/hongsunw/projects/aip-rahulgk/hongsunw/mwm/r2dreamer
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dmc]"
```

For Meta-World later: `uv pip install -e ".[dmc,metaworld]"`

## 2. Headless MuJoCo (login / batch nodes)

```bash
export MUJOCO_GL=egl          # GPU nodes (recommended)
# export MUJOCO_GL=osmesa     # CPU-only fallback
```

## 3. Smoke test (minimal steps)

```bash
./scripts/smoke_dmc_proprio.sh logdir/smoke_dmc_proprio
```

Defaults: `dmc_proprio` / `walker_walk`, `model.rep_loss=dreamer`, 800 env steps, 2 parallel envs.

GPU smoke:

```bash
DEVICE=cuda:0 MUJOCO_GL=egl ./scripts/smoke_dmc_proprio.sh logdir/smoke_gpu
```

## 4. Full baseline (after smoke passes)

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python train.py \
  logdir=./logdir/dreamer_walker \
  env=dmc_proprio \
  env.task=dmc_walker_walk \
  model.rep_loss=dreamer \
  device=cuda:0 \
  buffer.storage_device=cuda:0
```

Use `model.rep_loss=dreamer` (not `r2dreamer`) so plastic residual is the only contribution in Phase 3+.

Or submit sbatch:

```bash
mkdir -p logs
sbatch scripts/sbatch_dreamer_walker_pretrain.sbatch
```

Checkpoints are saved to `logdir/.../latest.pt` after each eval (and at end).

## 5. Phase 3 inference diagnostic (no training)

Loads frozen checkpoint, runs policy inference only, optional test-time plastic on prior logits:

```bash
./scripts/run_plastic_diagnostic.sh logdir/smoke_gpu_baseline
```

Compares frozen prior vs plastic under action-gain shift (`A:1.0 → B:0.7 → A:1.0`).

Outputs: `plastic_diagnostic/summary.csv` with `prior_mse`, `dyn_kl`, `trace_norm`.

## 6. Plastic module

Implementation: `plastic/logit_plastic_trace.py`, entrypoint `plastic_inference.py`.
