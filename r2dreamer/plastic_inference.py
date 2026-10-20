#!/usr/bin/env python3
"""Phase 3: inference-only plastic prior-logit diagnostic for R2-Dreamer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import rssm
import tools
from dreamer import Dreamer
from envs import make_env
from envs import wrappers
from plastic import LogitPlasticResidual, PlasticTrace


@dataclass
class PhaseSpec:
    phase_idx: int
    name: str
    action_gain: float
    steps: int


def parse_phases(text: str) -> list[PhaseSpec]:
    phases = []
    for idx, chunk in enumerate(text.split(",")):
        name, gain, steps = chunk.split(":")
        phases.append(PhaseSpec(phase_idx=idx, name=name, action_gain=float(gain), steps=int(steps)))
    return phases


def load_agent(
    logdir: Path,
    checkpoint: Path,
    device: str,
    *,
    seed: int | None = None,
) -> tuple[Dreamer, object, object]:
    cfg = OmegaConf.load(logdir / ".hydra/config.yaml")
    cfg.device = device
    cfg.buffer.storage_device = device
    cfg.model.compile = False
    cfg.env.env_num = 1
    cfg.env.eval_episode_num = 0
    if seed is not None:
        cfg.seed = int(seed)

    env = make_env(cfg.env, 0)
    env = wrappers.ActionGain(env, gain=1.0)
    agent = Dreamer(cfg.model, env.observation_space, env.action_space).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    agent.eval()
    for param in agent.parameters():
        param.requires_grad = False
    agent.clone_and_freeze()
    return agent, cfg, env


def flatten_logit(logit: torch.Tensor) -> torch.Tensor:
    return logit.reshape(logit.shape[0], -1)


def make_plastic(
    rssm_mod: rssm.RSSM,
    act_dim: int,
    *,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
) -> LogitPlasticResidual:
    return LogitPlasticResidual(
        flat_stoch=rssm_mod.flat_stoch,
        deter=rssm_mod._deter,
        act_dim=act_dim,
        out_dim=rssm_mod.flat_stoch,
        lr=plastic_lr,
        max_norm=max_trace_norm,
        persist_threshold=persist_threshold,
        gate_mode=gate_mode,
    )


@torch.no_grad()
def plastic_step_metrics(
    rssm_mod: rssm.RSSM,
    plastic: LogitPlasticResidual | None,
    trace: PlasticTrace | None,
    *,
    prev_stoch: torch.Tensor,
    prev_deter: torch.Tensor,
    prev_action: torch.Tensor,
    embed: torch.Tensor,
    is_first: torch.Tensor,
    use_plastic: bool,
) -> tuple[torch.Tensor, torch.Tensor, PlasticTrace | None, dict]:
    stoch, deter, post_logit = rssm_mod._obs_step_core(prev_stoch, prev_deter, prev_action, embed, is_first)
    post_flat = flatten_logit(post_logit)
    prior_logit = rssm_mod.prior_logit_from_deter(deter)
    prior_base = flatten_logit(prior_logit)

    gate_mean = 0.0
    trace_norm = 0.0
    if use_plastic and plastic is not None and trace is not None:
        x = plastic.make_input(stoch, prev_deter, prev_action)
        prior_flat = prior_base + plastic.read(trace, x)
        trace, gate, _decay_eff = plastic.write(trace, x, post_flat.detach() - prior_flat.detach())
        gate_mean = float(gate.mean().item())
        trace_norm = float(torch.linalg.vector_norm(trace.P.reshape(trace.P.shape[0], -1)).item())
        prior_logit = prior_flat.reshape_as(post_logit)
    else:
        prior_flat = prior_base

    prior_mse = float((post_flat.detach() - prior_flat).pow(2).mean().item())
    dyn_kl = float(rssm_mod.kl_loss(post_logit, prior_logit.detach(), free=1.0)[1].mean().item())
    stoch = rssm_mod.get_dist(post_logit).rsample()
    return stoch, deter, trace, {
        "prior_mse": prior_mse,
        "dyn_kl": dyn_kl,
        "gate": gate_mean,
        "trace_norm": trace_norm,
    }


@torch.no_grad()
def run_diagnostic(
    agent: Dreamer,
    env,
    phases: list[PhaseSpec],
    *,
    device: str,
    use_plastic: bool,
    plastic_lr: float,
    gate_mode: str,
    persist_threshold: float,
    max_trace_norm: float,
    run_label: str,
    seed: int,
) -> pd.DataFrame:
    rssm_mod = agent._frozen_rssm
    plastic = None
    trace = None
    if use_plastic:
        plastic = make_plastic(
            rssm_mod,
            agent.act_dim,
            plastic_lr=plastic_lr,
            gate_mode=gate_mode,
            persist_threshold=persist_threshold,
            max_trace_norm=max_trace_norm,
        )
        trace = plastic.initial(1, device)

    state = agent.get_initial_state(1)
    obs = env.reset()
    rows = []
    global_t = 0

    for phase in phases:
        if hasattr(env, "set_gain"):
            env.set_gain(phase.action_gain)
        for local_t in range(phase.steps):
            obs_td = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
            if "is_first" not in obs_td:
                obs_td["is_first"] = torch.zeros(1, dtype=torch.bool, device=device)
            for k in obs_td:
                if obs_td[k].dtype != torch.bool:
                    obs_td[k] = obs_td[k].float()

            embed = agent._frozen_encoder(agent.preprocess(obs_td))
            prev_stoch, prev_deter, prev_action = state["stoch"], state["deter"], state["prev_action"]
            is_first = obs_td["is_first"].reshape(-1)

            if use_plastic and trace is not None:
                trace = plastic.reset_where(trace, is_first)

            stoch, deter, trace, step_metrics = plastic_step_metrics(
                rssm_mod,
                plastic if use_plastic else None,
                trace if use_plastic else None,
                prev_stoch=prev_stoch,
                prev_deter=prev_deter,
                prev_action=prev_action,
                embed=embed,
                is_first=is_first,
                use_plastic=use_plastic,
            )
            action = agent._frozen_actor(rssm_mod.get_feat(stoch, deter)).mode
            state = TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(1,))

            obs, reward, done, _ = env.step(tools.to_np(action)[0])
            if done:
                obs = env.reset()
                state = agent.get_initial_state(1)

            rows.append(
                {
                    "t": global_t,
                    "phase_idx": phase.phase_idx,
                    "phase": phase.name,
                    "action_gain": phase.action_gain,
                    "local_t": local_t,
                    "reward": float(reward),
                    "use_plastic": use_plastic,
                    "run_label": run_label,
                    "seed": seed,
                    "gate_mode": gate_mode if use_plastic else "frozen",
                    "plastic_lr": plastic_lr if use_plastic else 0.0,
                    **step_metrics,
                }
            )
            global_t += 1

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_label", "use_plastic", "gate_mode", "plastic_lr", "phase_idx", "phase", "action_gain"]
    out = (
        df.groupby(keys, as_index=False)
        .agg(
            prior_mse_mean=("prior_mse", "mean"),
            prior_mse_first20=("prior_mse", lambda s: float(s.head(20).mean())),
            prior_mse_last20=("prior_mse", lambda s: float(s.tail(20).mean())),
            prior_mse_last=("prior_mse", "last"),
            dyn_kl_mean=("dyn_kl", "mean"),
            dyn_kl_last20=("dyn_kl", lambda s: float(s.tail(20).mean())),
            gate_mean=("gate", "mean"),
            trace_norm_max=("trace_norm", "max"),
            reward_mean=("reward", "mean"),
        )
        .sort_values(keys)
    )
    # recovery: frozen last20 - plastic last20 on shift phase (positive = plastic better)
    rows = []
    plastic_labels = set(out.loc[out["use_plastic"], "run_label"])
    for label in plastic_labels:
        frozen_label = f"{label}_with_frozen"
        plastic_b = out[(out["run_label"] == label) & (out["use_plastic"]) & (out["phase"].str.startswith("B"))]
        frozen_b = out[(out["run_label"] == frozen_label) & (~out["use_plastic"]) & (out["phase"].str.startswith("B"))]
        if len(frozen_b) and len(plastic_b):
            fb = frozen_b.iloc[0]
            pb = plastic_b.iloc[0]
            rows.append(
                {
                    "run_label": label,
                    "B_prior_mse_delta_last20": float(fb["prior_mse_last20"] - pb["prior_mse_last20"]),
                    "B_dyn_kl_delta_last20": float(fb["dyn_kl_last20"] - pb["dyn_kl_last20"]),
                }
            )
    if rows:
        delta = pd.DataFrame(rows)
        out = out.merge(delta, on="run_label", how="left")
        frozen_delta = delta.copy()
        frozen_delta["run_label"] = frozen_delta["run_label"] + "_with_frozen"
        out = out.merge(
            frozen_delta.rename(
                columns={
                    "B_prior_mse_delta_last20": "B_prior_mse_delta_last20_frozen_row",
                    "B_dyn_kl_delta_last20": "B_dyn_kl_delta_last20_frozen_row",
                }
            ),
            on="run_label",
            how="left",
        )
    return out


def plot_curves(df: pd.DataFrame, outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for label in df["run_label"].unique():
        sub = df[df["run_label"] == label]
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for use_plastic, g in sub.groupby("use_plastic"):
            name = g["gate_mode"].iloc[0] if use_plastic else "frozen"
            lr = g["plastic_lr"].iloc[0] if use_plastic else 0.0
            tag = f"plastic({name},lr={lr})" if use_plastic else "frozen"
            axes[0].plot(g["t"], g["prior_mse"], label=tag, alpha=0.85)
            axes[1].plot(g["t"], g["dyn_kl"], label=tag, alpha=0.85)
        for ax in axes:
            for _, ph in sub.drop_duplicates("phase_idx").iterrows():
                if ph["local_t"] == 0:
                    idx = sub[sub["phase_idx"] == ph["phase_idx"]].index[0]
                    t0 = sub.loc[idx, "t"]
                    ax.axvline(t0, color="gray", linestyle="--", alpha=0.35)
                    ax.text(t0, ax.get_ylim()[1] * 0.95, ph["phase"], fontsize=8, rotation=90, va="top")
        axes[0].set_ylabel("prior_mse")
        axes[1].set_ylabel("dyn_kl")
        axes[1].set_xlabel("step")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.25)
        axes[1].grid(True, alpha=0.25)
        fig.suptitle(f"Phase 3 curves: {label}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"curves_{label}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--phases", type=str, default="A1:1.0:200,B:0.7:1000,A2:1.0:300")
    p.add_argument("--plastic-lr", type=float, default=0.05)
    p.add_argument("--gate-mode", type=str, default="selective", choices=["selective", "surprise_only", "always_on"])
    p.add_argument("--persist-threshold", type=float, default=0.1)
    p.add_argument("--max-trace-norm", type=float, default=2.0)
    p.add_argument("--compare-frozen", action="store_true")
    p.add_argument("--run-label", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    checkpoint = args.checkpoint or (args.logdir / "latest.pt")
    outdir = args.outdir or (args.logdir / "plastic_diagnostic_v2")
    outdir.mkdir(parents=True, exist_ok=True)
    phases = parse_phases(args.phases)
    run_label = args.run_label or f"{args.gate_mode}_lr{args.plastic_lr}"

    frames = []
    agent, _, env = load_agent(args.logdir, checkpoint, args.device, seed=args.seed)
    frames.append(
        run_diagnostic(
            agent,
            env,
            phases,
            device=args.device,
            use_plastic=True,
            plastic_lr=args.plastic_lr,
            gate_mode=args.gate_mode,
            persist_threshold=args.persist_threshold,
            max_trace_norm=args.max_trace_norm,
            run_label=run_label,
            seed=args.seed,
        )
    )
    if args.compare_frozen:
        agent2, _, env2 = load_agent(args.logdir, checkpoint, args.device, seed=args.seed)
        frames.append(
            run_diagnostic(
                agent2,
                env2,
                phases,
                device=args.device,
                use_plastic=False,
                plastic_lr=0.0,
                gate_mode="frozen",
                persist_threshold=0.0,
                max_trace_norm=0.0,
                run_label=f"{run_label}_with_frozen",
                seed=args.seed,
            )
        )

    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize(metrics)
    metrics.to_csv(outdir / "step_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    plot_curves(metrics, outdir)
    meta = {**vars(args), "phases": [asdict(p) for p in phases], "run_label": run_label}
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\n=== {run_label} ===")
    cols = [
        "run_label",
        "use_plastic",
        "gate_mode",
        "plastic_lr",
        "phase",
        "prior_mse_first20",
        "prior_mse_last20",
        "dyn_kl_last20",
        "gate_mean",
        "trace_norm_max",
        "B_prior_mse_delta_last20",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
