from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from .config import ExperimentConfig
from .dynamics import (
    Batch,
    base_predict,
    delta_for_phase,
    generate_episode,
    make_system,
    predict,
    sample_batch,
)
from .rules import PlasticRule, RULES, cosine, project_frobenius, should_consistency_rollback


def mse(pred: torch.Tensor, target: torch.Tensor, core_dims: Optional[int] = None) -> torch.Tensor:
    if core_dims is not None:
        pred = pred[..., :core_dims]
        target = target[..., :core_dims]
    return (pred - target).pow(2).mean()


def batch_loss(M0: torch.Tensor, P: torch.Tensor, batch: Batch, core_dims: Optional[int] = None, target: str = "clean") -> float:
    pred = predict(M0, P, batch.x)
    y = batch.clean_next if target == "clean" else batch.obs_next
    return float(mse(pred, y, core_dims=core_dims).item())


def transition_loss(M0: torch.Tensor, P: torch.Tensor, x: torch.Tensor, target: torch.Tensor, core_dims: Optional[int] = None) -> float:
    pred = predict(M0, P, x)
    return float(mse(pred, target, core_dims=core_dims).item())


def current_losses(
    M0: torch.Tensor,
    P: torch.Tensor,
    x: torch.Tensor,
    clean_next: torch.Tensor,
    obs_next: torch.Tensor,
    core_dims: int,
) -> Tuple[float, float, float]:
    pred = predict(M0, P, x)
    return (
        float(mse(pred, clean_next).item()),
        float(mse(pred, obs_next).item()),
        float(mse(pred, clean_next, core_dims=core_dims).item()),
    )


def anchor_delta_loss(rule: PlasticRule, M0: torch.Tensor, P_before: torch.Tensor, P_candidate: torch.Tensor, state) -> Optional[float]:
    before = state.anchor.loss(M0, P_before)
    after = state.anchor.loss(M0, P_candidate)
    if before is None or after is None:
        return None
    return after - before


def should_anchor_rollback(cfg: ExperimentConfig, delta_anchor: Optional[float], anchor_size: int) -> bool:
    if delta_anchor is None:
        return False
    if anchor_size < cfg.anchor_min_size:
        return False
    return bool(delta_anchor > cfg.anchor_loss_tolerance)


def run_one_seed(
    cfg: ExperimentConfig,
    *,
    seed: int,
    rules: Sequence[str] = RULES,
) -> pd.DataFrame:
    M0, DeltaB = make_system(cfg)
    transitions = generate_episode(cfg, M0, DeltaB, seed=seed)

    rows: List[Dict[str, float | int | str | bool]] = []

    # Pre-generate validation/oracle batches once per seed so every rule is
    # evaluated on exactly the same held-out transitions.
    val_rng = torch.Generator(device=cfg.device).manual_seed(seed + 202501)
    oracle_rng = torch.Generator(device=cfg.device).manual_seed(seed + 303501)
    val_batches = [
        sample_batch(cfg, M0, DeltaB, tr.phase, cfg.val_batch, val_rng, corrupt=False)
        for tr in transitions
    ]
    oracle_batches = [
        sample_batch(cfg, M0, DeltaB, tr.phase, cfg.oracle_batch, oracle_rng, corrupt=False)
        for tr in transitions
    ]

    for rule_name in rules:
        rule = PlasticRule(rule_name, cfg, seed=seed)
        state = rule.init_state()

        for idx, tr in enumerate(transitions):
            val_batch = val_batches[idx]
            oracle_batch = oracle_batches[idx]

            P_before = state.P
            val_before = batch_loss(M0, P_before, val_batch)
            val_core_before = batch_loss(M0, P_before, val_batch, core_dims=cfg.core_dims)
            clean_loss, obs_loss, core_loss = current_losses(M0, P_before, tr.x, tr.clean_next, tr.obs_next, cfg.core_dims)

            # Add anchor before the update, using only observed loss and observed next-state.
            state.anchor.maybe_add(tr.x, tr.obs_next, obs_loss, cfg)
            anchor_size_before = len(state.anchor)

            # True per-sample negative gradient direction for observed transition loss.
            # This is a diagnostic; it is not used by the local rule.
            base = base_predict(M0, tr.x)
            err = (tr.obs_next - base) - (P_before @ tr.x)
            neg_grad = torch.outer(err.detach(), tr.x)

            dP, raw_gate, applied_gate = rule.update(
                state,
                M0=M0,
                x=tr.x,
                obs_next=tr.obs_next,
                oracle_batch=oracle_batch,
            )
            state.gate.update_obs_loss_ema(obs_loss, cfg)
            align = cosine(dP, neg_grad)
            candidate = project_frobenius(P_before + dP, cfg.max_trace_norm)
            val_after_candidate = batch_loss(M0, candidate, val_batch)
            val_core_after_candidate = batch_loss(M0, candidate, val_batch, core_dims=cfg.core_dims)
            candidate_delta_val = val_after_candidate - val_before
            candidate_delta_core = val_core_after_candidate - val_core_before

            rollback = False
            rollback_reason = "none"
            delta_anchor = anchor_delta_loss(rule, M0, P_before, candidate, state)

            if PlasticRule.uses_oracle_rollback(rule_name) and candidate_delta_val > cfg.rollback_tolerance:
                rollback = True
                rollback_reason = "oracle_val"
                state.P = P_before
            elif PlasticRule.uses_anchor_rollback(rule_name) and should_anchor_rollback(cfg, delta_anchor, anchor_size_before):
                rollback = True
                rollback_reason = "anchor"
                state.P = P_before
            elif PlasticRule.uses_consistency_rollback(rule_name):
                do_consistency, cons_reason = should_consistency_rollback(
                    cfg,
                    state,
                    P_before=P_before,
                    candidate=candidate,
                    dP=dP,
                    x=tr.x,
                    err=err,
                    applied_gate=applied_gate,
                    obs_loss=obs_loss,
                )
                if do_consistency:
                    rollback = True
                    rollback_reason = f"consistency:{cons_reason}"
                    state.P = P_before
                else:
                    state.P = candidate
            else:
                state.P = candidate

            state.gate.commit_err(err, cfg)
            applied_write_norm = 0.0 if rollback else float(torch.linalg.matrix_norm(state.P - P_before).item())
            val_after = batch_loss(M0, state.P, val_batch)
            val_core_after = batch_loss(M0, state.P, val_batch, core_dims=cfg.core_dims)
            Delta_current = delta_for_phase(tr.phase, DeltaB)
            p_error = float(torch.linalg.matrix_norm(state.P - Delta_current).item())
            trace_norm = float(torch.linalg.matrix_norm(state.P).item())
            false_write_context = bool(tr.corrupted or (cfg.canonical_scenario == "nuisance_distractor" and tr.phase == "B"))
            write_norm = float(torch.linalg.matrix_norm(dP).item())

            rows.append(
                {
                    "seed": seed,
                    "rule": rule_name,
                    "t": tr.t,
                    "phase": tr.phase,
                    "scenario": cfg.canonical_scenario,
                    "corrupted": bool(tr.corrupted),
                    "false_write_context": false_write_context,
                    "raw_gate": float(raw_gate),
                    "gate": float(applied_gate),
                    "candidate_write_norm": write_norm,
                    "write_norm": applied_write_norm,
                    "rollback": bool(rollback),
                    "rollback_reason": rollback_reason,
                    "anchor_size": int(anchor_size_before),
                    "anchor_delta_loss": float(delta_anchor) if delta_anchor is not None else float("nan"),
                    "current_clean_loss": clean_loss,
                    "current_obs_loss": obs_loss,
                    "current_core_loss": core_loss,
                    "val_loss_before": val_before,
                    "val_loss_after": val_after,
                    "candidate_delta_val": candidate_delta_val,
                    "candidate_delta_core": candidate_delta_core,
                    "applied_delta_val": val_after - val_before,
                    "val_core_before": val_core_before,
                    "val_core_after": val_core_after,
                    "applied_delta_core": val_core_after - val_core_before,
                    "grad_alignment": align,
                    "trace_norm": trace_norm,
                    "p_error_to_true_delta": p_error,
                }
            )

    return pd.DataFrame(rows)


def _phase_mean(s: pd.Series, df: pd.DataFrame, phase: str) -> float:
    return float(s[df.loc[s.index, "phase"] == phase].mean())


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Produce rule-level metrics that are easy to scan."""
    grouped = df.groupby("rule", sort=False)
    base = grouped.agg(
        mean_candidate_delta_val=("candidate_delta_val", "mean"),
        beneficial_update_rate=("candidate_delta_val", lambda s: float((s < 0).mean())),
        mean_applied_delta_val=("applied_delta_val", "mean"),
        applied_beneficial_rate=("applied_delta_val", lambda s: float((s < 0).mean())),
        mean_candidate_delta_core=("candidate_delta_core", "mean"),
        mean_applied_delta_core=("applied_delta_core", "mean"),
        mean_grad_alignment=("grad_alignment", "mean"),
        mean_val_loss=("val_loss_before", "mean"),
        auc_shift_val_loss=("val_loss_before", lambda s: _phase_mean(s, df, "B")),
        auc_nominal_val_loss=("val_loss_before", lambda s: _phase_mean(s, df, "A")),
        mean_core_val_loss=("val_core_before", "mean"),
        auc_shift_core_loss=("val_core_before", lambda s: _phase_mean(s, df, "B")),
        max_trace_norm=("trace_norm", "max"),
        mean_trace_norm=("trace_norm", "mean"),
        mean_p_error=("p_error_to_true_delta", "mean"),
        rollback_rate=("rollback", "mean"),
        anchor_rollback_rate=("rollback_reason", lambda s: float((s == "anchor").mean())),
        consistency_rollback_rate=("rollback_reason", lambda s: float(s.astype(str).str.startswith("consistency:").mean())),
        oracle_rollback_rate=("rollback_reason", lambda s: float((s == "oracle_val").mean())),
        mean_write_norm=("write_norm", "mean"),
        mean_candidate_write_norm=("candidate_write_norm", "mean"),
    ).reset_index()

    fw_rows = []
    for rule_name, sub in df.groupby("rule", sort=False):
        ctx = sub[sub["false_write_context"]]
        if len(ctx) == 0:
            rate = float("nan")
            avg_gate = float("nan")
            avg_write = float("nan")
        else:
            rate = float((ctx["gate"] > 0.5).mean())
            avg_gate = float(ctx["gate"].mean())
            avg_write = float(ctx["write_norm"].mean())
            avg_candidate_write = float(ctx["candidate_write_norm"].mean())
        fw_rows.append(
            {
                "rule": rule_name,
                "false_write_rate_gate_gt_0_5": rate,
                "avg_gate_in_false_context": avg_gate,
                "avg_write_norm_false_context": avg_write,
                "avg_candidate_write_norm_false_context": avg_candidate_write if len(ctx) != 0 else float("nan"),
            }
        )
    fw = pd.DataFrame(fw_rows)
    out = base.merge(fw, on="rule", how="left")

    # Recovery ratio relative to frozen and oracle on shifted phases. Higher is better.
    try:
        auc = dict(zip(out["rule"], out["auc_shift_val_loss"]))
        frozen = auc.get("frozen")
        oracle = auc.get("sgd_oracle")
        denom = frozen - oracle if frozen is not None and oracle is not None else None
        ratios = []
        for _, row in out.iterrows():
            if denom is None or abs(denom) < 1.0e-12:
                ratios.append(float("nan"))
            else:
                ratios.append(float((frozen - row["auc_shift_val_loss"]) / denom))
        out["recovery_ratio_vs_oracle"] = ratios
    except Exception:
        out["recovery_ratio_vs_oracle"] = float("nan")

    return out


def run_experiment(cfg: ExperimentConfig, outdir: Path, rules: Sequence[str] = RULES) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.validate()
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for k in range(cfg.num_seeds):
        frames.append(run_one_seed(cfg, seed=cfg.seed + k, rules=rules))
    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize(metrics)

    metrics.to_csv(outdir / "metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    pd.Series(asdict(cfg)).to_json(outdir / "config.json", indent=2)
    return metrics, summary
