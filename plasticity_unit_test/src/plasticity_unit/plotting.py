from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _smooth(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1, center=True).mean()


def plot_timeseries(df: pd.DataFrame, outdir: Path, smooth_window: int = 15) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mean = df.groupby(["rule", "t"], sort=False).agg(
        val_loss=("val_loss_before", "mean"),
        core_loss=("val_core_before", "mean"),
        trace_norm=("trace_norm", "mean"),
        p_error=("p_error_to_true_delta", "mean"),
        grad_alignment=("grad_alignment", "mean"),
        gate=("gate", "mean"),
        candidate_delta_val=("candidate_delta_val", "mean"),
    ).reset_index()

    plots = [
        ("val_loss", "Held-out prediction MSE", "heldout_prediction_mse.png"),
        ("candidate_delta_val", "Candidate update Δ held-out loss", "delta_heldout_loss.png"),
        ("grad_alignment", "Cosine with negative sample gradient", "gradient_alignment.png"),
        ("trace_norm", "Trace Frobenius norm", "trace_norm.png"),
        ("p_error", "Distance from true residual ΔM", "p_error_to_true_delta.png"),
        ("gate", "Average gate value", "gate.png"),
    ]

    for col, ylabel, filename in plots:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca()
        for rule, sub in mean.groupby("rule", sort=False):
            y = _smooth(sub[col], smooth_window)
            ax.plot(sub["t"], y, label=rule)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncols=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=180)
        plt.close(fig)


def plot_summary(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("beneficial_update_rate", "Beneficial update rate", "beneficial_update_rate.png"),
        ("mean_grad_alignment", "Mean gradient alignment", "mean_gradient_alignment.png"),
        ("recovery_ratio_vs_oracle", "Recovery ratio vs oracle", "recovery_ratio_vs_oracle.png"),
        ("max_trace_norm", "Max trace norm", "max_trace_norm.png"),
    ]
    for col, ylabel, filename in metrics:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca()
        ax.bar(summary["rule"], summary[col])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=180)
        plt.close(fig)
