"""Oracle action-scale diagnosis and frozen MPC return at gain 0.7.

The true gain is used only inside this diagnostic. No adaptive method sees it.
"""
import json
from pathlib import Path

import numpy as np
import torch

from mwm_feasibility import Gain, episode_return, load_agent
from envs import make_env


@torch.no_grad()
def paired_errors(agent, cfg, seed, steps=80, gain=0.7):
    cfg.seed = seed
    env = Gain(make_env(cfg), 1.0)
    # commanded actions from the nominal policy, then replayed under `gain`
    obs = env.reset()
    actions = []
    for _ in range(steps):
        o = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        z = agent.model.encode(o.unsqueeze(0).to(agent.device), None)
        _, info = agent.model.pi(z, None)
        a = info["mean"][0].detach().cpu()
        actions.append(a)
        obs, _, done, _ = env.step(a)
        if done:
            break
    env_s = Gain(make_env(cfg), gain)
    obs = env_s.reset()
    seen = []
    rows = {k: [] for k in ("frozen", "oracle")}
    for a in actions:
        o = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        z = agent.model.encode(o.unsqueeze(0).to(agent.device), None)
        a_row = a.view(1, -1).to(agent.device)
        # Confirm the wrapper scales before the repeat/physics step.
        scaled = (a_row * gain).detach().cpu()
        pred_f = agent.model.next(z, a_row, None)
        pred_o = agent.model.next(z, a_row * gain, None)
        nxt, _, done, _ = env_s.step(a)
        zo = agent.model.encode((nxt if torch.is_tensor(nxt) else torch.as_tensor(nxt)).unsqueeze(0).to(agent.device), None)
        rows["frozen"].append(float((pred_f - zo).pow(2).mean()))
        rows["oracle"].append(float((pred_o - zo).pow(2).mean()))
        seen.append(float(torch.max(torch.abs(scaled)).item()))
        obs = nxt
        if done:
            break
    # nominal reference: same seed, gain 1, model sees the commanded action
    env_n = Gain(make_env(cfg), 1.0)
    obs = env_n.reset()
    nom = []
    for a in actions:
        o = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        z = agent.model.encode(o.unsqueeze(0).to(agent.device), None)
        a_row = a.view(1, -1).to(agent.device)
        pred = agent.model.next(z, a_row, None)
        nxt, _, done, _ = env_n.step(a)
        zo = agent.model.encode((nxt if torch.is_tensor(nxt) else torch.as_tensor(nxt)).unsqueeze(0).to(agent.device), None)
        nom.append(float((pred - zo).pow(2).mean()))
        obs = nxt
        if done:
            break

    def win(xs):
        return {f"n{n}": float(np.mean(xs[:n])) for n in (10, 25, 50, 80) if xs[:n]}

    frozen, oracle = win(rows["frozen"]), win(rows["oracle"])
    excess = frozen["n80"] - win(nom)["n80"]
    removed = None if excess <= 1e-12 else (frozen["n80"] - oracle["n80"]) / excess
    return {
        "frozen": frozen,
        "oracle": oracle,
        "nominal": win(nom),
        "excess_mse_n80": excess,
        "fraction_excess_removed_n80": removed,
        "max_abs_scaled_action": max(seen) if seen else None,
        "n": len(rows["frozen"]),
    }


def main():
    agent, cfg, _ = load_agent()
    assert getattr(agent.model, "mwm", None) is None
    report = {"seeds": {}, "shift_returns": []}
    for seed in (0, 1, 2):
        report["seeds"][str(seed)] = paired_errors(agent, cfg, seed)
        print(seed, report["seeds"][str(seed)]["frozen"]["n80"], report["seeds"][str(seed)]["oracle"]["n80"], report["seeds"][str(seed)]["fraction_excess_removed_n80"], flush=True)
    for ep, seed in enumerate((1, 2, 3)):
        cfg.seed = seed
        env = Gain(make_env(cfg), 0.7)
        ret, n = episode_return(agent, env)
        report["shift_returns"].append({"seed": seed, "return": ret, "steps": n})
        print("shift return", seed, ret, n, flush=True)
    out = Path(__file__).resolve().parents[1] / "logs" / "action_oracle"
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / "oracle.json", flush=True)


if __name__ == "__main__":
    main()
