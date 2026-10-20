"""TD-MPC2 feasibility: gain shift, Hebbian latent residual, planner sensitivity.

Does not train a world model. Does not run the System 2 control benchmark.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from mwm_plastic import HebbianLatentResidual, inv_simnorm, simnorm
from tdmpc2 import TDMPC2


class Gain:
    def __init__(self, env, gain=1.0):
        self.env = env
        self.gain = float(gain)

    def set_gain(self, gain):
        self.gain = float(gain)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32)
        return self.env.step(action * self.gain)


def load_agent():
    root = Path(__file__).resolve().parent
    ckpt = root.parent / "checkpoints" / "walker-walk-1.pt"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=walker-walk",
                "obs=state",
                "model_size=5",
                "compile=false",
                "enable_wandb=false",
                "save_video=false",
                "hydra/launcher=basic",
                "seed=1",
                f"checkpoint={ckpt}",
            ],
        )
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(str(ckpt))
    agent.model.eval()
    return agent, cfg, env


def episode_return(agent, env, steps_cap=500):
    obs, done, total, t = env.reset(), False, 0.0, 0
    while not done and t < steps_cap:
        action = agent.act(obs, t0=t == 0, eval_mode=True)
        obs, reward, done, _ = env.step(action)
        total += float(reward)
        t += 1
    return total, t


@torch.no_grad()
def collect_actions(agent, env, steps):
    obs = env.reset()
    actions, obs_list = [], [obs]
    task = None
    for t in range(steps):
        o = torch.as_tensor(obs, device=agent.device).unsqueeze(0)
        z = agent.model.encode(o, task)
        _, info = agent.model.pi(z, task)
        action = info["mean"][0].detach().cpu().numpy()
        actions.append(action)
        obs, _, done, _ = env.step(torch.as_tensor(action))
        obs_list.append(obs)
        if done:
            break
    return actions, obs_list


@torch.no_grad()
def replay(agent, env, actions, plastic, write):
    plastic.write = write
    plastic.read = True
    if not write:
        saved = plastic.P.clone()
    obs = env.reset()
    task = None
    errs = []
    rows = []
    for t, action in enumerate(actions):
        o = torch.as_tensor(obs, device=agent.device).unsqueeze(0)
        z = agent.model.encode(o, task)
        a = torch.as_tensor(action, device=agent.device).view(1, -1)
        z_hat = agent.model._dynamics(torch.cat([z, a], -1))
        z_pred = plastic.correct(z, a, z_hat)
        nxt, _, done, _ = env.step(torch.as_tensor(action))
        z_obs = agent.model.encode(torch.as_tensor(nxt, device=agent.device).unsqueeze(0), task)
        err = float((z_pred - z_obs).pow(2).mean())
        errs.append(err)
        rows.append((z.detach().cpu(), a.detach().cpu(), z_hat.detach().cpu(), z_obs.detach().cpu()))
        plastic.update(z, a, z_pred, z_obs)
        obs = nxt
        if done:
            break
    if not write:
        plastic.P = saved
    return errs, rows


def offline_fit(plastic, rows, fit_n=40, steps_opt=60):
    train, test = rows[:fit_n], rows[fit_n:]
    device = plastic.P.device
    P = torch.nn.Parameter(torch.zeros_like(plastic.P))
    opt = torch.optim.Adam([P], lr=0.05)

    def pack(chunk):
        zs, az, hats, obs = zip(*chunk)
        z = torch.cat(zs).to(device)
        a = torch.cat(az).to(device)
        h = torch.cat(hats).to(device)
        o = torch.cat(obs).to(device)
        return z, a, h, o

    z, a, h, o = pack(train)
    for _ in range(steps_opt):
        opt.zero_grad()
        x = plastic.phi(z, a)
        pred = simnorm(inv_simnorm(h, plastic.simnorm_dim) + x @ P.T, plastic.simnorm_dim)
        loss = (pred - o).pow(2).mean() + 1e-4 * P.pow(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            flat = torch.linalg.vector_norm(P).clamp(min=1e-12)
            P.mul_(torch.clamp(plastic.max_norm / flat, max=1.0))
    with torch.no_grad():
        def score(chunk, Pm):
            z, a, h, o = pack(chunk)
            x = plastic.phi(z, a)
            pred = simnorm(inv_simnorm(h, plastic.simnorm_dim) + x @ Pm.T, plastic.simnorm_dim)
            frozen = (h - o).pow(2).mean()
            return float((pred - o).pow(2).mean()), float(frozen)
        tr = score(train, P)
        te = score(test, P) if test else (None, None)
    return {"train_mse": tr[0], "train_frozen": tr[1], "heldout_mse": te[0], "heldout_frozen": te[1], "P": P.detach()}


@torch.no_grad()
def rank_flip(agent, plastic, rows, P, n=12):
    device = agent.device
    P0 = plastic.P
    flips = 0
    for i, (z, _a, _h, _o) in enumerate(rows[:n]):
        z = z.to(device)
        g = torch.Generator(device="cpu")
        g.manual_seed(100 + i)
        noise = torch.randn(4, agent.cfg.action_dim, generator=g).to(device)
        base = torch.zeros(1, agent.cfg.action_dim, device=device)
        cands = (base + 0.3 * noise).clamp(-1, 1)
        scores = []
        for read, mat in ((False, P0), (True, P.to(device))):
            plastic.read = read
            plastic.P = mat
            sc = []
            for j in range(4):
                a = cands[j:j + 1]
                nxt = agent.model.next(z, a, None)
                q = agent.model.Q(nxt, a, None, return_type="avg")
                sc.append(float(q.reshape(-1)[0]))
            scores.append(np.array(sc))
        flips += int(scores[0].argmax() != scores[1].argmax())
    plastic.P = P0
    plastic.read = True
    return {"n": n, "argmax_flips": flips}


@torch.no_grad()
def planner_changes(agent, env, plastic):
    obs = env.reset()
    plastic.read = False
    torch.manual_seed(0)
    a0 = agent.act(obs, t0=True, eval_mode=True).numpy()
    plastic.read = True
    plastic.P = torch.randn_like(plastic.P) * 0.05
    torch.manual_seed(0)
    a1 = agent.act(obs, t0=True, eval_mode=True).numpy()
    plastic.P.zero_()
    return {"action_l2": float(np.linalg.norm(a0 - a1)), "a0": a0.tolist(), "a1": a1.tolist()}


def window(xs, n):
    return float(np.mean(xs[:n])) if xs[:n] else None


def main():
    agent, cfg, _env = load_agent()
    plastic = HebbianLatentResidual(cfg.latent_dim, cfg.action_dim, cfg.simnorm_dim, feature_dim=64, eta=0.05, max_norm=10.0).to(agent.device)
    agent.model.mwm = plastic
    out = Path(__file__).resolve().parents[1] / "logs" / "mwm_feasibility"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": "checkpoints/walker-walk-1.pt",
        "latent_dim": int(cfg.latent_dim),
        "action_dim": int(cfg.action_dim),
        "memory_bytes": plastic.memory_bytes(),
        "device": str(agent.device),
    }
    returns = []
    for ep in range(3):
        cfg.seed = ep + 1
        env = Gain(make_env(cfg), 1.0)
        ret, n = episode_return(agent, env)
        returns.append({"return": ret, "steps": n, "seed": ep + 1})
        print("nominal", ret, n, flush=True)
    report["nominal_return"] = returns
    env = Gain(make_env(cfg), 1.0)
    report["planner_sensitivity"] = planner_changes(agent, env, plastic)
    print("planner", report["planner_sensitivity"]["action_l2"], flush=True)

    seeds = {}
    stored = {}
    for seed in (0, 1, 2):
        cfg.seed = seed
        env1 = Gain(make_env(cfg), 1.0)
        actions, _ = collect_actions(agent, env1, 80)
        env1b = Gain(make_env(cfg), 1.0)
        plastic.P.zero_()
        plastic.proj_hits = 0
        nom_err, _ = replay(agent, env1b, actions, plastic, write=False)
        env7 = Gain(make_env(cfg), 0.7)
        plastic.P.zero_()
        shift_err, rows = replay(agent, env7, actions, plastic, write=True)
        P_online = plastic.P.detach().clone()
        env7b = Gain(make_env(cfg), 0.7)
        plastic.P.zero_()
        frozen_err, _ = replay(agent, env7b, actions, plastic, write=False)
        fit = offline_fit(plastic, rows)
        seeds[str(seed)] = {
            "gain1_frozen_mse": {f"n{k}": window(nom_err, k) for k in (10, 25, 50, 80)},
            "gain07_hebbian_mse": {f"n{k}": window(shift_err, k) for k in (10, 25, 50, 80)},
            "gain07_frozen_mse": {f"n{k}": window(frozen_err, k) for k in (10, 25, 50, 80)},
            "proj_hit_frac": plastic.proj_hits / max(plastic.steps, 1),
            "P_frob": float(torch.linalg.vector_norm(P_online)),
            "offline": {k: v for k, v in fit.items() if k != "P"},
            "rank_online": rank_flip(agent, plastic, rows, P_online),
            "rank_offline": rank_flip(agent, plastic, rows, fit["P"]),
        }
        stored[seed] = {"P": P_online, "rows_shift_actions": actions, "fitP": fit["P"]}
        print(seed, "g1", seeds[str(seed)]["gain1_frozen_mse"]["n80"], "g07", seeds[str(seed)]["gain07_frozen_mse"]["n80"], "heb", seeds[str(seed)]["gain07_hebbian_mse"]["n80"], flush=True)
    # stale: seed0 shift memory on seed1/2 shift, and nominal-trained memory
    plastic.P.zero_()
    cfg.seed = 0
    env_nom = Gain(make_env(cfg), 1.0)
    actions0, _ = collect_actions(agent, env_nom, 80)
    env_nom2 = Gain(make_env(cfg), 1.0)
    plastic.P.zero_()
    replay(agent, env_nom2, actions0, plastic, write=True)
    P_nominal = plastic.P.detach().clone()
    transfers = {}
    for seed in (1, 2):
        cfg.seed = seed
        env_s = Gain(make_env(cfg), 1.0)
        actions, _ = collect_actions(agent, env_s, 80)
        def eval_P(P):
            plastic.P = P.to(agent.device).clone()
            env_e = Gain(make_env(cfg), 0.7)
            err, _ = replay(agent, env_e, actions, plastic, write=False)
            return window(err, 80)
        transfers[str(seed)] = {
            "same_gain_memory": eval_P(stored[0]["P"]),
            "nominal_memory": eval_P(P_nominal),
            "zero": eval_P(torch.zeros_like(P_nominal)),
        }
    report["seeds"] = seeds
    report["transfers_mse80"] = transfers
    (out / "feasibility.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / "feasibility.json", flush=True)


if __name__ == "__main__":
    main()
