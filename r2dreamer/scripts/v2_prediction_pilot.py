"""Paired prequential prior comparison on a fixed walker gain-shift trajectory.

All methods see the same observations and actions. The scored prior is the one
stored before the target observation is used to update P.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from tensordict import TensorDict

from plastic.causal_s1 import CausalPlasticPrior, CausalTrace
from plastic_inference import load_agent


def batch_obs(obs, device, first: bool):
    td = {k: torch.as_tensor(v, device=device).unsqueeze(0).float() for k, v in obs.items()}
    td["is_first"] = torch.tensor([first], device=device)
    return td


@torch.no_grad()
def collect_cache(agent, env, seed: int, steps_a: int, steps_b: int):
    """Roll the frozen actor and cache the prior that predicts the next observation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    state = agent.get_initial_state(1)
    pending = None
    cached = []
    plan = [("A", steps_a, 1.0), ("B", steps_b, 0.7)]
    for phase, n, gain in plan:
        env.set_gain(gain)
        for t in range(n):
            first = pending is None
            td = batch_obs(obs, agent.device, first)
            embed = agent._frozen_encoder(agent.preprocess(td))
            is_first = td["is_first"].reshape(1)
            prev_stoch, prev_deter, prev_action = state["stoch"], state["deter"], state["prev_action"]
            _, deter, post_shaped = agent._frozen_rssm._obs_step_core(
                prev_stoch, prev_deter, prev_action, embed, is_first
            )
            post = agent._frozen_rssm.get_dist(post_shaped).mode
            post_logit = post_shaped.reshape(1, -1).cpu()
            if pending is not None:
                cached.append({**pending, "post_logit": post_logit})
            action = agent.actor(agent._frozen_rssm.get_feat(post, deter)).mode
            next_d = agent._frozen_rssm._deter_net(post, deter, action)
            base = agent._frozen_rssm.prior_logit_from_deter(next_d).reshape(1, -1).cpu()
            raw = torch.cat(
                [post.reshape(1, -1), deter, action / torch.clamp(action.abs(), min=1.0)],
                dim=-1,
            ).cpu()
            pending = {"base": base, "raw": raw, "phase": phase, "t": t}
            state = TensorDict({"stoch": post, "deter": deter, "prev_action": action}, batch_size=(1,))
            obs, _, done, _ = env.step(action.detach().cpu().numpy()[0])
            if done:
                obs = env.reset()
                state = agent.get_initial_state(1)
                pending = None
            if t % 25 == 0:
                print(f"seed {seed} {phase} t={t}", flush=True)
    return cached


def _unused_collect(agent, env, seed: int, steps_a: int, steps_b: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    state = agent.get_initial_state(1)
    rows = []
    phase_of = [("A", steps_a, 1.0), ("B", steps_b, 0.7)]
    for phase, n, gain in phase_of:
        env.set_gain(gain)
        for t in range(n):
            rows.append({"obs": {k: np.array(v) for k, v in obs.items()}, "phase": phase, "t": t, "gain": gain})
            action, state = agent.act(batch_obs(obs, agent.device, first=(t == 0 and phase == "A" and not rows[:-1])), state, eval=True)
            nxt, reward, done, _ = env.step(action.detach().cpu().numpy()[0])
            rows[-1]["action"] = action.detach().cpu().numpy()[0]
            rows[-1]["reward"] = float(reward)
            rows[-1]["done"] = bool(done)
            if done:
                obs = env.reset()
                state = agent.get_initial_state(1)
            else:
                obs = nxt
    return rows


@torch.no_grad()
def cache_transitions(agent, rows):
    """One RSSM pass. Item i predicts observation i from the previous action."""
    b = 1
    state = agent.get_initial_state(1)
    cached = []
    pending = None
    for i, row in enumerate(rows):
        first = i == 0 or rows[i - 1]["done"]
        obs = batch_obs(row["obs"], agent.device, first)
        embed = agent._frozen_encoder(agent.preprocess(obs))
        is_first = obs["is_first"].reshape(b)
        prev_stoch, prev_deter, prev_action = state["stoch"], state["deter"], state["prev_action"]
        _prev, deter, post_shaped = agent._frozen_rssm._obs_step_core(
            prev_stoch, prev_deter, prev_action, embed, is_first
        )
        post = agent._frozen_rssm.get_dist(post_shaped).mode
        post_logit = post_shaped.reshape(b, -1)
        if pending is not None and not first:
            cached.append({**pending, "post_logit": post_logit.cpu(), "phase": row["phase"], "t": row["t"]})
        action = torch.as_tensor(row["action"], device=agent.device).view(1, -1)
        next_d = agent._frozen_rssm._deter_net(post, deter, action)
        base = agent._frozen_rssm.prior_logit_from_deter(next_d).reshape(b, -1)
        raw = torch.cat(
            [post.reshape(b, -1), deter, action / torch.clamp(action.abs(), min=1.0)],
            dim=-1,
        )
        pending = {"base": base.cpu(), "raw": raw.cpu()}
        state = TensorDict({"stoch": post, "deter": deter, "prev_action": action}, batch_size=(1,))
    return cached


def walk_plastic(plastic: CausalPlasticPrior, cached, *, write: bool, init: torch.Tensor | None):
    trace = CausalTrace(P=plastic.initial(1, "cpu").P if init is None else init.clone())
    kls, mses = [], []
    t0 = time.perf_counter()
    for row in cached:
        x = plastic.featurize(row["raw"])
        pred = plastic.read_logits(trace, x, row["base"])
        q = CausalPlasticPrior.probs(row["post_logit"], plastic_n_classes(plastic))
        p = CausalPlasticPrior.probs(pred, plastic_n_classes(plastic))
        kl = float((q * (q.clamp(min=1e-8).log() - p.clamp(min=1e-8).log())).sum(-1))
        mse = float((row["post_logit"] - pred).pow(2).mean())
        kls.append({"kl": kl, "mse": mse, "phase": row["phase"], "t": row["t"], "norm": float(torch.linalg.vector_norm(trace.P))})
        if write:
            trace = plastic.write(trace, x, q, p)
    elapsed = time.perf_counter() - t0
    return kls, trace.P, elapsed / max(len(cached), 1)


def plastic_n_classes(plastic: CausalPlasticPrior) -> int:
    return int(getattr(plastic, "n_classes"))


def windows(rows, key):
    b = [r for r in rows if r["phase"] == "B"]
    out = {}
    for n in (10, 25, 50, 100):
        chunk = b[:n]
        out[f"B{n}_{key}"] = float(np.mean([r[key] for r in chunk])) if chunk else None
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    log = root / "logdir/dreamer_walker_pretrain_5202322"
    out = root / "logdir/v2_prediction_pilot"
    out.mkdir(parents=True, exist_ok=True)
    agent, _, env = load_agent(log, log / "latest.pt", "cpu", seed=0)
    k = agent.rssm._stoch * agent.rssm._discrete
    raw = k + agent.rssm._deter + agent.act_dim
    specs = {
        "proj64": CausalPlasticPrior(out_dim=k, in_dim=raw, eta=0.05, feature_dim=64, proj_seed=0),
        "full": CausalPlasticPrior(out_dim=k, in_dim=raw, eta=0.05, feature_dim=None, proj_seed=0),
    }
    for p in specs.values():
        p.n_classes = agent.rssm._discrete
    seeds = [0, 1, 2]
    caches = {}
    for seed in seeds:
        caches[seed] = collect_cache(agent, env, seed, 10, 100)
        print(f"cached seed {seed} transitions {len(caches[seed])}", flush=True)
    summary = []
    learned = {}
    for name, plastic in specs.items():
        for seed, cached in caches.items():
            for mode, write, init in (
                ("frozen", False, None),
                ("online", True, None),
                ("nowrite", False, None),
            ):
                stats, P, dt = walk_plastic(plastic, cached, write=write, init=init)
                rec = {"memory": name, "seed": seed, "mode": mode, "ms_per_step": 1000 * dt, "params": plastic.parameter_count}
                rec.update(windows(stats, "kl"))
                rec.update(windows(stats, "mse"))
                summary.append(rec)
                print(name, seed, mode, {k: round(rec[k], 4) for k in rec if k.startswith("B10")}, flush=True)
                if mode == "online":
                    learned[(name, seed)] = P
        for seed in (1, 2):
            stats, _, dt = walk_plastic(plastic, caches[seed], write=False, init=learned[(name, 0)])
            rec = {"memory": name, "seed": seed, "mode": "stale_from_seed0", "ms_per_step": 1000 * dt, "params": plastic.parameter_count}
            rec.update(windows(stats, "kl"))
            rec.update(windows(stats, "mse"))
            summary.append(rec)
            print(name, seed, "stale", {k: round(rec[k], 4) for k in rec if k.startswith("B10")}, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
