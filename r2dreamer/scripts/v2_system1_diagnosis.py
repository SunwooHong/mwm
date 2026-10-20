"""Shift-specific KL, update magnitude, local rules, and an offline residual ceiling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from plastic.causal_s1 import CausalPlasticPrior, CausalTrace
from plastic_inference import load_agent

UNIMIX = 0.01
NCLASS = 16
ETA = 0.05
DECAY = 0.001


def stub_render(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    size = tuple(base._size) + (3,)
    base.render = lambda *_a, **_k: np.zeros(size, np.uint8)


def batch_obs(obs, device, first: bool):
    td = {k: torch.as_tensor(np.asarray(v), device=device).unsqueeze(0).float() for k, v in obs.items() if k != "image"}
    td["image"] = torch.zeros(1, 64, 64, 3, device=device)
    td["is_first"] = torch.tensor([first], device=device)
    return td


def mixed_probs(logit, n_classes=NCLASS, unimix=UNIMIX):
    x = logit.reshape(logit.shape[0], -1, n_classes).float()
    p = torch.softmax(x, dim=-1)
    return p * (1.0 - unimix) + unimix / n_classes


def kl_rows(post, prior):
    q = mixed_probs(post)
    p = mixed_probs(prior)
    per = (q * (q.clamp(min=1e-8).log() - p.clamp(min=1e-8).log())).sum(-1)
    return per.sum(-1), per.mean(0)


@torch.no_grad()
def rollout(agent, env, seed, gain, steps, actions=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.set_gain(gain)
    obs = env.reset()
    state = agent.get_initial_state(1)
    pending = None
    cached, taken = [], []
    dev = agent.device
    for t in range(steps):
        first = pending is None
        td = batch_obs(obs, dev, first)
        embed = agent._frozen_encoder(agent.preprocess(td))
        is_first = td["is_first"].reshape(1)
        prev_s, prev_d, prev_a = state["stoch"], state["deter"], state["prev_action"]
        _, deter, post_shaped = agent._frozen_rssm._obs_step_core(prev_s, prev_d, prev_a, embed, is_first)
        post = agent._frozen_rssm.get_dist(post_shaped).mode
        post_logit = post_shaped.reshape(1, -1)
        if pending is not None:
            cached.append({**pending, "post": post_logit.detach().cpu(), "t": pending["t"]})
        if actions is None:
            action = agent.actor(agent._frozen_rssm.get_feat(post, deter)).mode
        else:
            action = torch.as_tensor(actions[t], device=dev).view(1, -1)
        taken.append(action.detach().cpu().numpy()[0])
        next_d = agent._frozen_rssm._deter_net(post, deter, action)
        base = agent._frozen_rssm.prior_logit_from_deter(next_d).reshape(1, -1)
        raw = torch.cat([post.reshape(1, -1), deter, action / torch.clamp(action.abs(), min=1.0)], -1)
        pending = {"base": base.detach().cpu(), "raw": raw.detach().cpu(), "action": action.detach().cpu(), "deter": deter.detach().cpu(), "post_state": post.detach().cpu(), "t": t}
        state = TensorDict({"stoch": post, "deter": deter, "prev_action": action}, batch_size=(1,))
        obs, _, done, _ = env.step(taken[-1])
        if done:
            break
        if t % 25 == 0:
            print(f"  seed {seed} gain {gain} t {t}", flush=True)
    return cached, np.stack(taken)


def apply_rule(plastic, cached, rule, init=None):
    trace = CausalTrace(P=plastic.initial(1, "cpu").P if init is None else init.clone())
    rows, hits = [], 0
    feat_n, err_n, upd_n, corr_n = [], [], [], []
    for row in cached:
        x = plastic.featurize(row["raw"])
        pred = plastic.read_logits(trace, x, row["base"])
        q = mixed_probs(row["post"]).reshape(1, -1)
        p = mixed_probs(pred).reshape(1, -1)
        total, _ = kl_rows(row["post"], pred)
        if rule == "prob":
            err = q - p
        elif rule == "logit":
            err = row["post"] - pred
            nrm = torch.linalg.vector_norm(err, dim=-1, keepdim=True).clamp(min=1e-12)
            if float(nrm) > 0.5:
                hits += 1
            err = err * torch.clamp(0.5 / nrm, max=1.0)
        else:
            err = torch.zeros_like(q)
        feat_n.append(float(torch.linalg.vector_norm(x)))
        err_n.append(float(torch.linalg.vector_norm(err)))
        scale = 1.0 / (x.pow(2).sum(-1, keepdim=True) + plastic.eps)
        delta = ETA * torch.einsum("bo,bi->boi", err, x * scale)
        upd_n.append(float(torch.linalg.vector_norm(delta)))
        corr_n.append(float(torch.linalg.vector_norm(pred - row["base"])))
        if rule != "frozen":
            P = (1.0 - DECAY) * trace.P + delta
            flat = torch.linalg.vector_norm(P.reshape(1, -1)).clamp(min=1e-12)
            if float(flat) > plastic.max_norm:
                hits += 1
            P = P * torch.clamp(plastic.max_norm / flat, max=1.0)
            trace = CausalTrace(P=P)
        rows.append({"t": row["t"], "kl": float(total), "anorm": float(torch.linalg.vector_norm(row["action"]))})
    s = trace.P[0].float()
    sv = torch.linalg.svdvals(s)
    rank = int((sv > 1e-3 * sv.max().clamp(min=1e-12)).sum())
    audit = {
        "feat_norm_mean": float(np.mean(feat_n)),
        "err_norm_mean": float(np.mean(err_n)),
        "update_norm_mean": float(np.mean(upd_n)),
        "correction_mean": float(np.mean(corr_n)),
        "clip_or_proj_frac": hits / max(len(cached), 1),
        "P_frob": float(torch.linalg.vector_norm(s)),
        "effective_rank": rank,
    }
    return rows, trace.P, audit


def windows(rows):
    return {f"n{n}": float(np.mean([r["kl"] for r in rows[:n]])) for n in (10, 25, 50, 100) if rows[:n]}


def offline_fit(plastic, cached, fit_n=50, steps_opt=80):
    train, test = cached[:fit_n], cached[fit_n:]
    P = torch.nn.Parameter(torch.zeros(1, plastic.out_dim, plastic.in_dim))
    opt = torch.optim.Adam([P], lr=0.05)

    def pack(rows):
        return (
            torch.cat([plastic.featurize(r["raw"]) for r in rows]),
            torch.cat([r["base"] for r in rows]),
            torch.cat([r["post"] for r in rows]),
        )

    X, B, Q = pack(train)
    for _ in range(steps_opt):
        opt.zero_grad()
        pred = B + torch.einsum("oi,bi->bo", P[0], X)
        kl, _ = kl_rows(Q, pred)
        (kl.mean() + 1e-4 * P.pow(2).mean()).backward()
        opt.step()
        with torch.no_grad():
            flat = torch.linalg.vector_norm(P.reshape(1, -1)).clamp(min=1e-12)
            P.mul_(torch.clamp(plastic.max_norm / flat, max=1.0))
    with torch.no_grad():
        def score(rows):
            x, b, q = pack(rows)
            pred = b + torch.einsum("oi,bi->bo", P[0], x)
            kl, _ = kl_rows(q, pred)
            base, _ = kl_rows(q, b)
            return float(kl.mean()), float(base.mean())
        tr, te = score(train), score(test)
    return {"train_kl": tr[0], "train_frozen": tr[1], "heldout_kl": te[0], "heldout_frozen": te[1], "P": P.detach()}


def grad_alignment(post, prior):
    logit = prior.detach().clone().requires_grad_(True)
    kl, _ = kl_rows(post.detach(), logit)
    kl.sum().backward()
    g = logit.grad.detach().reshape(1, -1)
    q = mixed_probs(post).reshape(1, -1)
    p = mixed_probs(prior).reshape(1, -1)
    pq = torch.softmax(post.reshape(1, -1, NCLASS), -1).reshape(1, -1)
    pp = torch.softmax(prior.reshape(1, -1, NCLASS), -1).reshape(1, -1)

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))

    return {"cos_unimix_q_minus_p": cos(g, q - p), "cos_plain_q_minus_p": cos(g, pq - pp), "grad_norm": float(g.norm()), "prob_err_norm": float((q - p).norm())}


@torch.no_grad()
def action_flips(agent, plastic, cached, P, n=15):
    """Re-predict one step for 4 matched candidates. Count argmax flips vs P=0."""
    dev = agent.device
    P = P.to(dev)
    flips, spans, rhos = 0, [], []
    act_dim = agent.act_dim
    for i, row in enumerate(cached[:n]):
        post = row["post_state"].to(dev)
        deter = row["deter"].to(dev)
        g = torch.Generator(device="cpu")
        g.manual_seed(1000 + i)
        noise = torch.randn(4, act_dim, generator=g).to(dev)
        base_a = row["action"].to(dev).view(1, -1)
        cands = (base_a + 0.25 * noise).clamp(-1, 1)
        cands[0] = base_a
        sf, sp = [], []
        for j in range(4):
            action = cands[j:j+1]
            next_d = agent._frozen_rssm._deter_net(post, deter, action)
            base = agent._frozen_rssm.prior_logit_from_deter(next_d).reshape(1, -1)
            raw = torch.cat([post.reshape(1, -1), deter, action / torch.clamp(action.abs(), min=1.0)], -1)
            x = plastic.featurize(raw.cpu()).to(dev)
            logit = base + torch.einsum("oi,bi->bo", P[0], x)
            def score(logit_):
                nxt = agent._frozen_rssm.get_dist(logit_.reshape_as(agent._frozen_rssm.prior_logit_from_deter(next_d))).mode
                feat = agent._frozen_rssm.get_feat(nxt, next_d)
                return float((agent._frozen_reward(feat).mode() + agent._frozen_cont(feat).mean * agent._frozen_value(feat).mode()).reshape(-1)[0])
            sf.append(score(base))
            sp.append(score(logit))
        sf, sp = np.array(sf), np.array(sp)
        flips += int(sf.argmax() != sp.argmax())
        spans.append(float(sp.max() - sp.min()))
        if sf.std() > 1e-6 and sp.std() > 1e-6:
            rhos.append(float(np.corrcoef(sf, sp)[0, 1]))
    return {"n": n, "argmax_flips": flips, "mean_score_span": float(np.mean(spans)), "mean_rank_corr": None if not rhos else float(np.mean(rhos))}


def eval_static(plastic, cached, P):
    xs = torch.cat([plastic.featurize(r["raw"]) for r in cached])
    b = torch.cat([r["base"] for r in cached])
    q = torch.cat([r["post"] for r in cached])
    pred = b + torch.einsum("oi,bi->bo", P[0], xs)
    kl, _ = kl_rows(q, pred)
    base, _ = kl_rows(q, b)
    return float(kl.mean()), float(base.mean())


def main():
    root = Path(__file__).resolve().parents[1]
    log = root / "logdir/dreamer_walker_pretrain_5202322"
    out = root / "logdir/v2_system1_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    agent, _, env = load_agent(log, log / "latest.pt", dev, seed=0)
    stub_render(env)
    k = agent.rssm._stoch * agent.rssm._discrete
    raw = k + agent.rssm._deter + agent.act_dim
    plastic = CausalPlasticPrior(out_dim=k, in_dim=raw, eta=ETA, decay=DECAY, feature_dim=64, proj_seed=0)
    steps = 101
    packs = {}
    report = {"device": dev, "unimix": UNIMIX, "eta": ETA, "seeds": {}}
    align = None
    for seed in (0, 1, 2):
        nom, actions = rollout(agent, env, seed, 1.0, steps)
        shifted, _ = rollout(agent, env, seed, 0.7, steps, actions=actions)
        natural, _ = rollout(agent, env, seed, 0.7, steps, actions=None)
        packs[seed] = {"nom": nom, "shift": shifted, "natural": natural}
        block = {}
        stored = {}
        for label, data in (("gain1", nom), ("gain07_paired", shifted), ("gain07_natural", natural)):
            out_rules = {}
            for rule in ("frozen", "prob", "logit"):
                rows, P, audit = apply_rule(plastic, data, rule)
                out_rules[rule] = {"kl": windows(rows), "audit": audit}
                if rule == "prob":
                    stored[label] = P
            corr = np.corrcoef([r["anorm"] for r in apply_rule(plastic, data, "frozen")[0]], [r["kl"] for r in apply_rule(plastic, data, "frozen")[0]])
            # avoid double compute
            block[label] = out_rules
        frozen_rows, _, _ = apply_rule(plastic, shifted, "frozen")
        block["action_kl_corr_shift"] = float(np.corrcoef([r["anorm"] for r in frozen_rows], [r["kl"] for r in frozen_rows])[0, 1])
        block["offline_shift"] = {k: v for k, v in offline_fit(plastic, shifted).items() if k != "P"}
        fit = offline_fit(plastic, shifted)
        block["offline_shift"] = {k: v for k, v in fit.items() if k != "P"}
        stored["offline"] = fit["P"]
        if align is None:
            align = grad_alignment(shifted[10]["post"], shifted[10]["base"])
        block["action_rank_online"] = action_flips(agent, plastic, shifted, stored["gain07_paired"])
        block["action_rank_offline"] = action_flips(agent, plastic, shifted, stored["offline"])
        report["seeds"][str(seed)] = block
        print(seed, "KL100 gain1", block["gain1"]["frozen"]["kl"]["n100"], "gain07", block["gain07_paired"]["frozen"]["kl"]["n100"], "offline", block["offline_shift"], flush=True)
        packs[seed]["P"] = stored
    transfers = {}
    for tgt in (1, 2):
        transfers[str(tgt)] = {
            "same_gain_from_seed0": eval_static(plastic, packs[tgt]["shift"], packs[0]["P"]["gain07_paired"]),
            "nominal_memory_on_shift": eval_static(plastic, packs[tgt]["shift"], packs[0]["P"]["gain1"]),
            "shift_memory_on_nominal": eval_static(plastic, packs[tgt]["nom"], packs[0]["P"]["gain07_paired"]),
            "offline_seed0_on_shift": eval_static(plastic, packs[tgt]["shift"], packs[0]["P"]["offline"]),
        }
    report["transfers_kl_and_frozen"] = transfers
    report["grad_alignment"] = align
    (out / "diagnosis.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / "diagnosis.json", flush=True)


if __name__ == "__main__":
    main()
