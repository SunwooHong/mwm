"""Deterministic audit of legacy plastic control path and causal System 1 v2."""

import torch

from plastic.causal_s1 import CausalPlasticPrior, prequential_step, select_action
from plastic.logit_plastic_trace import LogitPlasticResidual


def test_legacy_write_does_not_require_grad_and_clips_separately():
    plastic = LogitPlasticResidual(flat_stoch=8, deter=4, act_dim=2, out_dim=8, lr=0.1, gate_mode="always_on")
    trace = plastic.initial(2, "cpu")
    x = torch.randn(2, plastic.in_dim)
    signed = torch.randn(2, 8) * 10
    raw_mse = signed.pow(2).mean(-1)
    new, gate, _ = plastic.write(trace, x, signed)
    assert gate.min() == 1
    assert not new.P.requires_grad
    assert raw_mse.max() > 4
    assert torch.linalg.vector_norm(new.P.reshape(2, -1), dim=-1).max() <= plastic.max_norm + 1e-4


def test_legacy_matrix_is_not_lightweight_at_size12m():
    stoch, discrete, deter, act = 32, 16, 2048, 6
    k = stoch * discrete
    d = k + deter + act
    params = k * d
    assert params == 512 * 2566
    assert params * 4 > 5_000_000


def test_prequential_scores_before_write_and_p0_matches_base():
    torch.manual_seed(0)
    b, k, classes = 2, 16, 4
    plastic = CausalPlasticPrior(out_dim=k, in_dim=10, eta=1.0, decay=0.0, max_norm=20.0)
    trace = plastic.initial(b, "cpu")
    raw = torch.randn(b, 10)
    base = torch.randn(b, k)
    post = base.clone()
    post[:, 0::classes] += 3.0
    trace2, m1 = prequential_step(plastic, trace, raw_x=raw, base_logit=base, post_logit=post, n_classes=classes)
    assert torch.allclose(m1["kl"], m1["kl_base"])
    assert torch.allclose(m1["correction"], torch.zeros(b))
    _, m2 = prequential_step(plastic, trace2, raw_x=raw, base_logit=base, post_logit=post, n_classes=classes)
    assert (m2["kl"] < m1["kl"]).all()
    assert m2["correction"].min() > 0


def test_selector_changes_action_only_when_read():
    torch.manual_seed(1)
    b, k, n = 1, 8, 4
    plastic = CausalPlasticPrior(out_dim=k, in_dim=6, eta=0.5, max_norm=8)
    trace = plastic.initial(b, "cpu")
    x = torch.zeros(b, 6)
    x[:, 3] = 1.0
    base = torch.zeros(b, k)
    q = torch.zeros(b, k)
    q[:, 0] = 1
    p = torch.softmax(base, -1)
    trace = plastic.write(trace, plastic.featurize(x), q, p)
    cands = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])

    def prior_fn(action):
        raw = torch.zeros(b, 6)
        idx = int(action.reshape(-1)[0].item())
        raw[:, idx] = 1.0
        return raw, base

    def score_fn(logit):
        return logit[:, 0]

    off, _ = select_action(plastic, trace, candidates=cands, score_fn=score_fn, prior_fn=prior_fn, read=False)
    on, scores = select_action(plastic, trace, candidates=cands, score_fn=score_fn, prior_fn=prior_fn, read=True)
    assert torch.equal(off, cands[:, 0])
    assert not torch.equal(on, off)
    assert scores.shape == (b, n)


def test_envs_isolated_and_reset():
    plastic = CausalPlasticPrior(out_dim=4, in_dim=3, eta=1.0, decay=0.0, max_norm=10)
    trace = plastic.initial(2, "cpu")
    x = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
    q = torch.tensor([[1.0, 0, 0, 0], [0.0, 0, 0, 1]])
    p = torch.full((2, 4), 0.25)
    trace = plastic.write(trace, x, q, p)
    assert not torch.allclose(trace.P[0], trace.P[1])
    trace = plastic.reset_where(trace, torch.tensor([True, False]))
    assert torch.count_nonzero(trace.P[0]) == 0
    assert torch.count_nonzero(trace.P[1]) > 0


def test_no_backward_on_write():
    plastic = CausalPlasticPrior(out_dim=4, in_dim=3)
    trace = plastic.initial(1, "cpu")
    x = torch.randn(1, 3, requires_grad=True)
    with torch.no_grad():
        x = x.detach()
    out = plastic.write(trace, x, torch.zeros(1, 4), torch.ones(1, 4) / 4)
    assert out.P.grad_fn is None


if __name__ == "__main__":
    test_legacy_write_does_not_require_grad_and_clips_separately()
    test_legacy_matrix_is_not_lightweight_at_size12m()
    test_prequential_scores_before_write_and_p0_matches_base()
    test_selector_changes_action_only_when_read()
    test_envs_isolated_and_reset()
    test_no_backward_on_write()
    print("ok")
