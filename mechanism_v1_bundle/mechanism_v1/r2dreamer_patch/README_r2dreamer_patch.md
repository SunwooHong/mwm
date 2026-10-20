# R2-Dreamer / DreamerV3 Integration Patch Sketch

Do this only after the synthetic unit test passes.

## Why prior-logit residual first?

In the R2-Dreamer RSSM, the deterministic transition is computed first, then the observation path produces posterior logits and the imagination path produces prior logits. The cleanest signed local error for a first Dreamer integration is therefore:

```text
error = stopgrad(post_logit - prior_logit)
```

So the first patch should add the plastic residual to **prior logits**, not directly to `deter`.

## Conceptual patch

In `RSSM.__init__`:

```python
from r2dreamer_patch.logit_plastic_trace import LogitPlasticResidual

self._plastic = LogitPlasticResidual(
    flat_stoch=self.flat_stoch,
    deter=self._deter,
    act_dim=act_dim,
    out_dim=self._stoch * self._discrete,
    lr=config.plastic_lr,
    decay=config.plastic_decay,
    max_norm=config.plastic_max_norm,
)
```

Change `initial` to also return trace:

```python
trace = self._plastic.initial(batch_size, self._device)
return stoch, deter, trace
```

In `obs_step`, after the deterministic transition:

```python
deter = self._deter_net(stoch, deter, prev_action)
x_plastic = self._plastic.make_input(stoch, deter, prev_action)
base_prior_logit = self._img_net(deter)
prior_logit = base_prior_logit + self._plastic.read(trace, x_plastic).reshape_as(base_prior_logit)

x_obs = torch.cat([deter, embed], dim=-1)
post_logit = self._obs_net(x_obs)

flat_error = (post_logit.detach() - prior_logit.detach()).reshape(post_logit.shape[0], -1)
trace, gate = self._plastic.write(trace, x_plastic, flat_error)

stoch = self.get_dist(post_logit).rsample()
return stoch, deter, post_logit, prior_logit, trace, gate
```

In `img_step`, read but do not write:

```python
deter = self._deter_net(stoch, deter, prev_action)
x_plastic = self._plastic.make_input(stoch, deter, prev_action)
base_prior_logit = self._img_net(deter)
prior_logit = base_prior_logit + self._plastic.read(trace, x_plastic).reshape_as(base_prior_logit)
stoch = self.get_dist(prior_logit).rsample()
return stoch, deter, prior_logit, trace
```

You will then need to thread `trace` through `observe`, `img_step`, and `imagine_with_action`. For initial experiments, disable trace writes during training and enable them only during shifted-environment evaluation.
