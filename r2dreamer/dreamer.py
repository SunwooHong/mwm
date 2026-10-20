import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from networks import Projector
from plastic import LogitPlasticResidual, PlasticTrace
from plastic.causal_s1 import CausalPlasticPrior, CausalTrace
from optim import LaProp, clip_grad_agc_
from tools import to_f32


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm.RSSM(
            config.rssm,
            self.embed_size,
            self.act_dim,
        )
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        plastic_cfg = getattr(config, "plastic", None)
        self.plastic_eval_only = bool(getattr(plastic_cfg, "eval_only", True)) if plastic_cfg else True
        self._plastic_inference = bool(getattr(plastic_cfg, "inference", False)) if plastic_cfg else False
        self._plastic = None
        self.last_plastic_metrics = {}
        self._plastic_flush_boost = 0.0
        self._flush_mode = "off"
        self._flush_boost_max = 0.5
        self._slow_wm_loss_ema = float("inf")
        self._slow_wm_updates = 0
        self._plastic_flush_k_base = 0.0
        if plastic_cfg and bool(getattr(plastic_cfg, "enabled", False)):
            self._plastic = LogitPlasticResidual(
                flat_stoch=self.rssm.flat_stoch,
                deter=self.rssm._deter,
                act_dim=self.act_dim,
                out_dim=self.rssm.flat_stoch,
                lr=float(plastic_cfg.lr),
                decay=float(getattr(plastic_cfg, "decay", 0.001)),
                max_norm=float(plastic_cfg.max_norm),
                persist_threshold=float(plastic_cfg.persist_threshold),
                gate_mode=str(plastic_cfg.gate_mode),
                flush_k=float(getattr(plastic_cfg, "flush_k", 0.0)),
                flush_err_scale=float(getattr(plastic_cfg, "flush_err_scale", 0.15)),
            )
            self._flush_boost_max = float(getattr(plastic_cfg, "flush_boost_max", 0.5))
            self._plastic_flush_k_base = float(getattr(plastic_cfg, "flush_k", 0.0))

        self.train()
        self.clone_and_freeze()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        if hasattr(self, "decoder"):
            self._frozen_decoder = copy.deepcopy(self.decoder)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.decoder.named_parameters(), self._frozen_decoder.named_parameters()
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    def set_plastic_inference(self, active: bool) -> None:
        """Toggle test-time plastic updates inside act()."""
        self._plastic_inference = bool(active)

    def set_consolidation_mode(self, mode: str) -> None:
        """Flush mode: off | coupled (slow θ + decay boost) | error_only (flush without slow θ)."""
        self._flush_mode = str(mode)
        if mode == "off":
            self._plastic_flush_boost = 0.0
            if self._plastic is not None:
                self._plastic.flush_k = 0.0
        elif mode == "coupled":
            if self._plastic is not None:
                self._plastic.flush_k = self._plastic_flush_k_base
        elif mode == "error_only":
            self._plastic_flush_boost = 0.0
            if self._plastic is not None:
                self._plastic.flush_k = self._plastic_flush_k_base
        else:
            raise ValueError(f"Unknown flush mode: {mode}")

    def sync_frozen_wm(self) -> None:
        """Copy live encoder/rssm/(decoder) into frozen inference copies after slow θ update."""
        with torch.no_grad():
            for live, frozen in (
                (self.encoder, self._frozen_encoder),
                (self.rssm, self._frozen_rssm),
            ):
                for s, d in zip(live.parameters(), frozen.parameters()):
                    d.data.copy_(s.data)
            if hasattr(self, "decoder"):
                for s, d in zip(self.decoder.parameters(), self._frozen_decoder.parameters()):
                    d.data.copy_(s.data)

    def _update_flush_boost(self, slow_dyn_loss: float) -> None:
        if self._flush_mode != "coupled":
            return
        if self._slow_wm_loss_ema == float("inf"):
            self._slow_wm_loss_ema = slow_dyn_loss
        improving = slow_dyn_loss < self._slow_wm_loss_ema
        self._slow_wm_loss_ema = 0.9 * self._slow_wm_loss_ema + 0.1 * slow_dyn_loss
        if self._slow_wm_updates >= 2 and improving:
            self._plastic_flush_boost = min(
                self._flush_boost_max,
                self._plastic_flush_boost + 0.05,
            )
        else:
            self._plastic_flush_boost *= 0.98

    @staticmethod
    def _flatten_logit(logit: torch.Tensor) -> torch.Tensor:
        return logit.reshape(logit.shape[0], -1)

    def _pack_trace(self, trace: PlasticTrace) -> dict:
        packed = {
            "plastic_P": trace.P,
            "plastic_ema": trace.ema,
            "plastic_var": trace.var,
            "plastic_persist": trace.persist,
        }
        if trace.prev_err is not None:
            packed["plastic_prev_err"] = trace.prev_err
        return packed

    def _unpack_trace(self, state: TensorDict) -> PlasticTrace:
        return PlasticTrace(
            P=state["plastic_P"],
            ema=state["plastic_ema"],
            var=state["plastic_var"],
            persist=state["plastic_persist"],
            prev_err=state.get("plastic_prev_err"),
        )

    def _plastic_should_run(self, eval: bool) -> bool:
        if self._plastic is None or not self._plastic_inference:
            return False
        if self.plastic_eval_only and not eval:
            return False
        return True

    @torch.no_grad()
    def _img_step_plan(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        action: torch.Tensor,
        trace: PlasticTrace | None,
        *,
        use_plastic_read: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_deter = deter
        deter = self._frozen_rssm._deter_net(stoch, deter, action)
        prior_logit = self._frozen_rssm.prior_logit_from_deter(deter)
        if use_plastic_read and self._plastic is not None and trace is not None:
            x = self._plastic.make_input(stoch, prev_deter, action)
            prior_flat = self._flatten_logit(prior_logit) + self._plastic.read(trace, x)
            prior_logit = prior_flat.reshape_as(prior_logit)
        stoch = self._frozen_rssm.get_dist(prior_logit).rsample()
        return stoch, deter

    @torch.no_grad()
    def _observe_step(
        self,
        obs,
        state: TensorDict,
        *,
        use_plastic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, TensorDict]:
        p_obs = self.preprocess(obs)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = state["stoch"], state["deter"], state["prev_action"]
        is_first = obs["is_first"].reshape(state.batch_size[0])

        if use_plastic:
            if "plastic_P" not in state.keys():
                trace = self._plastic.initial(state.batch_size[0], self.device)
            else:
                trace = self._unpack_trace(state)
            trace = self._plastic.reset_where(trace, is_first)
            self._plastic.reset_dir_persist_where(is_first)

            stoch, deter, post_logit = self._frozen_rssm._obs_step_core(
                prev_stoch, prev_deter, prev_action, embed, is_first
            )
            post_flat = self._flatten_logit(post_logit)
            prior_base = self._flatten_logit(self._frozen_rssm.prior_logit_from_deter(deter))
            x = self._plastic.make_input(stoch, prev_deter, prev_action)
            prior_flat = prior_base + self._plastic.read(trace, x)
            trace, gate, decay_eff = self._plastic.write(
                trace,
                x,
                post_flat.detach() - prior_flat.detach(),
                flush_boost=self._plastic_flush_boost,
            )
            stoch = self._frozen_rssm.get_dist(post_logit).rsample()
            self.last_plastic_metrics = {
                "prior_mse": float((post_flat.detach() - prior_flat).pow(2).mean().item()),
                "gate": float(gate.mean().item()),
                "trace_norm": float(torch.linalg.vector_norm(trace.P.reshape(trace.P.shape[0], -1)).mean().item()),
                "lambda_eff": float(decay_eff.mean().item()),
                "flush_boost": float(self._plastic_flush_boost),
            }
            next_state = TensorDict(
                {
                    "stoch": stoch,
                    "deter": deter,
                    "prev_action": prev_action,
                    **self._pack_trace(trace),
                },
                batch_size=state.batch_size,
            )
        else:
            stoch, deter, post_logit = self._frozen_rssm._obs_step_core(
                prev_stoch, prev_deter, prev_action, embed, is_first
            )
            post_flat = self._flatten_logit(post_logit)
            prior_flat = self._flatten_logit(self._frozen_rssm.prior_logit_from_deter(deter))
            stoch = self._frozen_rssm.get_dist(post_logit).rsample()
            next_state = TensorDict(
                {"stoch": stoch, "deter": deter, "prev_action": prev_action},
                batch_size=state.batch_size,
            )
            self.last_plastic_metrics = {
                "prior_mse": float((post_flat.detach() - prior_flat).pow(2).mean().item()),
                "gate": 0.0,
                "trace_norm": 0.0,
            }
        return stoch, deter, next_state

    @torch.no_grad()
    def act_planner(
        self,
        obs,
        state: TensorDict,
        *,
        use_plastic_wm: bool,
        cem_horizon: int = 12,
        cem_samples: int = 256,
        cem_elites: int = 32,
        cem_iterations: int = 5,
    ) -> tuple[torch.Tensor, TensorDict]:
        """Observe, then plan with CEM using frozen reward head (+ optional plastic WM)."""
        from plastic.cem_planner import CEMConfig, plan_cem

        _, _, next_state = self._observe_step(obs, state, use_plastic=use_plastic_wm)
        action = plan_cem(
            self,
            next_state,
            use_plastic_wm=use_plastic_wm,
            config=CEMConfig(
                horizon=cem_horizon,
                num_samples=cem_samples,
                num_elites=cem_elites,
                iterations=cem_iterations,
            ),
        )
        next_state["prev_action"] = action
        return action, next_state

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        use_plastic = self._plastic_should_run(eval)
        stoch, deter, next_state = self._observe_step(obs, state, use_plastic=use_plastic)

        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()
        next_state["prev_action"] = action
        return action, next_state

    def enable_causal_s1(self, *, feature_dim: int = 64, eta: float = 0.05, proj_seed: int = 0) -> None:
        """Attach MWM v2 memory. Does not change Dreamer.act."""
        k = self.rssm._stoch * self.rssm._discrete
        raw = k + self.rssm._deter + self.act_dim
        self._causal = CausalPlasticPrior(
            out_dim=k,
            in_dim=raw,
            eta=eta,
            feature_dim=feature_dim,
            proj_seed=proj_seed,
        ).to(self.device)
        self._causal_n_classes = int(self.rssm._discrete)

    @torch.no_grad()
    def act_causal(
        self,
        obs,
        state: TensorDict,
        *,
        read_plastic: bool,
        write_plastic: bool,
        n_candidates: int = 4,
        cand_seed: int = 0,
    ) -> tuple[torch.Tensor, TensorDict, dict]:
        """One-step selector. Original act() is unchanged.

        The observation updates P only as the target of the prediction stored
        on the previous call. Candidate priors are scored before that write's
        result is applied to the same observation.
        """
        if not hasattr(self, "_causal"):
            self.enable_causal_s1()
        b = state.batch_size[0]
        if "causal_P" not in state.keys():
            state = state.update({"causal_P": self._causal.initial(b, self.device).P})
        trace = CausalTrace(P=state["causal_P"])
        is_first = obs["is_first"].reshape(b)
        trace = self._causal.reset_where(trace, is_first)

        p_obs = self.preprocess(obs)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = state["stoch"], state["deter"], state["prev_action"]
        stoch, deter, post_shaped = self._frozen_rssm._obs_step_core(
            prev_stoch, prev_deter, prev_action, embed, is_first
        )
        post_logit = post_shaped.reshape(b, -1)
        pred_metrics = {"preupdate_kl": torch.zeros(b, device=self.device)}
        if "pending_logit" in state.keys() and write_plastic:
            pending = state["pending_logit"]
            valid = ~is_first
            q = self._causal.probs(post_logit, self._causal_n_classes)
            p = self._causal.probs(pending, self._causal_n_classes)
            kl = (q * (q.clamp(min=1e-8).log() - p.clamp(min=1e-8).log())).sum(-1)
            pred_metrics["preupdate_kl"] = torch.where(valid, kl, torch.zeros_like(kl))
            if valid.any():
                x_prev = state["pending_x"]
                trace = self._causal.write(trace, x_prev, q, p)
                if not read_plastic:
                    pass

        stoch = self._frozen_rssm.get_dist(post_shaped).mode
        feat = self._frozen_rssm.get_feat(stoch, deter)
        # actor and _frozen_actor share .data after clone_and_freeze, so S2 updates are visible here.
        dist = self.actor(feat)
        mode = dist.mode
        g = torch.Generator(device="cpu")
        g.manual_seed(int(cand_seed))
        noise = torch.randn(b, n_candidates, self.act_dim, generator=g).to(mode.device)
        cands = (mode.unsqueeze(1) + 0.25 * noise).clamp(-1, 1)
        cands[:, 0] = mode

        scores = []
        pred_logits = []
        xs = []
        for i in range(n_candidates):
            action_i = cands[:, i]
            prev_d = deter
            next_d = self._frozen_rssm._deter_net(stoch, deter, action_i)
            base_shaped = self._frozen_rssm.prior_logit_from_deter(next_d)
            base = base_shaped.reshape(b, -1)
            raw = torch.cat(
                [
                    stoch.reshape(b, -1),
                    prev_d,
                    action_i / torch.clamp(action_i.abs(), min=1.0),
                ],
                dim=-1,
            )
            x = self._causal.featurize(raw)
            logit = self._causal.read_logits(trace, x, base) if read_plastic else base
            nxt = self._frozen_rssm.get_dist(logit.reshape_as(base_shaped)).mode
            fut = self._frozen_rssm.get_feat(nxt, next_d)
            rew = self._frozen_reward(fut).mode().reshape(b)
            cont = self._frozen_cont(fut).mean.reshape(b)
            val = self._frozen_value(fut).mode().reshape(b)
            scores.append(rew + cont * val)
            pred_logits.append(logit)
            xs.append(x)
        stacked = torch.stack(scores, dim=-1)
        idx = stacked.argmax(dim=-1)
        batch_ix = torch.arange(b, device=self.device)
        action = cands[batch_ix, idx]
        chosen_logit = torch.stack(pred_logits, dim=1)[batch_ix, idx]
        chosen_x = torch.stack(xs, dim=1)[batch_ix, idx]
        changed = ~torch.all(torch.isclose(action, mode), dim=-1)
        info = {
            "scores": stacked.detach(),
            "score_span": (stacked.max(-1).values - stacked.min(-1).values).detach(),
            "changed_from_mode": changed.detach(),
            "preupdate_kl": pred_metrics["preupdate_kl"].detach(),
            "trace_norm": torch.linalg.vector_norm(trace.P.reshape(b, -1), dim=-1).detach(),
            "read_plastic": read_plastic,
            "write_plastic": write_plastic,
        }
        next_state = TensorDict(
            {
                "stoch": stoch,
                "deter": deter,
                "prev_action": action,
                "causal_P": trace.P,
                "pending_logit": chosen_logit,
                "pending_x": chosen_x,
            },
            batch_size=state.batch_size,
        )
        return action, next_state, info

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        state = TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,))
        if self._plastic is not None:
            state.update(self._pack_trace(self._plastic.initial(B, self.device)))
        return state

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # update latent vectors in replay buffer
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # reward and continue
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # (B, T, ...) -> (B*T, ...)
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (B*T, T_imag, 1)
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    def _expand_trace(self, trace: PlasticTrace, batch_size: int) -> PlasticTrace:
        if batch_size == trace.P.shape[0]:
            return trace
        expand = lambda t: t.expand(batch_size, *t.shape[1:])
        return PlasticTrace(
            P=expand(trace.P),
            ema=expand(trace.ema),
            var=expand(trace.var),
            persist=expand(trace.persist),
            prev_err=None if trace.prev_err is None else expand(trace.prev_err),
        )

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, trace: PlasticTrace | None = None, *, use_plastic_wm: bool = False):
        """Roll out the policy in latent space (frozen or plastic prior dynamics)."""
        # (B, S, K), (B, D)
        feats = []
        actions = []
        stoch, deter = start
        batch_size = stoch.shape[0]
        tr = self._expand_trace(trace, batch_size) if trace is not None and use_plastic_wm else None
        actor = self.actor if use_plastic_wm else self._frozen_actor
        for _ in range(imag_horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter)
            action = actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            if use_plastic_wm and tr is not None:
                stoch, deter = self._img_step_plan(stoch, deter, action, tr, use_plastic_read=True)
            else:
                stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)

        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    def _actor_only_grad_mask(self) -> None:
        for name, param in self._named_params.items():
            if param.grad is not None and not name.startswith("actor."):
                param.grad = None

    def _cal_grad_actor_plastic(
        self,
        data,
        initial,
        trace: PlasticTrace | None,
        *,
        use_plastic_wm: bool,
    ):
        """Actor policy loss on imagined rollouts (plastic or frozen WM). WM/reward heads frozen."""
        losses = {}
        metrics = {}
        B, T = data.shape

        embed = self._frozen_encoder(data)
        with torch.no_grad():
            post_stoch, post_deter, _ = self._frozen_rssm.observe(embed, data["action"], initial, data["is_first"])
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1, trace, use_plastic_wm=use_plastic_wm)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        metrics["adv"] = torch.mean(adv)
        metrics["rew_imag"] = torch.mean(imag_reward)
        metrics["action_entropy"] = torch.mean(entropy)
        return losses, metrics

    def _set_opt_lr(self, scale: float) -> None:
        base = getattr(self, "_opt_base_lr", None)
        if base is None:
            base = float(self._optimizer.param_groups[0]["lr"])
            self._opt_base_lr = base
        lr = base * float(scale)
        for group in self._optimizer.param_groups:
            group["lr"] = lr

    def update_actor_plastic(
        self,
        replay_buffer,
        trace: PlasticTrace | None,
        *,
        use_plastic_wm: bool,
    ) -> dict:
        """Single actor-only update using plastic (or frozen) imagination dynamics."""
        self._set_opt_lr(getattr(self, "_actor_lr_scale", 1.0))
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            loss_terms, mets = self._cal_grad_actor_plastic(p_data, initial, trace, use_plastic_wm=use_plastic_wm)
            policy_loss = loss_terms["policy"] * self._loss_scales["policy"]
        self._optimizer.zero_grad(set_to_none=True)
        self._scaler.scale(policy_loss).backward()
        self._scaler.unscale_(self._optimizer)
        self._actor_only_grad_mask()
        self._agc([p for n, p in self._named_params.items() if n.startswith("actor.")])
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad(set_to_none=True)
        metrics.update({f"adapt/{k}": v for k, v in mets.items()})
        metrics["adapt/loss_policy"] = policy_loss.detach()
        metrics["adapt/lr"] = float(self._optimizer.param_groups[0]["lr"])
        return metrics

    def _wm_only_grad_mask(self) -> None:
        wm_prefixes = ("encoder.", "rssm.", "decoder.", "projector.")
        for name, param in self._named_params.items():
            if param.grad is not None and not name.startswith(wm_prefixes):
                param.grad = None

    def _cal_grad_slow_wm(self, data, initial):
        """World-model-only loss (dynamics + representation) for slow θ fine-tuning."""
        losses = {}
        metrics = {}
        B, T = data.shape
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key]))
                for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            feat = self.rssm.get_feat(post_stoch, post_deter)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        else:
            raise NotImplementedError(f"slow wm update not implemented for rep_loss={self.rep_loss}")
        total_loss = sum(v * self._loss_scales[k] for k, v in losses.items() if k in self._loss_scales)
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics["slow/opt_loss"] = total_loss
        return total_loss, metrics

    def update_slow_wm(self, replay_buffer) -> dict:
        """Single slow world-model update; syncs frozen WM and may boost plastic flush."""
        self._set_opt_lr(getattr(self, "_wm_lr_scale", 0.02))
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            wm_loss, mets = self._cal_grad_slow_wm(p_data, initial)
        self._optimizer.zero_grad(set_to_none=True)
        self._scaler.scale(wm_loss).backward()
        self._scaler.unscale_(self._optimizer)
        self._wm_only_grad_mask()
        wm_params = [
            p
            for n, p in self._named_params.items()
            if n.startswith(("encoder.", "rssm.", "decoder.", "projector."))
        ]
        self._agc(wm_params)
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad(set_to_none=True)
        self.sync_frozen_wm()
        self._slow_wm_updates += 1
        dyn_loss = float(mets["loss/dyn"].detach())
        self._update_flush_boost(dyn_loss)
        metrics.update({f"slow/{k.replace('loss/', 'loss_')}": v for k, v in mets.items() if k.startswith("loss/")})
        metrics["slow/opt_loss"] = wm_loss.detach()
        metrics["slow/dyn_loss"] = mets["loss/dyn"].detach()
        metrics["slow/flush_boost"] = torch.tensor(self._plastic_flush_boost)
        metrics["slow/wm_updates"] = torch.tensor(float(self._slow_wm_updates))
        return metrics

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
