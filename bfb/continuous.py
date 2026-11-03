"""Continuous-action FB, BFB and RFB with a DDPG actor and frozen dynamics encoder.

The FB objective is shared conceptually with the discrete implementation; the target policy is an actor instead
of an enumeration of actions. The actor maximizes F(s,a,z,h)^T z. B remains independent of dynamics.
"""

from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from bfb.agent import FBAgent, State
from bfb.networks import MLP, Backward, DynamicsEncoder, NextStatePredictor, project_to_sphere
from bfb.vmf import w_quantile_table


class ForwardContinuous(nn.Module):
    dim: int
    hidden: tuple
    embed: int

    @nn.compact
    def __call__(self, obs, action, z, h):
        task = z if h is None else jnp.concatenate([z, h], -1)
        state_action = MLP((self.embed,), self.embed)(jnp.concatenate([obs, action], -1))
        state_task = MLP((self.embed,), self.embed)(jnp.concatenate([obs, task], -1))
        ensemble = nn.vmap(
            MLP, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0, axis_size=2
        )
        return ensemble(self.hidden, self.dim)(jnp.concatenate([state_action, state_task], -1))


class Actor(nn.Module):
    action_dim: int
    hidden: tuple

    @nn.compact
    def __call__(self, obs, z, h):
        x = jnp.concatenate([obs, z] if h is None else [obs, z, h], -1)
        return nn.tanh(MLP(self.hidden, self.action_dim)(x))


class ContinuousFBAgent(FBAgent):
    def __init__(self, cfg, obs_dim, action_dim=2, seed=0):
        self.cfg = cfg
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.use_context = bool(cfg.use_context)
        if cfg.z_sampling == "vmf" and (not self.use_context or cfg.h_dim != cfg.z_dim):
            raise ValueError("RFB requires context and matching h_dim/z_dim")
        if cfg.obs_encoding != "xy" or cfg.cone_kappa is not None or cfg.oracle_context:
            raise ValueError("continuous agents require xy inputs and do not support grid cone/oracle-ID variants")
        self.h_stats = None
        self.z_w_table = w_quantile_table(cfg.kappa, cfg.z_dim) if cfg.z_sampling == "vmf" else None
        self.task_w_table = w_quantile_table(cfg.task_kappa, cfg.z_dim) if cfg.task_kappa is not None else None
        self.h_w_table = (
            w_quantile_table(cfg.belief_kappa, cfg.h_dim) if self.use_context and cfg.belief_kappa is not None else None
        )
        self.forward = ForwardContinuous(cfg.z_dim, tuple(cfg.f_hidden_dims), cfg.f_embed_dim)
        self.backward = Backward(cfg.z_dim, tuple(cfg.b_hidden_dims))
        self.actor = Actor(action_dim, tuple(cfg.actor_hidden_dims))
        self.encoder = DynamicsEncoder(
            cfg.h_dim,
            cfg.encoder.emb_dim,
            cfg.encoder.num_layers,
            cfg.encoder.num_heads,
            cfg.encoder.mlp_dim,
            cfg.encoder.delta_scale,
        )
        self.predictor = NextStatePredictor(tuple(cfg.encoder.predictor_hidden_dims), obs_dim)
        kf, kb, ka, ke, kp = jax.random.split(jax.random.key(seed), 5)
        obs, action, z = jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim)), jnp.zeros((1, cfg.z_dim))
        h = jnp.zeros((1, cfg.h_dim)) if self.use_context else None
        params = {
            "forward": self.forward.init(kf, obs, action, z, h)["params"],
            "backward": self.backward.init(kb, obs)["params"],
        }
        actor_params = self.actor.init(ka, obs, z, h)["params"]
        self.fb_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))
        self.actor_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.actor_lr))
        # Keep actor/critic states together so the inherited scan, checkpoint and belief bank remain applicable.
        all_params = {**params, "actor": actor_params}
        opts = {"fb": self.fb_tx.init(params), "actor": self.actor_tx.init(actor_params)}
        self.state = State(all_params, all_params, opts, jnp.zeros((), jnp.int32))
        self.enc_state = None
        if self.use_context:
            enc_params = {
                "encoder": self.encoder.init(ke, obs[:, None], action[:, None], obs[:, None])["params"],
                "predictor": self.predictor.init(kp, obs[:, None], action[:, None], h[:, None])["params"],
            }
            self.enc_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.encoder.lr))
            self.enc_state = State(enc_params, None, self.enc_tx.init(enc_params), jnp.zeros((), jnp.int32))

    def onehot(self, actions):
        return actions  # continuous actions are already vectors

    def B(self, params, obs):
        b = self.backward.apply({"params": params["backward"]}, obs)
        return project_to_sphere(b) if self.cfg.normalize_b else b

    def F(self, params, obs, action, z, h):
        return self.forward.apply({"params": params["forward"]}, obs, action, z, h)

    def update_fb(self, state, batch, key):
        cfg = self.cfg
        kz, ka = jax.random.split(key)
        z = self.sample_z(state.params, batch, kz)
        h = batch.get("h")
        batch_size = z.shape[0]
        next_action = self.actor.apply({"params": state.target_params["actor"]}, batch["next_obs"], z, h)
        noise = jnp.clip(cfg.actor_noise * jax.random.normal(ka, next_action.shape), -0.3, 0.3)
        next_action = jnp.clip(next_action + noise, -1, 1)
        target_f = self.F(state.target_params, batch["next_obs"], next_action, z, h)
        target_b = self.B(state.target_params, batch["next_obs"])
        target_values = jnp.einsum("ebd,cd->ebc", target_f, target_b)
        target_m = target_values.min(0) if cfg.target_agg == "min" else target_values.mean(0)
        off = 1 - jnp.eye(batch_size)
        critic_params = {k: state.params[k] for k in ("forward", "backward")}

        def critic_loss(params):
            f = self.F(params, batch["obs"], batch["actions"], z, h)
            b = self.B(params, batch["next_obs"])
            m = jnp.einsum("ebd,cd->ebc", f, b)
            continuation = batch.get("not_done", jnp.ones(batch_size))
            error = m - cfg.discount * continuation[None, :, None] * target_m[None]
            diagonal_weight = (1 - cfg.discount) if cfg.fb_loss == "lsif" else 1.0
            diagonal = -diagonal_weight * jnp.diagonal(m, axis1=1, axis2=2).mean()
            off_loss = (0.5 * ((error * off) ** 2).sum(-1) / (batch_size - 1)).mean()
            cov = b @ b.T
            ortho = -jnp.diag(cov).mean() + (0.5 * ((cov * off) ** 2).sum(-1) / (batch_size - 1)).mean()
            loss = off_loss + diagonal + cfg.ortho_coef * ortho
            return loss, {"fb/loss": off_loss + diagonal, "fb/ortho": ortho, "fb/diag": diagonal}

        (_, info), grads = jax.value_and_grad(critic_loss, has_aux=True)(critic_params)
        updates, fb_opt = self.fb_tx.update(grads, state.opt_state["fb"], critic_params)
        critic_params = optax.apply_updates(critic_params, updates)

        def actor_loss(params):
            action = self.actor.apply({"params": params}, batch["obs"], z, h)
            f = self.F(critic_params, batch["obs"], action, z, h)
            q = jnp.einsum("ebd,bd->eb", f, z).min(0)
            # Scale-invariant actor objective prevents drifting Q magnitudes from changing the effective LR.
            scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(q).mean(), 1.0))
            return (
                -q.mean() / scale
                + cfg.action_penalty * (action**2).mean()
                + cfg.get("bc_coef", 0.0) * ((action - batch["actions"]) ** 2).mean()
            )

        actor_loss_value, actor_grads = jax.value_and_grad(actor_loss)(state.params["actor"])
        updates, actor_opt = self.actor_tx.update(actor_grads, state.opt_state["actor"], state.params["actor"])
        actor_params = optax.apply_updates(state.params["actor"], updates)
        params = {**critic_params, "actor": actor_params}
        target = optax.incremental_update(params, state.target_params, cfg.tau)
        info["actor/loss"] = actor_loss_value
        return State(params, target, {"fb": fb_opt, "actor": actor_opt}, state.step + 1), info

    @partial(jax.jit, static_argnums=0)
    def act(self, params, obs, z, h=None):
        return self.actor.apply({"params": params["actor"]}, obs, z, h)

    def load(self, path):
        import pickle

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        params = jax.device_put(checkpoint["fb"])
        opts = {
            "fb": self.fb_tx.init({k: params[k] for k in ("forward", "backward")}),
            "actor": self.actor_tx.init(params["actor"]),
        }
        self.state = self.state.replace(params=params, target_params=params, opt_state=opts)
        if self.use_context:
            self.enc_state = self.enc_state.replace(params=jax.device_put(checkpoint["enc"]))
        self.h_stats = jax.device_put(checkpoint["h_stats"])
