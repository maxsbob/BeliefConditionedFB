"""Successor-feature style zero-shot baselines for discrete actions: HILP and Laplacian (LAP).

Both learn a state representation phi(s) and a latent-conditioned Q-function Q(s, a, z) for the intrinsic reward
defined by phi, with the policy pi_z(s) = argmax_a Q(s, a, z). They follow the JAX implementations in onestep-fb
(https://github.com/chongyi-zheng/onestep-fb), with a greedy discrete Q-learning head instead of an actor.

* HILP (Park et al., 2024): phi is trained so that -||phi(s) - phi(g)|| is the temporal distance (IVL with expectile
  regression on goal-conditioned rewards). Intrinsic reward r = (phi(s') - phi(s))^T z; for a goal g the task vector
  is z = (phi(g) - phi(s_t)) / ||.||, recomputed at every step (Park et al., 2024).
* LAP (Wu et al., 2019): phi minimises ||phi(s) - phi(s')||^2 under an orthonormality constraint (graph Laplacian
  eigenfunctions). Intrinsic reward r = phi(s')^T z; for a goal g, z = phi(g) / ||.||.

Neither baseline is conditioned on the dynamics context (as in the paper).
"""

from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from bfb.agent import FBAgent, State
from bfb.data import sample_transitions
from bfb.networks import MLP, encode_obs, project_to_sphere


class Phi(nn.Module):
    """phi(s); an ensemble of 2 for HILP (as in the reference implementation)."""

    dim: int
    hidden_dims: tuple
    num_ensembles: int = 1

    @nn.compact
    def __call__(self, obs):
        if self.num_ensembles == 1:
            return MLP(self.hidden_dims, self.dim)(obs)
        ens = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_ensembles,
        )
        return ens(self.hidden_dims, self.dim)(obs)


class QNet(nn.Module):
    num_actions: int
    hidden_dims: tuple

    @nn.compact
    def __call__(self, obs, z):
        ens = nn.vmap(
            MLP, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0, axis_size=2
        )
        return ens(self.hidden_dims, self.num_actions)(jnp.concatenate([obs, z], -1))  # (2, B, A)


class LatentQAgent:
    use_context = False
    perturb_task = FBAgent.perturb_task

    def __init__(self, cfg, obs_dim: int, num_actions: int, episode_length: int, seed: int = 0):
        self.cfg = cfg
        self.num_actions = num_actions
        self.episode_length = episode_length
        self.method = cfg.name
        assert self.method in ("hilp", "lap")
        self.recompute_z_every_step = self.method == "hilp"
        self.phi = Phi(cfg.z_dim, tuple(cfg.phi_hidden_dims), 2 if self.method == "hilp" else 1)
        self.q = QNet(num_actions, tuple(cfg.q_hidden_dims))
        k1, k2 = jax.random.split(jax.random.key(seed))
        obs, z = encode_obs(jnp.zeros((1, obs_dim)), cfg.obs_encoding), jnp.zeros((1, cfg.z_dim))
        params = {"phi": self.phi.init(k1, obs)["params"], "q": self.q.init(k2, obs, z)["params"]}
        self.tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))
        self.state = State(params, params, self.tx.init(params), jnp.zeros((), jnp.int32))
        self.enc_state = None

    # ------------------------------------------------------------------ representation
    def phis(self, params, obs):
        out = self.phi.apply({"params": params["phi"]}, encode_obs(obs, self.cfg.obs_encoding))
        return out.mean(0) if self.method == "hilp" else out  # 'repr_agg = mean' for the HILP ensemble

    def hilp_loss(self, params, target_params, batch):
        cfg = self.cfg

        def dist(x, y):
            return -jnp.sqrt(jnp.maximum(((x - y) ** 2).sum(-1), 1e-6))

        enc = lambda o: encode_obs(o, cfg.obs_encoding)
        phi_t = lambda o: self.phi.apply({"params": target_params["phi"]}, enc(o))
        phi = lambda o: self.phi.apply({"params": params["phi"]}, enc(o))
        g = batch["goals"]
        reward, mask = batch["goal_rewards"], batch["goal_masks"]
        target_next_vs = dist(phi_t(batch["next_obs"]), phi_t(g))  # (2, B)
        q = reward + cfg.discount * mask * target_next_vs.min(0)
        adv = q - dist(phi_t(batch["obs"]), phi_t(g)).mean(0)
        qs = reward[None] + cfg.discount * mask[None] * target_next_vs
        vs = dist(phi(batch["obs"]), phi(g))
        weight = jnp.where(adv >= 0, cfg.expectile, 1 - cfg.expectile)
        return (weight[None] * (qs - vs) ** 2).mean(), {"repr/v": vs.mean()}

    def lap_loss(self, params, batch):
        phi = self.phis(params, batch["obs"])
        phi_next = self.phis(params, batch["next_obs"])
        phi_rand = self.phis(params, batch["goals"])  # independent states for the orthonormality term
        consistency = ((phi - phi_next) ** 2).sum(-1).mean()
        ortho = ((phi * phi_rand).sum(-1) ** 2 - (phi**2).sum(-1) - (phi_rand**2).sum(-1)).mean()
        return consistency + self.cfg.ortho_coef * ortho, {"repr/consistency": consistency, "repr/ortho": ortho}

    # ------------------------------------------------------------------ latents and intrinsic rewards
    def sample_z(self, params, batch, key):
        bs = batch["obs"].shape[0]
        k_prior, k_perm, k_mix = jax.random.split(key, 3)
        z = project_to_sphere(jax.random.normal(k_prior, (bs, self.cfg.z_dim)))
        phi = jax.lax.stop_gradient(self.phis(params, batch["obs"]))
        phi_perm = phi[jax.random.permutation(k_perm, bs)]
        if self.method == "hilp":
            # phi Cov^{-1} (HILP latent sampling); a regularised solve instead of pinv (much faster on GPU).
            cov = phi_perm.T @ phi_perm / bs + 1e-4 * jnp.eye(phi_perm.shape[-1])
            phi_perm = jnp.linalg.solve(cov, phi_perm.T).T
        z_phi = project_to_sphere(phi_perm)
        return jnp.where(jax.random.uniform(k_mix, (bs, 1)) < self.cfg.z_mix_ratio, z, z_phi)

    def intrinsic_reward(self, params, batch, z):
        phi_next = self.phis(params, batch["next_obs"])
        if self.method == "hilp":
            return ((phi_next - self.phis(params, batch["obs"])) * z).sum(-1)
        return (phi_next * z).sum(-1)

    # ------------------------------------------------------------------ update
    def update(self, state: State, batch, key):
        cfg = self.cfg
        z = self.sample_z(state.params, batch, key)
        r = jax.lax.stop_gradient(self.intrinsic_reward(state.params, batch, z))
        tq = self.q.apply(
            {"params": state.target_params["q"]}, encode_obs(batch["next_obs"], cfg.obs_encoding), z
        ).mean(0)  # (B, A)
        target = r + cfg.discount * tq.max(-1)

        def loss_fn(params):
            if self.method == "hilp":
                repr_loss, info = self.hilp_loss(params, state.target_params, batch)
            else:
                repr_loss, info = self.lap_loss(params, batch)
            qs = self.q.apply({"params": params["q"]}, encode_obs(batch["obs"], cfg.obs_encoding), z)
            qa = jnp.take_along_axis(qs, batch["actions"][None, :, None], axis=-1)[..., 0]
            q_loss = ((qa - target[None]) ** 2).mean()
            return repr_loss + q_loss, {**info, "repr/loss": repr_loss, "q/loss": q_loss, "q/mean": qa.mean()}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = self.tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        target_params = optax.incremental_update(params, state.target_params, cfg.tau)
        return State(params, target_params, opt_state, state.step + 1), info

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def train_chunk(self, state, arrays, bank, key, num_steps, batch_size):
        cfg = self.cfg
        num_layouts, n = arrays["actions"].shape
        L = self.episode_length

        def body(state, key):
            k_s, k_g, k_geo, k_mix, k_r, k_u = jax.random.split(key, 6)
            batch, layout, t = sample_transitions(arrays, k_s, batch_size)
            # Goals: a future state of the same episode (geometric offset) w.p. p_trajgoal, else a random state.
            ep_end = (t // L + 1) * L - 1
            u = jax.random.uniform(k_geo, (batch_size,), minval=1e-12)
            offset = jnp.floor(jnp.log(u) / jnp.log(cfg.discount)).astype(jnp.int32) + 1
            traj_goal = arrays["obs"][layout, jnp.minimum(t + offset, ep_end)]
            rand_goal = arrays["obs"][
                jax.random.randint(k_r, (batch_size,), 0, num_layouts), jax.random.randint(k_g, (batch_size,), 0, n)
            ]
            use_traj = jax.random.uniform(k_mix, (batch_size, 1)) < cfg.p_trajgoal
            goals = jnp.where(use_traj, traj_goal, rand_goal)
            success = (jnp.abs(batch["obs"] - goals).sum(-1) < 1e-6).astype(jnp.float32)
            batch.update(goals=goals, goal_rewards=success - 1.0, goal_masks=1.0 - success)
            return self.update(state, batch, k_u)

        state, infos = jax.lax.scan(body, state, jax.random.split(key, num_steps))
        return state, jax.tree.map(lambda x: x[-1], infos)

    # ------------------------------------------------------------------ inference
    @partial(jax.jit, static_argnums=0)
    def infer_z_goal(self, params, goal_obs, start_obs):
        if self.method == "hilp":
            return project_to_sphere(self.phis(params, goal_obs) - self.phis(params, start_obs))
        return project_to_sphere(self.phis(params, goal_obs))

    @partial(jax.jit, static_argnums=0)
    def q_values(self, params, obs, z, h=None):
        return self.q.apply({"params": params["q"]}, encode_obs(obs, self.cfg.obs_encoding), z).mean(0)

    @partial(jax.jit, static_argnums=0)
    def act(self, params, obs, z, h=None):
        return self.q_values(params, obs, z).argmax(-1)

    def save(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(jax.device_get({"params": self.state.params}), f)

    def load(self, path):
        import pickle

        with open(path, "rb") as f:
            params = jax.device_put(pickle.load(f)["params"])
        self.state = self.state.replace(params=params, target_params=params, opt_state=self.tx.init(params))
