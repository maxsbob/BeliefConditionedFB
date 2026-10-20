"""Forward-Backward agents for discrete actions: vanilla FB, Belief-FB (BFB) and Rotation-FB (RFB).

All three share one implementation and differ only in two switches:

* ``use_context``: F is conditioned on the belief h = f_dyn(context) (BFB, RFB). B is never conditioned (Sec. 3.2).
* ``z_sampling``: ``uniform`` samples task vectors uniformly on the sphere (FB, BFB); ``vmf`` samples them from a
  von Mises-Fisher distribution centred at the inferred context direction h (RFB, Sec. 3.3 / Algorithm 2).
  In both cases a fraction ``1 - z_mix_ratio`` of the batch uses backward-induced task vectors z = B(s).

Training is two-stage (Algorithm 1): f_dyn is first pre-trained with a next-state prediction objective, then frozen
while F and B are trained with the FB loss (Eq. 13).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from bfb.data import sample_contexts, sample_queries, sample_transitions
from bfb.networks import encode_obs, Backward, DynamicsEncoder, ForwardDiscrete, NextStatePredictor, project_to_sphere
from bfb.vmf import householder_from_north_pole, sample_vmf, w_quantile_table


@flax.struct.dataclass
class State:
    params: Any
    target_params: Any
    opt_state: Any
    step: jnp.ndarray


class FBAgent:
    def __init__(self, cfg, obs_dim: int, num_actions: int, seed: int = 0):
        self.cfg = cfg
        self.obs_dim, self.num_actions = obs_dim, num_actions
        self.use_context = bool(cfg.use_context)
        self.h_stats = None  # set after encoder pre-training (see belief_stats)
        # Inverse-CDF tables for fast vMF sampling (see bfb.vmf.w_quantile_table).
        self.z_w_table = w_quantile_table(cfg.kappa, cfg.z_dim) if cfg.z_sampling == 'vmf' else None
        self.task_w_table = w_quantile_table(cfg.task_kappa, cfg.z_dim) if cfg.task_kappa is not None else None
        self.h_w_table = (w_quantile_table(cfg.belief_kappa, cfg.h_dim)
                          if self.use_context and cfg.belief_kappa is not None else None)
        h_dim = cfg.h_dim if self.use_context else None
        if cfg.z_sampling == 'vmf' or cfg.cone_kappa is not None:
            assert self.use_context and cfg.h_dim == cfg.z_dim, 'RFB needs a context encoder with h_dim == z_dim.'
        self.cone_w_min = None
        if cfg.cone_kappa is not None:
            table = w_quantile_table(cfg.cone_kappa, cfg.z_dim, num_quantiles=1000)
            self.cone_w_min = float(table[20])  # 2% quantile

        self.forward = ForwardDiscrete(
            cfg.z_dim, num_actions, tuple(cfg.f_hidden_dims), tuple(cfg.f_preprocessor_dims), cfg.f_embed_dim
        )
        self.backward = Backward(cfg.z_dim, tuple(cfg.b_hidden_dims))  # unnormalised output
        self.encoder = DynamicsEncoder(
            cfg.h_dim, cfg.encoder.emb_dim, cfg.encoder.num_layers, cfg.encoder.num_heads, cfg.encoder.mlp_dim
        )
        self.predictor = NextStatePredictor(
            tuple(cfg.encoder.predictor_hidden_dims), obs_dim if cfg.encoder.objective == 'mse' else 1
        )

        key = jax.random.key(seed)
        kf, kb, ke, kp = jax.random.split(key, 4)
        obs = jnp.zeros((1, obs_dim))
        z = jnp.zeros((1, cfg.z_dim))
        h = jnp.zeros((1, h_dim)) if self.use_context else None
        obs_f = encode_obs(obs, cfg.obs_encoding)
        fb_params = {
            'forward': self.forward.init(kf, obs_f, z, None if cfg.cone_drop_h else h)['params'],
            'backward': self.backward.init(kb, obs_f)['params'],
        }
        self.fb_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))
        self.state = State(fb_params, fb_params, self.fb_tx.init(fb_params), jnp.zeros((), jnp.int32))

        if self.use_context:
            a = jnp.zeros((1, 1, num_actions))
            enc_params = {
                'encoder': self.encoder.init(ke, obs[None], a, obs[None])['params'],
                'predictor': self.predictor.init(kp, obs[None], a, jnp.zeros((1, 1, h_dim)))['params'],
            }
            self.enc_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.encoder.lr))
            self.enc_state = State(enc_params, None, self.enc_tx.init(enc_params), jnp.zeros((), jnp.int32))
        else:
            self.enc_state = None

    # ------------------------------------------------------------------ context encoder
    def onehot(self, actions):
        return jax.nn.one_hot(actions, self.num_actions)

    @partial(jax.jit, static_argnums=0)
    def encode(self, enc_params, obs, actions, next_obs):
        """Belief N(mu, sigma) over h from a batch of contexts of shape (B, T, ...)."""
        return self.encoder.apply({'params': enc_params['encoder']}, obs, self.onehot(actions), next_obs)

    def update_encoder(self, state: State, batch, key):
        """Algorithm 1, lines 4-10, with a reparametrised belief h shared over the sequence.

        The belief is inferred from the context transitions (``ctx_*``) and used to predict the outcome of *query*
        transitions (``q_*``). With ``encoder.query = context`` the queries are the context transitions themselves
        (Algorithm 1 as written); with ``heldout`` they are other transitions from the same layout, which forces h to
        summarise the layout instead of the particular trajectory.

        ``encoder.objective``: ``mse`` regresses s' (paper); ``blocked`` classifies whether the move was blocked by a
        wall, the only layout-dependent part of the dynamics (s' is a deterministic function of s, a and this bit).
        """
        enc_cfg = self.cfg.encoder

        def loss_fn(params):
            mu, log_std = self.encoder.apply(
                {'params': params['encoder']}, batch['ctx_obs'], self.onehot(batch['ctx_actions']), batch['ctx_next_obs']
            )
            h = mu + jax.random.normal(key, mu.shape) * jnp.exp(log_std)
            q_obs, q_act, q_next = batch['q_obs'], self.onehot(batch['q_actions']), batch['q_next_obs']
            h_seq = jnp.broadcast_to(h[:, None], (*q_obs.shape[:2], h.shape[-1]))
            pred = self.predictor.apply({'params': params['predictor']}, q_obs, q_act, h_seq)
            blocked = (jnp.abs(q_next - q_obs).sum(-1) < 1e-6).astype(jnp.float32)
            if enc_cfg.objective == 'mse':
                recon = ((pred - q_next) ** 2).sum(-1).mean()
            else:
                recon = optax.sigmoid_binary_cross_entropy(pred[..., 0], blocked).mean()
            accuracy = ((pred[..., 0] > 0) == (blocked > 0)).mean()
            # Gaussian prior on h (KL to N(0, I)).
            kl = 0.5 * (mu**2 + jnp.exp(2 * log_std) - 1.0 - 2 * log_std).sum(-1).mean()
            loss = recon + enc_cfg.kl_coef * kl
            return loss, {
                'encoder/recon': recon,
                'encoder/kl': kl,
                'encoder/std': jnp.exp(log_std).mean(),
                'encoder/blocked_acc': accuracy,
            }

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = self.enc_tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return state.replace(params=params, opt_state=opt_state, step=state.step + 1), info

    # ------------------------------------------------------------------ FB
    def F(self, params, obs, z, h):
        """Successor features F(s, ., z_pol [, h]) where z is the task (reward) vector and z_pol the policy code."""
        obs = encode_obs(obs, self.cfg.obs_encoding)
        z_pol = self.policy_code(z, h) if self.cfg.cone_kappa is not None else z
        h_in = h if (self.use_context and not self.cfg.cone_drop_h) else None
        return self.forward.apply({'params': params['forward']}, obs, z_pol, h_in)

    def policy_code(self, z, h):
        """Cone embedding (RFB): map the task vector z into the spherical cap around the belief direction h.

        z (unit) is written in polar form around the north pole e1; its polar coordinate w = <z, e1> in [-1, 1] is
        compressed affinely to [w_min, 1], where w_min is the 2% quantile of <x, e1> under vMF(e1, cone_kappa). The cap
        is then rotated onto h with the Householder reflection e1 -> h (Algorithm 2). The map is a bijection between
        the sphere and the cap, so every task keeps its own policy, but the policies of a context live in its own cone
        of policy space (block-separable parametrisation, Theorem 2); cone_kappa sets the cone width, i.e. the
        diversity of behaviours per context.
        """
        u = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
        u1, rest = u[..., :1], u[..., 1:]
        w = 1.0 - (1.0 - u1) * (1.0 - self.cone_w_min) / 2.0
        scale = jnp.sqrt(jnp.clip(1.0 - w**2, 0.0)) / jnp.sqrt(jnp.clip(1.0 - u1**2, 1e-12))
        north = jnp.concatenate([w, rest * scale], -1)
        h_dir = h / jnp.linalg.norm(h, axis=-1, keepdims=True)
        return project_to_sphere(householder_from_north_pole(north, h_dir))

    def B(self, params, obs):
        b = self.backward.apply({'params': params['backward']}, encode_obs(obs, self.cfg.obs_encoding))
        return project_to_sphere(b) if self.cfg.normalize_b else b

    def sample_z(self, params, batch, key):
        """Mixture of 'prior' task vectors (uniform or vMF around h) and backward-induced ones z = B(s)."""
        bs = batch['obs'].shape[0]
        k_prior, k_mix, k_perm, k_task = jax.random.split(key, 4)
        if self.cfg.z_sampling == 'uniform':
            z_prior = project_to_sphere(jax.random.normal(k_prior, (bs, self.cfg.z_dim)))
        else:
            z_prior = project_to_sphere(sample_vmf(k_prior, batch['h'], self.cfg.kappa, self.z_w_table))
        goals = batch['next_obs'][jax.random.permutation(k_perm, bs)]
        z_b = project_to_sphere(jax.lax.stop_gradient(self.B(params, goals)))
        if self.cfg.task_kappa is not None:
            # Task rotation: train on codes rotated around the goal codes, z ~ vMF(B(g), task_kappa), which is the
            # distribution the agent acts with at test time (eval.task_kappa).
            z_b = project_to_sphere(sample_vmf(k_task, z_b, self.cfg.task_kappa, self.task_w_table))
        use_prior = jax.random.uniform(k_mix, (bs, 1)) < self.cfg.z_mix_ratio
        return jnp.where(use_prior, z_prior, z_b)

    def update_fb(self, state: State, batch, key):
        """FB update with the TD least-squares importance fitting (TD-LSIF) loss.

        Following onestep-fb (https://github.com/chongyi-zheng/onestep-fb), the diagonal term is weighted by
        (1 - discount), i.e. F^T B approximates the *normalised* successor measure (1 - discount) M / rho. This keeps
        F^T B at the scale of 1 / rho instead of 1 / ((1 - discount) rho) and is what makes greedy targets stable here.
        ``fb_loss: original`` gives the unnormalised loss of Eq. (13).
        """
        cfg = self.cfg
        z = self.sample_z(state.params, batch, key)
        h = batch.get('h')
        bs = z.shape[0]
        agg = (lambda x: x.mean(0)) if cfg.target_agg == 'mean' else (lambda x: x.min(0))

        # Target successor features at s' for the target policy pi_z (greedy w.r.t. Q = F^T z, Eq. 10).
        tF = self.F(state.target_params, batch['next_obs'], z, h)  # (2, B, A, d)
        tQ = agg(jnp.einsum('ebad,bd->eba', tF, z))  # (B, A)
        if cfg.boltzmann_temp is None:
            pi = jax.nn.one_hot(tQ.argmax(-1), self.num_actions)
        else:
            pi = jax.nn.softmax(tQ / cfg.boltzmann_temp, -1)
        tF = jnp.einsum('ba,ebad->ebd', pi, tF)  # (2, B, d)
        tB = self.B(state.target_params, batch['next_obs'])
        target_M = agg(jnp.einsum('ebd,cd->ebc', tF, tB))  # (B, B)

        off = 1.0 - jnp.eye(bs)
        diag_weight = (1.0 - cfg.discount) if cfg.fb_loss == 'lsif' else 1.0

        def loss_fn(params):
            F = self.F(params, batch['obs'], z, h)
            F = jnp.take_along_axis(F, batch['actions'][None, :, None, None], axis=2)[:, :, 0]  # (2, B, d)
            Bn = self.B(params, batch['next_obs'])
            M = jnp.einsum('ebd,cd->ebc', F, Bn)  # (2, B, B)
            diff = M - cfg.discount * target_M[None]
            fb_offdiag = (0.5 * ((diff * off) ** 2).sum(-1) / (bs - 1)).mean()
            fb_diag = -diag_weight * jnp.diagonal(M, axis1=1, axis2=2).mean()
            fb_loss = fb_offdiag + fb_diag
            cov = Bn @ Bn.T
            ortho = -jnp.diag(cov).mean() + (0.5 * ((cov * off) ** 2).sum(-1) / (bs - 1)).mean()
            loss = fb_loss + cfg.ortho_coef * ortho
            Q = jnp.einsum('ebd,bd->eb', F, z)
            return loss, {
                'fb/loss': fb_loss,
                'fb/diag': fb_diag,
                'fb/offdiag': fb_offdiag,
                'fb/ortho': ortho,
                'fb/Q': Q.mean(),
                'fb/M_diag': jnp.diagonal(M, axis1=1, axis2=2).mean(),
                'fb/M_offdiag': (M * off).sum((1, 2)).mean() / (bs * (bs - 1)),
            }

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = self.fb_tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        target = optax.incremental_update(params, state.target_params, cfg.tau)
        return State(params, target, opt_state, state.step + 1), info

    # ------------------------------------------------------------------ fused training loops (lax.scan on device)
    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7))
    def train_encoder_chunk(self, state, arrays, key, num_steps, batch_size, context_len, episode_length):
        num_layouts = arrays['actions'].shape[0]

        def body(state, key):
            k_l, k_c, k_u = jax.random.split(key, 3)
            layouts = jax.random.randint(k_l, (batch_size,), 0, num_layouts)
            k_c, k_q = jax.random.split(k_c)
            ctx = sample_contexts(arrays, k_c, layouts, context_len, episode_length)
            batch = {f'ctx_{k}': v for k, v in ctx.items()}
            if self.cfg.encoder.query == 'heldout':
                q = sample_queries(arrays, k_q, layouts, self.cfg.encoder.num_queries)
            else:
                q = ctx
            batch.update({f'q_{k}': v for k, v in q.items()})
            return self.update_encoder(state, batch, k_u)

        state, infos = jax.lax.scan(body, state, jax.random.split(key, num_steps))
        return state, jax.tree.map(lambda x: x[-1], infos)

    @partial(jax.jit, static_argnums=(0, 4, 5, 6))
    def build_belief_bank(self, enc_params, arrays, key, bank_size, context_len, episode_length):
        """Beliefs (mu, log_std) of the frozen encoder for ``bank_size`` random contexts of every training layout.

        Drawing h from this bank during FB training is equivalent to encoding a freshly sampled context of the
        transition's layout at every step (the encoder is frozen), but avoids running the transformer on every update.
        """
        num_layouts = arrays['actions'].shape[0]

        def per_layout(layout, key):
            ctx = sample_contexts(arrays, key, jnp.full((bank_size,), layout), context_len, episode_length)
            return self.encoder.apply({'params': enc_params['encoder']}, ctx['obs'], self.onehot(ctx['actions']),
                                      ctx['next_obs'])

        return jax.lax.map(lambda x: per_layout(*x), (jnp.arange(num_layouts), jax.random.split(key, num_layouts)))

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def train_chunk(self, state, arrays, bank, key, num_steps, batch_size):
        """``num_steps`` FB updates.

        ``bank`` = (mu, log_std, h_stats) with beliefs of shape (N, K, h_dim) for the training layouts, or None for
        vanilla FB.
        """

        def body(state, key):
            k_s, k_b, k_n, k_r, k_u, k_g = jax.random.split(key, 6)
            batch, layout, _ = sample_transitions(arrays, k_s, batch_size)
            if self.use_context:
                mu_bank, log_std_bank, h_stats = bank
                j = jax.random.randint(k_b, (batch_size,), 0, mu_bank.shape[1])
                mu, std = mu_bank[layout, j], jnp.exp(log_std_bank[layout, j])
                h = self.process_h(mu + std * jax.random.normal(k_n, mu.shape), h_stats)  # reparametrised sample
                if self.cfg.belief_kappa is not None:
                    # Belief rotation: rotate the context direction by a random vMF(h, belief_kappa) rotation.
                    h = project_to_sphere(sample_vmf(k_r, h, self.cfg.belief_kappa, self.h_w_table))
                if self.cfg.geodesic_kappa is not None:
                    h = self.geodesic_rotation(k_g, h, mu_bank, h_stats)
                batch['h'] = h
            return self.update_fb(state, batch, k_u)

        state, infos = jax.lax.scan(body, state, jax.random.split(key, num_steps))
        return state, jax.tree.map(lambda x: x[-1], infos)

    def geodesic_rotation(self, key, h, mu_bank, h_stats):
        """Rotate each belief along the great circle towards the belief of a random other context.

        The rotation angle is t * angle(h, h_other) with t ~ Beta(1, geodesic_kappa) restricted to [0, 1/2], so
        kappa -> inf recovers BFB and small kappa makes F share behaviour between neighbouring contexts. Unlike
        isotropic vMF noise, the perturbations stay in the region of belief space occupied by real contexts, which is
        where beliefs of unseen dynamics fall.
        """
        bs = h.shape[0]
        k_l, k_j, k_t = jax.random.split(key, 3)
        other = mu_bank[jax.random.randint(k_l, (bs,), 0, mu_bank.shape[0]),
                        jax.random.randint(k_j, (bs,), 0, mu_bank.shape[1])]
        u = h / jnp.linalg.norm(h, axis=-1, keepdims=True)
        v = self.process_h(other, h_stats)
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        t = 0.5 * jax.random.beta(k_t, 1.0, self.cfg.geodesic_kappa, (bs, 1))
        omega = jnp.arccos(jnp.clip((u * v).sum(-1, keepdims=True), -1 + 1e-6, 1 - 1e-6))
        rotated = (jnp.sin((1 - t) * omega) * u + jnp.sin(t * omega) * v) / jnp.sin(omega)
        return project_to_sphere(rotated)

    # ------------------------------------------------------------------ belief pre-processing
    @staticmethod
    def belief_stats(bank_mu):
        """Per-dimension mean / std of the training beliefs (bank of shape (N, K, h_dim))."""
        flat = bank_mu.reshape(-1, bank_mu.shape[-1])
        return {'mean': flat.mean(0), 'std': flat.std(0) + 1e-6}

    def process_h(self, h, h_stats):
        """Belief fed to F. With ``standardize_h`` the belief is standardised with training statistics and projected
        to the sphere of radius sqrt(h_dim), so that its *direction* encodes the context (and can be rotated)."""
        if not self.cfg.standardize_h or h_stats is None:
            return h
        return project_to_sphere((h - h_stats['mean']) / h_stats['std'])

    # ------------------------------------------------------------------ inference
    @partial(jax.jit, static_argnums=0)
    def infer_z_goal(self, params, goal_obs, start_obs=None):
        """Goal-reaching task r(s) = 1{s = g}: z_r = B(g) (Eq. 9)."""
        return project_to_sphere(self.B(params, goal_obs))

    @partial(jax.jit, static_argnums=0)
    def q_values(self, params, obs, z, h=None):
        return jnp.einsum('ebad,bd->eba', self.F(params, obs, z, h), z).min(0)

    @partial(jax.jit, static_argnums=(0, 3))
    def perturb_task(self, key, z, kappa):
        """Task code z' ~ vMF(z, kappa), projected to the sphere (stochastic but goal-directed behaviour)."""
        return project_to_sphere(sample_vmf(key, z, kappa, jnp.asarray(w_quantile_table(kappa, z.shape[-1]))))

    @partial(jax.jit, static_argnums=(0, 6, 7))
    def act_marginal(self, params, obs, z, h, key, num_samples, kappa):
        """Greedy action w.r.t. Q averaged over beliefs h' ~ vMF(h, kappa) (Bayesian marginalisation of the belief)."""
        table = jnp.asarray(w_quantile_table(kappa, h.shape[-1]))

        def q_for(k):
            h_k = project_to_sphere(sample_vmf(k, h, kappa, table))
            return self.q_values(params, obs, z, h_k)

        q = jax.vmap(q_for)(jax.random.split(key, num_samples)).mean(0)
        return q.argmax(-1)

    @partial(jax.jit, static_argnums=0)
    def act(self, params, obs, z, h=None):
        return self.q_values(params, obs, z, h).argmax(-1)

    def save(self, path):
        import pickle

        with open(path, 'wb') as f:
            pickle.dump(jax.device_get({
                'fb': self.state.params,
                'enc': None if self.enc_state is None else self.enc_state.params,
                'h_stats': self.h_stats,
            }), f)

    def load(self, path):
        import pickle

        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        self.state = self.state.replace(params=jax.device_put(ckpt['fb']))
        if ckpt['enc'] is not None:
            self.enc_state = self.enc_state.replace(params=jax.device_put(ckpt['enc']))
        self.h_stats = None if ckpt['h_stats'] is None else jax.device_put(ckpt['h_stats'])
