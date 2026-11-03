"""Networks: forward map F, backward map B, permutation-invariant dynamics encoder f_dyn and next-state predictor."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def encode_obs(obs, encoding: str, grid_size: int = 11):
    """Input features of a normalised grid position: ``xy`` (the 2 coordinates) or ``onehot`` (one-hot row and
    column, 2 * grid_size features), which gives the networks cell-level resolution."""
    if encoding == "xy":
        return obs
    if encoding != "onehot":
        raise ValueError(f"unknown observation encoding: {encoding}")
    idx = jnp.round(obs * (grid_size - 1)).astype(jnp.int32)
    return jnp.concatenate([jax.nn.one_hot(idx[..., 0], grid_size), jax.nn.one_hot(idx[..., 1], grid_size)], -1)


def project_to_sphere(x, eps: float = 1e-8):
    """Normalise the last axis to norm sqrt(dim) (the convention used for z and B in FB)."""
    return x * jnp.sqrt(x.shape[-1]) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


class MLP(nn.Module):
    """MLP following Touati et al. (2023): the first layer uses LayerNorm + tanh, the rest GELU (+ LayerNorm)."""

    hidden_dims: Sequence[int]
    out_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=nn.initializers.orthogonal())(x)
            if i == 0:
                x = nn.tanh(nn.LayerNorm()(x))
            else:
                x = nn.gelu(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.orthogonal())(x)


class ForwardDiscrete(nn.Module):
    """F(s, ., z [, h]) for discrete actions: outputs one d-dim vector per action, for an ensemble of 2 networks."""

    z_dim: int
    num_actions: int
    hidden_dims: Sequence[int]
    preprocessor_dims: Sequence[int] = ()
    embed_dim: int = 512

    @nn.compact
    def __call__(self, obs, z, h=None):
        # The task input is the augmented vector [h; z] (Eq. 6).
        task = z if h is None else jnp.concatenate([h, z], -1)
        if self.preprocessor_dims:
            # Appendix E.1.2: s and (s, [h; z]) get their own preprocessors embedding them into 512-d spaces.
            # As in metamotivo, the state embedding sees only the state; the context enters with the task.
            emb_s = MLP(self.preprocessor_dims, self.embed_dim)(obs)
            emb_sz = MLP(self.preprocessor_dims, self.embed_dim)(jnp.concatenate([obs, task], -1))
            x = jnp.concatenate([emb_s, emb_sz], -1)
        else:
            x = jnp.concatenate([obs, task], -1)
        ensemble = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        out = ensemble(self.hidden_dims, self.z_dim * self.num_actions)(x)
        return out.reshape(2, *obs.shape[:-1], self.num_actions, self.z_dim)  # (2, B, A, d)


class Backward(nn.Module):
    """B(s). Normalisation (optional) is applied by the agent."""

    z_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden_dims, self.z_dim)(obs)


class EncoderBlock(nn.Module):
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(x.shape[-1])(nn.gelu(nn.Dense(self.mlp_dim)(y)))
        return x + y


class DynamicsEncoder(nn.Module):
    """f_dyn: a set of transitions {(s, a, s')} -> Gaussian belief N(mu, sigma) over the context h.

    No positional embeddings and mean pooling, so the output is invariant to the order of transitions.
    """

    h_dim: int
    emb_dim: int = 96
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 256
    delta_scale: float | None = None

    @nn.compact
    def __call__(self, obs, actions_onehot, next_obs):
        outcome = next_obs if self.delta_scale is None else (next_obs - obs) / self.delta_scale
        x = jnp.concatenate([obs, actions_onehot, outcome], -1)
        x = nn.Dense(self.emb_dim)(x)
        for _ in range(self.num_layers):
            x = EncoderBlock(self.num_heads, self.mlp_dim)(x)
        x = nn.LayerNorm()(x).mean(-2)
        mu = nn.Dense(self.h_dim)(x)
        log_std = jnp.clip(nn.Dense(self.h_dim)(x), -10.0, 2.0)
        return mu, log_std


class NextStatePredictor(nn.Module):
    """g_pred(s, a, h) -> s'."""

    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, obs, actions_onehot, h):
        x = jnp.concatenate([obs, actions_onehot, h], -1)
        for size in self.hidden_dims:
            x = nn.gelu(nn.Dense(size)(x))
        return nn.Dense(self.out_dim)(x)
