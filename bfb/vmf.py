"""von Mises-Fisher sampling (Wood, 1994) with a Householder rotation from the north pole (Algorithm 2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _sample_w(key, kappa: float, dim: int, shape, num_proposals: int = 16):
    """Sample the component along the mean direction with Wood's rejection sampler.

    ``num_proposals`` proposals are drawn in parallel and the first accepted one is kept; the acceptance rate is
    high (> 0.5 for all kappa, dim), so the probability that none is accepted is negligible. In that case the last
    proposal is used.
    """
    d1 = dim - 1.0
    b = d1 / (2.0 * kappa + jnp.sqrt(4.0 * kappa**2 + d1**2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + d1 * jnp.log(1.0 - x0**2)

    k_beta, k_unif = jax.random.split(key)
    eps = jax.random.beta(k_beta, d1 / 2.0, d1 / 2.0, (num_proposals, *shape))
    w = (1.0 - (1.0 + b) * eps) / (1.0 - (1.0 - b) * eps)
    u = jax.random.uniform(k_unif, (num_proposals, *shape), minval=1e-12)
    accept = kappa * w + d1 * jnp.log(1.0 - x0 * w) - c >= jnp.log(u)
    first = jnp.argmax(accept, axis=0)  # index of first accepted proposal (0 if none accepted)
    first = jnp.where(accept.any(0), first, num_proposals - 1)
    return jnp.take_along_axis(w, first[None], axis=0)[0]


def householder_from_north_pole(x, mu):
    """Apply the reflection that maps e_1 to mu (unit vectors) to each row of x."""
    e1 = jnp.zeros_like(mu).at[..., 0].set(1.0)
    u = e1 - mu
    norm2 = (u**2).sum(-1, keepdims=True)
    reflected = x - 2.0 * u * (u * x).sum(-1, keepdims=True) / jnp.maximum(norm2, 1e-12)
    return jnp.where(norm2 < 1e-12, x, reflected)


def w_quantile_table(kappa: float, dim: int, num_quantiles: int = 4096, grid_size: int = 200_001) -> np.ndarray:
    """Inverse CDF of w = <x, mu> for x ~ vMF(mu, kappa) on S^{dim-1}, density ~ exp(kappa w) (1 - w^2)^((dim-3)/2).

    Computed once by numerical integration; sampling w then costs one interpolation (much faster on GPU than the
    Beta draws needed by Wood's rejection sampler, which it matches in distribution up to discretisation).
    """
    w = np.linspace(-1.0, 1.0, grid_size)[1:-1]
    log_density = kappa * w + 0.5 * (dim - 3) * np.log1p(-(w**2))
    density = np.exp(log_density - log_density.max())
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    qs = (np.arange(num_quantiles) + 0.5) / num_quantiles
    return np.interp(qs, cdf, w).astype(np.float32)


def sample_vmf(key, mean_direction, kappa: float, w_table=None):
    """One unit-norm vMF(mu, kappa) sample per row of ``mean_direction`` (which is normalised first).

    ``w_table`` (from ``w_quantile_table``) replaces Wood's rejection sampler for the component along mu.
    """
    mu = mean_direction / (jnp.linalg.norm(mean_direction, axis=-1, keepdims=True) + 1e-8)
    batch, dim = mu.shape
    k_w, k_v = jax.random.split(key)
    if w_table is None:
        w = _sample_w(k_w, kappa, dim, (batch,))[:, None]
    else:
        w_table = jnp.asarray(w_table)
        n = w_table.shape[0]
        u = jax.random.uniform(k_w, (batch,)) * (n - 1)
        lo = jnp.floor(u).astype(jnp.int32)
        frac = u - lo
        w = (w_table[lo] * (1 - frac) + w_table[jnp.minimum(lo + 1, n - 1)] * frac)[:, None]
    v = jax.random.normal(k_v, (batch, dim - 1))
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    north = jnp.concatenate([w, jnp.sqrt(jnp.clip(1.0 - w**2, 0.0)) * v], -1)
    return householder_from_north_pole(north, mu)
