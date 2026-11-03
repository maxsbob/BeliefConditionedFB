"""Diagnostics for the context encoder: how well does the belief h identify the (hidden) layout?"""

from __future__ import annotations

import numpy as np

from bfb.data import normalize_obs
from bfb.envs.fourrooms import START_POS, make_layout, random_walk


def belief_means(agent, layout_seeds, context_len: int, num_contexts: int, seed: int = 0):
    """Belief means h for ``num_contexts`` exploration episodes (from the start cell) in every layout."""
    rng = np.random.default_rng(seed)
    grids = np.repeat(np.stack([make_layout(s) for s in layout_seeds]), num_contexts, 0)
    ctx = random_walk(grids, np.tile(np.array(START_POS), (len(grids), 1)), context_len, rng)
    mu, _ = agent.encode(
        agent.enc_state.params,
        normalize_obs(ctx["observations"]),
        ctx["actions"],
        normalize_obs(ctx["next_observations"]),
    )
    return np.asarray(mu).reshape(len(layout_seeds), num_contexts, -1)


def layout_identification_accuracy(h: np.ndarray) -> float:
    """Leave-one-out nearest-centroid accuracy of predicting the layout from h (array of shape (layouts, contexts, d))."""
    n_layouts, n_ctx, _ = h.shape
    correct = 0
    for i in range(n_layouts):
        for j in range(n_ctx):
            centroids = h.mean(1).copy()
            centroids[i] = np.delete(h[i], j, 0).mean(0)
            pred = np.argmin(((centroids - h[i, j]) ** 2).sum(-1))
            correct += pred == i
    return correct / (n_layouts * n_ctx)
