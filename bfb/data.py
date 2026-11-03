"""Offline datasets for Randomized Four-Rooms (Appendix D.1, Table 4).

A dataset with N training layouts always contains the same number of transitions (1M by default): every layout gets
``total_transitions / (N * episode_length)`` random-walk episodes, i.e. 1000 / 500 / 333 / 200 episodes of length
100 for N = 10 / 20 / 30 / 50 (333 * 30 * 100 = 999,000 for N = 30).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from bfb.envs.fourrooms import GRID_SIZE, make_layout, random_free_cells, random_walk, train_layout_seeds


def episodes_per_layout(num_layouts: int, total_transitions: int, episode_length: int) -> int:
    if min(num_layouts, total_transitions, episode_length) <= 0:
        raise ValueError("dataset dimensions must be positive")
    if total_transitions < num_layouts * episode_length:
        raise ValueError("dataset needs at least one complete episode per layout")
    return total_transitions // (num_layouts * episode_length)


def generate_dataset(num_layouts: int, total_transitions: int = 1_000_000, episode_length: int = 100, seed: int = 0):
    """Uniform random exploration from uniformly random start cells, ``episodes_per_layout`` episodes per layout."""
    rng = np.random.default_rng(seed)
    seeds = train_layout_seeds(num_layouts)
    num_episodes = episodes_per_layout(num_layouts, total_transitions, episode_length)
    obs, actions, next_obs = [], [], []
    for s in seeds:
        grids = np.repeat(make_layout(s)[None], num_episodes, 0)
        walk = random_walk(grids, random_free_cells(grids, rng), episode_length, rng)
        obs.append(walk["observations"])
        actions.append(walk["actions"])
        next_obs.append(walk["next_observations"])
    return {
        "schema_version": np.array(1),
        "generation_seed": np.array(seed),
        "requested_transitions": np.array(total_transitions),
        "layout_seeds": seeds,
        "grids": np.stack([make_layout(s) for s in seeds]),
        "observations": np.stack(obs).astype(np.int8),  # (N, E, L, 2)
        "actions": np.stack(actions).astype(np.int8),  # (N, E, L)
        "next_observations": np.stack(next_obs).astype(np.int8),
    }


def load_or_generate(path: str, num_layouts: int, total_transitions: int, episode_length: int, seed: int = 0):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = generate_dataset(num_layouts, total_transitions, episode_length, seed)
        tmp = path + f".tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    expected = (num_layouts, episodes_per_layout(num_layouts, total_transitions, episode_length), episode_length)
    if data["actions"].shape != expected or not np.array_equal(data["layout_seeds"], train_layout_seeds(num_layouts)):
        raise ValueError(f"dataset cache {path} does not match the requested dimensions/layouts; use a new path")
    if "generation_seed" in data:
        if int(data["generation_seed"]) != seed or int(data["requested_transitions"]) != total_transitions:
            raise ValueError(f"dataset cache {path} was generated with different settings; use a new path")
    elif seed != 0:
        raise ValueError(f"legacy dataset {path} has no seed metadata; regenerate at a new path for nonzero seeds")
    return data


def normalize_obs(obs):
    """Map grid coordinates to [0, 1]."""
    return np.asarray(obs, dtype=np.float32) / (GRID_SIZE - 1)


class FourRoomsDataset:
    """Device-resident dataset. Sampling is done with jax inside the jitted training loop.

    Arrays are stored per layout as streams of transitions (episodes concatenated): shape (N, E * L, ...).
    """

    def __init__(self, data: dict):
        actions = np.asarray(data["actions"])
        self.num_layouts, self.num_episodes, self.episode_length = actions.shape
        n = self.num_episodes * self.episode_length
        self.size = self.num_layouts * n
        self.arrays = {
            "obs": jnp.asarray(normalize_obs(data["observations"]).reshape(self.num_layouts, n, -1)),
            "next_obs": jnp.asarray(normalize_obs(data["next_observations"]).reshape(self.num_layouts, n, -1)),
            "actions": jnp.asarray(actions.astype(np.int32).reshape(self.num_layouts, n)),
        }


def sample_transitions(arrays, key, batch_size: int):
    """Uniform transitions; also returns the (hidden) layout index and the stream index of each transition."""
    num_layouts, n = arrays["actions"].shape[:2]
    k_l, k_t = jax.random.split(key)
    layout = jax.random.randint(k_l, (batch_size,), 0, num_layouts)
    t = jax.random.randint(k_t, (batch_size,), 0, n)
    batch = {
        "obs": arrays["obs"][layout, t],
        "actions": arrays["actions"][layout, t],
        "next_obs": arrays["next_obs"][layout, t],
    }
    if "not_done" in arrays:
        batch["not_done"] = arrays["not_done"][layout, t]
    return batch, layout, t


def sample_contexts(arrays, key, layouts, context_len: int, episode_length: int):
    """Contiguous windows of ``context_len`` transitions from the given layouts.

    Windows start at an episode boundary when ``context_len`` is a multiple of the episode length (a 100-step context
    is exactly one exploration episode, as at test time) and anywhere in the layout's stream otherwise (contexts
    longer than one episode span consecutive episodes of the same layout).
    """
    n = arrays["actions"].shape[1]
    assert context_len <= n
    if context_len <= episode_length:
        # Never stitch the end of one episode to the random reset of another for short contexts.
        k_ep, k_start = jax.random.split(key)
        episode = jax.random.randint(k_ep, layouts.shape, 0, n // episode_length)
        offset = jax.random.randint(k_start, layouts.shape, 0, episode_length - context_len + 1)
        start = episode * episode_length + offset
    elif context_len % episode_length == 0:
        num_starts = (n - context_len) // episode_length + 1
        start = jax.random.randint(key, layouts.shape, 0, num_starts) * episode_length
    else:
        start = jax.random.randint(key, layouts.shape, 0, n - context_len + 1)
    idx = start[:, None] + jnp.arange(context_len)
    l = layouts[:, None]
    return {"obs": arrays["obs"][l, idx], "actions": arrays["actions"][l, idx], "next_obs": arrays["next_obs"][l, idx]}


def sample_queries(arrays, key, layouts, num_queries: int):
    """Uniformly random transitions from the given layouts (any episode), shape (B, num_queries, ...)."""
    n = arrays["actions"].shape[1]
    idx = jax.random.randint(key, (*layouts.shape, num_queries), 0, n)
    l = layouts[:, None]
    return {"obs": arrays["obs"][l, idx], "actions": arrays["actions"][l, idx], "next_obs": arrays["next_obs"][l, idx]}
