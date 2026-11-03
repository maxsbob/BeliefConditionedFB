"""Official AntWind data and evaluation. Training uses contexts 0..15; test uses 16..19.

The public NPZ's unused `info` object arrays are never unpickled. Supplied episode IDs are kept intact. Context
comes from reward-free slices of the supplied expert trajectories; this is an explicitly stronger context
collection policy than random exploration, and is reported as such in results.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from bfb.envs.antwind import make_antwind
from bfb.experiment import file_sha256
from bfb.networks import project_to_sphere


def read_context(path, context_id):
    with np.load(Path(path) / f"default_ind_{context_id}.npz", allow_pickle=False) as data:
        return {k: data[k].astype(np.float32) for k in ("observation", "next_observation", "actions", "reward", "done")}


def load_antwind(path, num_train_contexts=16, episodes=200):
    if num_train_contexts != 16:
        raise ValueError("the declared AntWind split uses exactly 16 training contexts")
    data = [read_context(path, i) for i in range(num_train_contexts)]
    obs = np.stack([d["observation"][:episodes] for d in data])
    nxt = np.stack([d["next_observation"][:episodes] for d in data])
    mean = obs.mean((0, 1, 2))
    std = np.maximum(obs.std((0, 1, 2)), 0.1)
    normalize = lambda x: np.clip((x - mean) / std, -10, 10).astype(np.float32)
    arrays = {
        "obs": normalize(obs).reshape(num_train_contexts, -1, obs.shape[-1]),
        "next_obs": normalize(nxt).reshape(num_train_contexts, -1, obs.shape[-1]),
        "actions": np.stack([d["actions"][:episodes] for d in data]).reshape(num_train_contexts, -1, 8),
    }
    # Time limits are not physical terminals. Ant termination is defined by torso height and finite state.
    healthy = (nxt[..., 0] >= 0.2) & (nxt[..., 0] <= 1.0) & np.isfinite(nxt).all(-1)
    arrays["not_done"] = healthy.reshape(num_train_contexts, -1).astype(np.float32)
    rewards = np.stack([d["reward"][:episodes] for d in data]).reshape(-1)
    labels = {"obs": arrays["next_obs"].reshape(-1, obs.shape[-1]), "rewards": rewards}
    hashes = {str(i): file_sha256(Path(path) / f"default_ind_{i}.npz") for i in range(20)}
    return arrays, {"mean": mean, "std": std}, labels, hashes


def evaluate_antwind(
    agent,
    path,
    normalization,
    labels,
    split="test",
    context_len=100,
    num_contexts=5,
    max_steps=256,
    seed=20261103,
    online_belief=False,
    context_mode="correct",
    policy="agent",
    frame_callback=None,
    render_kwargs=None,
):
    """Optionally call frame_callback(step, envs, returns, active) for rendering, without changing policy inputs.

    The callback may render environments but must not step them or alter physics. Render options are ignored
    unless a callback is supplied. Returns and activity masks passed to the callback are copies.
    """
    if split not in ("train", "test") or policy not in ("agent", "random", "zero"):
        raise ValueError("invalid AntWind split/policy")
    if context_mode not in ("correct", "shuffled") or not 1 <= context_len <= 256:
        raise ValueError("invalid AntWind context mode/length")
    ids = list(range(16)) if split == "train" else list(range(16, 20))
    rng = np.random.default_rng(seed)
    mean, std = normalization["mean"], normalization["std"]
    normalize = lambda x: np.clip((x - mean) / std, -10, 10).astype(np.float32)
    histories = {k: [] for k in ("obs", "actions", "next_obs")}
    for context_id in ids:
        d = read_context(path, context_id)
        episode_ids = rng.integers(len(d["observation"]), size=num_contexts)
        for key, name in [("obs", "observation"), ("actions", "actions"), ("next_obs", "next_observation")]:
            value = d[name][episode_ids, :context_len]
            histories[key].append(value if key == "actions" else normalize(value))
    histories = {k: np.concatenate(v) for k, v in histories.items()}
    batch_size = len(ids) * num_contexts
    permutation = np.roll(np.arange(batch_size).reshape(len(ids), num_contexts), 1, axis=0).ravel()
    h = None

    def infer_h():
        if agent is None or not agent.use_context:
            return None
        mu, _ = agent.encode(
            agent.enc_state.params, *[jnp.asarray(histories[k]) for k in ("obs", "actions", "next_obs")]
        )
        value = agent.process_h(mu, agent.h_stats)
        return value[permutation] if context_mode == "shuffled" else value

    h = infer_h()
    if policy == "agent":
        indices = np.random.default_rng(seed + 1).choice(len(labels["rewards"]), size=10000, replace=False)
        b = agent.B(agent.state.params, jnp.asarray(labels["obs"][indices]))
        z = project_to_sphere((b * jnp.asarray(labels["rewards"][indices, None])).mean(0, keepdims=True))
        z = jnp.repeat(z, batch_size, 0)
    options = {"render_mode": None}
    if frame_callback is not None:
        options = {"render_mode": "rgb_array", **(render_kwargs or {})}
    envs = [make_antwind(i, **options) for i in ids for _ in range(num_contexts)]
    returns = np.zeros(batch_size)
    active = np.ones(batch_size, dtype=bool)
    lengths = np.zeros(batch_size, dtype=np.int32)
    try:
        obs = np.stack([env.reset(seed=seed + j % num_contexts)[0] for j, env in enumerate(envs)])
        if frame_callback is not None:
            frame_callback(0, envs, returns.copy(), active.copy())
        for step in range(max_steps):
            if policy == "random":
                actions = rng.uniform(-1, 1, (batch_size, 8))
            elif policy == "zero":
                actions = np.zeros((batch_size, 8))
            else:
                actions = np.asarray(agent.act(agent.state.params, jnp.asarray(normalize(obs)), z, h))
            nxt = obs.copy()
            for j, env in enumerate(envs):
                if active[j]:
                    nxt[j], reward, terminated, truncated, _ = env.step(actions[j])
                    returns[j] += reward
                    lengths[j] += 1
                    active[j] = not (terminated or truncated)
            if online_belief and h is not None:
                for key, value in [("obs", normalize(obs)), ("actions", actions), ("next_obs", normalize(nxt))]:
                    updated = np.concatenate([histories[key][:, 1:], value[:, None]], axis=1)
                    histories[key] = np.where(active[:, None, None], updated, histories[key])
                h = infer_h()
            obs = nxt
            if frame_callback is not None:
                frame_callback(step + 1, envs, returns.copy(), active.copy())
            if not active.any():
                break
    finally:
        for env in envs:
            env.close()
    return {
        "return": float(returns.mean()),
        "returns": returns.reshape(len(ids), num_contexts),
        "lengths": lengths.reshape(len(ids), num_contexts),
        "context_ids": ids,
        "context_source": "official_offline_expert_transitions",
        "reward_labels": "10000 training transitions",
    }
