"""Paired, reward-free-context evaluation of the continuous diagnostic benchmarks."""

from dataclasses import asdict

import jax.numpy as jnp
import numpy as np

from bfb.envs.pointmass import PointMass, contexts

GOALS = np.array([[0.75, 0.75], [-0.75, 0.75], [0.75, -0.75], [-0.75, -0.75]], dtype=np.float32)


def evaluate_continuous(
    agent,
    family="pointmass",
    split="iid",
    context_len=64,
    num_contexts=8,
    num_dynamics=16,
    max_steps=80,
    seed=20261103,
    online_belief=False,
    policy="agent",
    context_mode="correct",
    action_reference=None,
    trajectory_callback=None,
):
    """Evaluate fixed weights; optionally observe (step, observations, successes, done) on copied arrays."""
    if policy not in ("agent", "random", "oracle") or context_mode not in ("correct", "zero", "shuffled"):
        raise ValueError("invalid policy or context intervention")
    rng_context, rng_policy = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    dynamics = contexts(family, split, num_dynamics)
    context_dynamics = [d for d in dynamics for _ in range(num_contexts)]
    env = PointMass(context_dynamics, family)
    obs = env.reset(rng_context.uniform(-0.8, 0.8, (env.batch, 2)))
    history = {"obs": [], "actions": [], "next_obs": []}
    for _ in range(context_len):
        action = rng_context.uniform(-1, 1, (env.batch, 2)).astype(np.float32)
        nxt = env.step(action)
        for k, value in [("obs", obs), ("actions", action), ("next_obs", nxt)]:
            history[k].append(value)
        obs = nxt
    history = {k: np.repeat(np.stack(v, 1), len(GOALS), axis=0) for k, v in history.items()}
    adapter = None
    if action_reference is not None:
        from bfb.affine_adaptation import AffineActionAdapter

        adapter = AffineActionAdapter(history["obs"], history["actions"], history["next_obs"], action_reference)
    env = PointMass([d for d in context_dynamics for _ in GOALS], family)
    obs = env.reset(np.zeros((env.batch, 2), dtype=np.float32))
    goals = np.tile(GOALS, (len(context_dynamics), 1))
    goal_obs = goals if env.obs_dim == 2 else np.concatenate([goals, np.zeros_like(goals)], -1)
    success = np.zeros(env.batch, dtype=bool)
    steps = np.full(env.batch, max_steps, dtype=np.int32)
    distances = np.zeros(env.batch, dtype=np.float32)
    z = agent.infer_z_goal(agent.state.params, jnp.asarray(goal_obs)) if policy == "agent" else None
    permutation = np.roll(np.arange(env.batch).reshape(num_dynamics, -1), 1, axis=0).ravel()

    def belief():
        if policy != "agent" or not agent.use_context:
            return None
        actions = history["actions"] if adapter is None else adapter.to_reference(history["obs"], history["actions"])
        mu, _ = agent.encode(
            agent.enc_state.params, jnp.asarray(history["obs"]), jnp.asarray(actions), jnp.asarray(history["next_obs"])
        )
        h = agent.process_h(mu, agent.h_stats)
        return jnp.zeros_like(h) if context_mode == "zero" else h[permutation] if context_mode == "shuffled" else h

    h = belief()
    if trajectory_callback is not None:
        trajectory_callback(0, obs.copy(), success.astype(float), success.copy())
    for t in range(max_steps):
        if policy == "random":
            action = rng_policy.uniform(-1, 1, (env.batch, 2))
        elif policy == "oracle":
            desired = (goals - obs[:, :2]) / env.step_scale
            if env.obs_dim == 4:
                desired = (np.clip(desired, -1, 1) - env.damping * obs[:, 2:]) / (1 - env.damping)
            force = (desired - env.wind) / env.gain
            action = np.einsum("bji,bj->bi", env.rotation, force)
        else:
            action = np.asarray(agent.act(agent.state.params, jnp.asarray(obs), z, h))
            if adapter is not None:
                action = adapter.to_actual(obs, action)
        nxt = env.step(action)
        distances = np.linalg.norm(nxt[:, :2] - goals, axis=-1)
        hit = (distances <= 0.12) & ~success
        steps[hit] = t + 1
        success |= hit
        # Finished rollouts do not contribute fabricated stationary transitions to the beliefs.
        if online_belief and h is not None:
            for k, value in [("obs", obs), ("actions", action), ("next_obs", nxt)]:
                updated = np.concatenate([history[k][:, 1:], value[:, None]], axis=1)
                history[k] = np.where(success[:, None, None], history[k], updated)
            h = belief()
        obs = nxt
        if trajectory_callback is not None:
            trajectory_callback(t + 1, obs.copy(), success.astype(float), success.copy())
    shape = (num_dynamics, num_contexts, len(GOALS))
    return {
        "return": float(success.mean()),
        "per_dynamics": success.reshape(shape).mean((1, 2)),
        "successes": success.reshape(shape),
        "steps_to_goal": steps.reshape(shape),
        "final_distance": distances.reshape(shape),
        "dynamics": [asdict(d) for d in dynamics],
    }
