"""Zero-shot evaluation protocol for Randomized Four-Rooms (Appendix C.2 / D).

For every evaluation layout (train layouts = layouts seen in the dataset, test layouts = 20 fixed held-out layouts):

1. *Context.* The agent performs one reward-free random-exploration episode of ``context_len`` steps starting from
   (1, 1). The transitions are passed to f_dyn and the belief mean is used as h (BFB / RFB only).
2. *Task.* For each of the 3 fixed goals (one in each room other than the start room) the task vector is
   z = B(goal) (Eq. 9) for FB-based agents (see ``bfb/baselines.py`` for HILP / LAP).
3. *Rollout.* Starting from (1, 1) (``start='fixed'``, default; ``'random'`` = uniformly random free cell) the agent acts greedily, a = argmax_a F(s, a, z[, h])^T z, for at most 100 steps.
   The return is 1 if the goal is reached and 0 otherwise (sparse goal reward, undiscounted).

The score of a set of layouts is the mean return over layouts x goals x ``num_contexts`` independent exploration
episodes (the only source of randomness, since the environment and the greedy policy are deterministic).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bfb.data import normalize_obs
from bfb.envs.fourrooms import START_POS, VecFourRooms, eval_goals, make_layout, random_free_cells, random_walk


def evaluate(
    agent,
    layout_seeds,
    context_len: int,
    num_contexts: int = 5,
    max_steps: int = 100,
    seed: int = 0,
    start: str = "fixed",
    online_belief: bool = False,
    belief_samples: int = 0,
    belief_kappa=None,
    epsilon: float = 0.0,
    task_kappa=None,
    grids_override=None,
    env_kwargs=None,
    context_mode: str = "correct",
    policy: str = "agent",
    action_calibration: bool = False,
    guard_blocked: bool = False,
    trajectory_callback=None,
):
    """Mean return over layouts x goals x contexts (see module docstring).

    Belief options (BFB / RFB):
        online_belief: re-encode the most recent ``context_len`` transitions (exploration context followed by the
            agent's own transitions) at every step, so the belief keeps being refined during the rollout.
        belief_samples, belief_kappa: act greedily w.r.t. the Q-values averaged over ``belief_samples`` beliefs
            drawn from vMF(h, belief_kappa) instead of the point estimate h.
        trajectory_callback: optional observer called with (step, positions, returns, done), including step 0.
            Arrays are copied; recording a rollout does not change evaluation random streams.
    """
    if start not in ("fixed", "random") or context_mode not in ("correct", "zero", "shuffled"):
        raise ValueError("invalid start or context_mode")
    if policy not in ("agent", "random", "oracle"):
        raise ValueError("invalid policy")
    if min(context_len, num_contexts, max_steps, len(layout_seeds)) <= 0:
        raise ValueError("evaluation sizes must be positive")
    if not 0 <= epsilon <= 1 or (belief_samples and belief_kappa is None):
        raise ValueError("invalid exploration or belief sampling settings")
    # Independent streams give all agents identical contexts, starts and environment noise.
    rng, ctx_rng, env_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    env_kwargs = {} if env_kwargs is None else env_kwargs
    goals = eval_goals()
    grids_one = np.stack([make_layout(s) for s in layout_seeds]) if grids_override is None else grids_override
    n_layouts, n_goals = len(layout_seeds), len(goals)

    # Episode index order: (layout, context, goal).
    layout_idx = np.repeat(np.arange(n_layouts), num_contexts * n_goals)
    goal_idx = np.tile(np.arange(n_goals), n_layouts * num_contexts)
    grids = grids_one[layout_idx]
    goal_pos = goals[goal_idx]
    if start == "fixed":
        starts = np.tile(np.array(START_POS), (len(grids), 1))
    else:  # uniformly random free cell other than the goal (the protocol of the original main_dynamics_discrete.py)
        starts = random_free_cells(grids, rng)
        same = (starts == goal_pos).all(-1)
        while same.any():
            starts[same] = random_free_cells(grids[same], rng)
            same = (starts == goal_pos).all(-1)

    params = None if agent is None else agent.state.params
    h = None
    use_context = policy == "agent" and agent.use_context
    oracle_context = use_context and getattr(agent, "oracle_bank", None) is not None
    action_maps = None
    collision_memory = None
    if (use_context and not oracle_context) or action_calibration or guard_blocked:
        from bfb.adaptation import CollisionMemory, infer_action_map

        ctx_grids = np.repeat(grids_one, num_contexts, 0)
        ctx = random_walk(
            ctx_grids, np.tile(np.array(START_POS), (len(ctx_grids), 1)), context_len, ctx_rng, **env_kwargs
        )
        buf = {k: np.repeat(ctx[k], n_goals, 0) for k in ("observations", "actions", "next_observations")}
        if action_calibration:
            action_maps = infer_action_map(ctx["observations"], ctx["actions"], ctx["next_observations"])
            ctx["actions"] = np.take_along_axis(action_maps, ctx["actions"], axis=1)
            action_maps = np.repeat(action_maps, n_goals, axis=0)
        if guard_blocked:
            collision_memory = CollisionMemory(len(grids))
            collision_memory.update(buf["observations"], buf["actions"], buf["next_observations"])
    if oracle_context:
        # Oracle-ID: one-hot layout index for training layouts, zeros for layouts never seen in training.
        n_train = agent.oracle_bank.shape[0]
        ids = np.asarray(layout_seeds)
        onehot = np.where((ids < n_train)[:, None], np.asarray(agent.oracle_bank)[np.minimum(ids, n_train - 1)], 0.0)
        h = jnp.asarray(np.repeat(onehot, num_contexts * n_goals, 0), dtype=jnp.float32)
    elif use_context:
        mu, _ = agent.encode(
            agent.enc_state.params,
            normalize_obs(ctx["observations"]),
            jnp.asarray(ctx["actions"]),
            normalize_obs(ctx["next_observations"]),
        )
        h = jnp.repeat(agent.process_h(mu, agent.h_stats), n_goals, 0)

    # Ablations preserve the amount of information but break its association with the evaluated dynamics.
    permutation = np.roll(np.arange(len(grids)).reshape(n_layouts, -1), 1, axis=0).ravel()

    def ablate(belief):
        if belief is None or context_mode == "correct":
            return belief
        return jnp.zeros_like(belief) if context_mode == "zero" else belief[permutation]

    h = ablate(h)
    z = agent.infer_z_goal(params, normalize_obs(goal_pos), normalize_obs(starts)) if policy == "agent" else None
    env = VecFourRooms(grids, max_steps=max_steps, rng=env_rng, **env_kwargs)
    pos = env.reset(starts, goal_pos)
    returns = np.zeros(len(grids), dtype=np.float32)
    steps_to_goal = np.full(len(grids), max_steps, dtype=np.int32)
    if trajectory_callback is not None:
        trajectory_callback(0, pos.copy(), returns.copy(), np.zeros(len(grids), dtype=bool))
    recompute_z = getattr(agent, "recompute_z_every_step", False)
    if policy == "oracle":
        from bfb.envs.fourrooms import MOTIONS
        from bfb.envs.grid_suites import distances

        distance_maps = np.stack([distances(g, goal) for g, goal in zip(grids, goal_pos)])
        motions = MOTIONS[np.asarray(env_kwargs.get("action_map", np.arange(4)))]
    for step in range(max_steps):
        if recompute_z:  # HILP: z = (phi(g) - phi(s_t)) / ||.|| at every step
            z = agent.infer_z_goal(params, normalize_obs(goal_pos), normalize_obs(pos))
        if policy == "agent" and (action_calibration or guard_blocked):
            task = z
            if task_kappa is not None:
                task = agent.perturb_task(jax.random.key(int(rng.integers(2**31))), z, task_kappa)
            q = np.asarray(agent.q_values(params, normalize_obs(pos), task, h))
            if action_maps is not None:
                q = np.take_along_axis(q, action_maps, axis=1)
            if collision_memory is not None:
                q = np.where(collision_memory.mask(pos), -np.inf, q)
            actions = q.argmax(-1)
        elif policy == "random":
            actions = rng.integers(4, size=len(grids))
        elif policy == "oracle":
            nxt = np.clip(pos[:, None, :] + motions[None], 0, grids.shape[-1] - 1)
            actions = distance_maps[np.arange(len(grids))[:, None], nxt[..., 0], nxt[..., 1]].argmin(-1)
        elif task_kappa is not None:  # act with a task code sampled around z at every step (vMF(z, task_kappa))
            key = jax.random.key(int(rng.integers(2**31)))
            actions = np.asarray(agent.act(params, normalize_obs(pos), agent.perturb_task(key, z, task_kappa), h))
        elif belief_samples and h is not None:
            key = jax.random.key(int(rng.integers(2**31)))
            actions = np.asarray(
                agent.act_marginal(params, normalize_obs(pos), z, h, key, belief_samples, belief_kappa)
            )
        else:
            actions = np.asarray(agent.act(params, normalize_obs(pos), z, h))
        if epsilon > 0:  # epsilon-greedy (diagnostic only)
            explore = rng.random(len(actions)) < epsilon
            actions = np.where(explore, rng.integers(4, size=len(actions)), actions)
        prev = pos
        pos, reward, done = env.step(actions)
        if collision_memory is not None:
            collision_memory.update(prev, actions, pos)
        if online_belief and h is not None and not oracle_context:
            for k, v in (("observations", prev), ("actions", actions), ("next_observations", pos)):
                buf[k] = np.concatenate([buf[k][:, 1:], v[:, None]], 1)
            encoder_actions = buf["actions"]
            if action_calibration:
                # Reuse all initial permutation evidence: the mapping is fixed for each episode.
                encoder_actions = np.take_along_axis(action_maps, encoder_actions, axis=1)
            mu, _ = agent.encode(
                agent.enc_state.params,
                normalize_obs(buf["observations"]),
                jnp.asarray(encoder_actions),
                normalize_obs(buf["next_observations"]),
            )
            h = ablate(agent.process_h(mu, agent.h_stats))
        steps_to_goal[reward > 0] = step + 1
        returns += reward
        if trajectory_callback is not None:
            trajectory_callback(step + 1, pos.copy(), returns.copy(), done.copy())
        if done.all():
            break
    per_layout = returns.reshape(n_layouts, num_contexts * n_goals).mean(1)
    per_goal = returns.reshape(-1, n_goals).mean(0)
    return {
        "return": float(returns.mean()),
        "per_layout": per_layout,
        "per_goal": per_goal,
        "successes": returns.reshape(n_layouts, num_contexts, n_goals),
        "steps_to_goal": steps_to_goal.reshape(n_layouts, num_contexts, n_goals),
        "layout_seeds": np.asarray(layout_seeds),
        "context_mode": context_mode,
    }
