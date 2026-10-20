"""Animated side-by-side rollouts of several agents on held-out layouts (unseen dynamics).

Each agent first explores the layout for ``context_len`` reward-free steps (not shown), then pursues each of the
3 evaluation goals from the start cell with the evaluation protocol of ``bfb/evaluation.py``: online belief update
and task codes z' ~ vMF(z, task_kappa). Layouts are the first held-out layouts and the random seed is fixed, so
the rollouts are not selected for success.

    uv run scripts/make_gif.py --runs <fb run> <bfb run> <rfb run> --out figures/rollouts.gif
"""

from __future__ import annotations

import argparse
import io

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from bfb.data import normalize_obs  # noqa: E402
from bfb.envs.fourrooms import START_POS, VecFourRooms, eval_goals, make_layout, random_walk, test_layout_seeds  # noqa: E402
from visualize import load_agent  # noqa: E402

COLORS = {'FB': '#7f7f7f', 'BFB': '#fc8d62', 'RFB': '#d95f02'}


def rollout(agent, cfg, grid, goal, rng, task_kappa, max_steps=100):
    """Trajectory (list of positions) of one episode, using the same inference as the evaluation."""
    params = agent.state.params
    h, buf = None, None
    if agent.use_context:
        ctx = random_walk(grid[None], np.array([START_POS]), cfg.context_len, rng)
        buf = {k: ctx[k] for k in ('observations', 'actions', 'next_observations')}
    env = VecFourRooms(grid[None], max_steps=max_steps)
    pos = env.reset(np.array([START_POS]), goal[None])
    z = agent.infer_z_goal(params, normalize_obs(goal[None]), normalize_obs(pos))
    path = [pos[0].copy()]
    for _ in range(max_steps):
        if agent.use_context:
            mu, _ = agent.encode(agent.enc_state.params, normalize_obs(buf['observations']),
                                 jnp.asarray(buf['actions']), normalize_obs(buf['next_observations']))
            h = agent.process_h(mu, agent.h_stats)
        key = jax.random.key(int(rng.integers(2**31)))
        a = np.asarray(agent.act(params, normalize_obs(pos), agent.perturb_task(key, z, task_kappa), h))
        prev = pos
        pos, _, done = env.step(a)
        path.append(pos[0].copy())
        if agent.use_context:
            for k, v in (('observations', prev), ('actions', a), ('next_observations', pos)):
                buf[k] = np.concatenate([buf[k][:, 1:], v[:, None]], 1)
        if done[0]:
            break
    return path, bool((pos[0] == goal).all())


def draw(ax, grid, goal, path, t, name, success):
    img = np.where(grid[..., None] == 1, np.array([0.18, 0.18, 0.2]), np.array([0.97, 0.97, 0.95]))
    ax.imshow(img)
    p = np.array(path[: t + 1])
    ax.plot(p[:, 1], p[:, 0], '-', color=COLORS[name], lw=2.5, alpha=0.6)
    ax.scatter([START_POS[1]], [START_POS[0]], marker='s', s=90, color='#1f78b4')
    ax.scatter([goal[1]], [goal[0]], marker='*', s=260, color='#33a02c')
    ax.scatter([p[-1, 1]], [p[-1, 0]], s=150, color=COLORS[name], edgecolor='k', zorder=3)
    finished = t >= len(path) - 1
    status = ('reached goal' if success else 'failed') if finished else f'step {t}'
    ax.set_title(f'{name}: {status}', fontsize=11,
                 color=('#33a02c' if success else '#e31a1c') if finished else 'black')
    ax.set_xticks([]), ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='runs in display order (e.g. FB, BFB, RFB)')
    parser.add_argument('--layouts', type=int, default=2, help='number of held-out layouts to show')
    parser.add_argument('--task-kappa', type=float, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='figures/rollouts.gif')
    args = parser.parse_args()

    agents = []
    for run in args.runs:
        agent, cfg = load_agent(run)
        agents.append((cfg.agent.name.upper(), agent, cfg))

    frames = []
    for layout_seed in test_layout_seeds()[: args.layouts]:
        grid = make_layout(int(layout_seed))
        for gi, goal in enumerate(eval_goals()):
            episodes = []
            for name, agent, cfg in agents:
                rng = np.random.default_rng(args.seed + 1000 * int(layout_seed) + gi)
                episodes.append(rollout(agent, cfg, grid, goal, rng, args.task_kappa))
            length = max(len(p) for p, _ in episodes)
            for t in list(range(length)) + [length - 1] * 6:  # hold the final frame
                fig, axes = plt.subplots(1, len(agents), figsize=(3.1 * len(agents), 3.5))
                for ax, (name, _, _), (path, success) in zip(np.atleast_1d(axes), agents, episodes):
                    draw(ax, grid, goal, path, min(t, len(path) - 1), name, success)
                fig.suptitle(f'Unseen layout {layout_seed}, goal {gi + 1}/3', fontsize=11)
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=70)
                plt.close(fig)
                frames.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    print(f'{len(frames)} frames written to {args.out}')


if __name__ == '__main__':
    main()
