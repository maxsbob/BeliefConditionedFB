"""Qualitative OOD figures from saved checkpoints.

1. Beliefs: PCA of belief means h for training and held-out layouts (cf. Fig. 7).
2. Q-values and greedy policies on a training and a held-out layout for several agents (cf. Fig. 12).

    uv run scripts/visualize.py --runs runs/fig5/fb/<run> runs/fig5/bfb/<run> runs/fig5/rfb/<run> --out figures
"""

from __future__ import annotations

import argparse
import json
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from bfb.agent import FBAgent
from bfb.baselines import LatentQAgent
from bfb.data import normalize_obs
from bfb.diagnostics import belief_means
from bfb.envs.fourrooms import NUM_ACTIONS, START_POS, eval_goals, make_layout, random_walk, test_layout_seeds

ARROWS = ['↑', '↓', '←', '→']


def load_agent(run_dir):
    with open(os.path.join(run_dir, 'final.json')) as f:
        cfg = OmegaConf.create(json.load(f)['config'])
    if cfg.agent.name in ('hilp', 'lap'):
        agent = LatentQAgent(cfg.agent, 2, NUM_ACTIONS, cfg.data.episode_length, cfg.seed)
    else:
        agent = FBAgent(cfg.agent, 2, NUM_ACTIONS, cfg.seed)
    agent.load(os.path.join(run_dir, 'checkpoint.pkl'))
    return agent, cfg


def plot_beliefs(agent, cfg, out):
    train = np.arange(min(cfg.num_train_layouts, 10))
    test = test_layout_seeds()[:5]
    h_train = belief_means(agent, train, cfg.context_len, 30, seed=1)
    h_test = belief_means(agent, test, cfg.context_len, 30, seed=2)
    pca = PCA(2).fit(h_train.reshape(-1, h_train.shape[-1]))
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = plt.get_cmap('tab10')
    for i in range(len(train)):
        p = pca.transform(h_train[i])
        ax.scatter(p[:, 0], p[:, 1], s=8, color=cmap(i % 10), alpha=0.7)
    for i in range(len(test)):
        p = pca.transform(h_test[i])
        ax.scatter(p[:, 0], p[:, 1], s=26, marker='x', color='k' if i == 0 else plt.get_cmap('Dark2')(i), alpha=0.8)
    ax.set_title(f'Beliefs h (PCA): {len(train)} train layouts (dots), {len(test)} held-out (crosses)', fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def q_map(agent, cfg, layout_seed, goal, rng):
    grid = make_layout(layout_seed)
    free = np.argwhere(grid == 0)
    h = None
    if agent.use_context:
        ctx = random_walk(grid[None], np.array([START_POS]), cfg.context_len, rng)
        mu, _ = agent.encode(agent.enc_state.params, normalize_obs(ctx['observations']), ctx['actions'],
                             normalize_obs(ctx['next_observations']))
        h = jnp.repeat(agent.process_h(mu, agent.h_stats), len(free), 0)
    g = np.repeat(goal[None], len(free), 0)
    z = agent.infer_z_goal(agent.state.params, normalize_obs(g), normalize_obs(free))
    q = np.asarray(agent.q_values(agent.state.params, normalize_obs(free), z, h))
    values = np.full(grid.shape, np.nan)
    actions = np.full(grid.shape, -1)
    values[free[:, 0], free[:, 1]] = q.max(-1)
    actions[free[:, 0], free[:, 1]] = q.argmax(-1)
    return grid, values, actions


def plot_q_maps(agents, out, layouts, goal_idx=2):
    goal = eval_goals()[goal_idx]
    fig, axes = plt.subplots(len(layouts), len(agents), figsize=(2.6 * len(agents), 2.7 * len(layouts)), squeeze=False)
    for col, (name, (agent, cfg)) in enumerate(agents.items()):
        for row, (split, seed) in enumerate(layouts):
            grid, values, actions = q_map(agent, cfg, seed, goal, np.random.default_rng(0))
            ax = axes[row, col]
            v = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-8)
            img = plt.get_cmap('inferno')(np.nan_to_num(v))
            img[grid == 1] = (0.1, 0.1, 0.1, 1)
            ax.imshow(img)
            for (r, c), a in np.ndenumerate(actions):
                if a >= 0:
                    ax.text(c, r, ARROWS[a], ha='center', va='center', fontsize=7, color='white')
            ax.text(START_POS[1], START_POS[0], 'S', ha='center', va='center', fontsize=9, color='cyan', weight='bold')
            ax.text(goal[1], goal[0], '★', ha='center', va='center', fontsize=11, color='lime')
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(split)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', default='figures')
    parser.add_argument('--train-layout', type=int, default=0)
    parser.add_argument('--test-layout', type=int, default=None, help='default: first held-out layout')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    agents = {}
    for run in args.runs:
        agent, cfg = load_agent(run)
        agents[cfg.agent.name.upper()] = (agent, cfg)
        if agent.use_context:
            plot_beliefs(agent, cfg, os.path.join(args.out, f'beliefs_{cfg.agent.name}.png'))
    test_layout = args.test_layout if args.test_layout is not None else int(test_layout_seeds()[0])
    plot_q_maps(agents, os.path.join(args.out, 'q_maps.png'),
                [('train layout', args.train_layout), ('held-out layout', test_layout)])
    print(f'figures written to {args.out}/')


if __name__ == '__main__':
    main()
