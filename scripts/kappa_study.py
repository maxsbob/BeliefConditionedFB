"""Influence of kappa on behaviour diversity (cf. Fig. 6): returns when acting with task codes z' ~ vMF(z, kappa).

Small kappa = diverse (more random) behaviour, large kappa = nearly greedy; 'greedy' = argmax with z itself.

    uv run scripts/kappa_study.py --runs runs/fig5/bfb/num_train_layouts20_context_len100_seed* \
        runs/fig5/rfb/num_train_layouts20_context_len100_seed* --out figures/kappa.png
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from bfb.envs.fourrooms import test_layout_seeds, train_layout_seeds
from bfb.evaluation import evaluate
from visualize import load_agent

COLORS = {'bfb': '#fc8d62', 'rfb': '#d95f02', 'fb': '#bababa'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--kappas', type=float, nargs='+', default=[50, 100, 200, 400, 1000])
    parser.add_argument('--num-contexts', type=int, default=20)
    parser.add_argument('--out', default='figures/kappa.png')
    args = parser.parse_args()

    results = defaultdict(lambda: defaultdict(list))  # (agent, split) -> kappa -> [returns over seeds]
    for run in args.runs:
        agent, cfg = load_agent(run)
        for kappa in [None, *args.kappas]:
            for split, seeds in [('train', train_layout_seeds(cfg.num_train_layouts)), ('test', test_layout_seeds())]:
                ret = evaluate(agent, seeds, cfg.context_len, args.num_contexts, seed=0, online_belief=agent.use_context,
                               task_kappa=kappa)['return']
                results[(cfg.agent.name, split)][kappa if kappa else 'greedy'].append(ret)
        print(f'{run}: done', flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    table = {f'{a}/{s}': {str(k): v for k, v in d.items()} for (a, s), d in results.items()}
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(table, f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, split in zip(axes, ['train', 'test']):
        for agent in ['fb', 'bfb', 'rfb']:
            d = results.get((agent, split))
            if not d:
                continue
            m = np.array([np.mean(d[k]) for k in args.kappas])
            s = np.array([np.std(d[k]) for k in args.kappas])
            ax.plot(args.kappas, m, marker='o', color=COLORS[agent], label=agent.upper())
            ax.fill_between(args.kappas, m - s, m + s, color=COLORS[agent], alpha=0.2)
            ax.axhline(np.mean(d['greedy']), ls=':', color=COLORS[agent])
        ax.set_xscale('log')
        ax.set_xlabel(r'$\kappa$ of the acting distribution  (dotted: greedy)')
        ax.set_title('Train layouts' if split == 'train' else 'Held-out layouts (unseen dynamics)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Return')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(json.dumps({k: {kk: round(float(np.mean(vv)), 3) for kk, vv in v.items()} for k, v in table.items()}, indent=1))


if __name__ == '__main__':
    main()
