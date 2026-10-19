"""Re-evaluate saved runs, optionally with different inference-time options.

    uv run scripts/eval_checkpoint.py runs/fig5/rfb/<run> --online-belief
    uv run scripts/eval_checkpoint.py runs/fig5/rfb/<run> --belief-samples 16 --belief-kappa 100 --num-contexts 10
"""

from __future__ import annotations

import argparse
import json
import os

from bfb.envs.fourrooms import test_layout_seeds, train_layout_seeds
from bfb.evaluation import evaluate
from visualize import load_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--start', choices=['fixed', 'random'], default='fixed')
    parser.add_argument('--num-contexts', type=int, default=10)
    parser.add_argument('--online-belief', action='store_true')
    parser.add_argument('--belief-samples', type=int, default=0)
    parser.add_argument('--belief-kappa', type=float, default=None)
    parser.add_argument('--context-len', type=int, default=None, help='default: the training context length')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', default=None, help='if set, results are saved to <run>/eval_<tag>.json')
    args = parser.parse_args()

    for run in args.runs:
        agent, cfg = load_agent(run)
        context_len = args.context_len or cfg.context_len
        res = {}
        for split, seeds in [('train', train_layout_seeds(cfg.num_train_layouts)), ('test', test_layout_seeds())]:
            out = evaluate(agent, seeds, context_len, args.num_contexts, cfg.eval.max_steps, seed=args.seed,
                           start=args.start, online_belief=args.online_belief, belief_samples=args.belief_samples,
                           belief_kappa=args.belief_kappa)
            res[split] = out['return']
        print(f"{run}: train={res['train']:.3f} test={res['test']:.3f}", flush=True)
        if args.tag:
            with open(os.path.join(run, f'eval_{args.tag}.json'), 'w') as f:
                json.dump({'args': vars(args), **res}, f, indent=1)


if __name__ == '__main__':
    main()
