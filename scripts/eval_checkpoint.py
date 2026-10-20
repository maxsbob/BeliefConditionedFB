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
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--task-kappa', type=float, default=None)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', default=None, help='if set, results are saved to <run>/eval_<tag>.json')
    parser.add_argument('--rescore', action='store_true',
                        help="recompute the run's final metrics with the default protocol of configs/train.yaml and "
                             "store them in final.json (for runs trained before the protocol was changed)")
    args = parser.parse_args()
    if args.rescore:
        return rescore(args.runs)

    for run in args.runs:
        agent, cfg = load_agent(run)
        context_len = args.context_len or cfg.context_len
        res = {}
        for split, seeds in [('train', train_layout_seeds(cfg.num_train_layouts)), ('test', test_layout_seeds())]:
            out = evaluate(agent, seeds, context_len, args.num_contexts, cfg.eval.max_steps, seed=args.seed,
                           start=args.start, online_belief=args.online_belief, belief_samples=args.belief_samples,
                           belief_kappa=args.belief_kappa, epsilon=args.epsilon, task_kappa=args.task_kappa)
            res[split] = out['return']
            res[f'{split}_per_goal'] = [round(float(x), 3) for x in out['per_goal']]
        print(f"{run}: train={res['train']:.3f} {res['train_per_goal']} test={res['test']:.3f} {res['test_per_goal']}",
              flush=True)
        if args.tag:
            with open(os.path.join(run, f'eval_{args.tag}.json'), 'w') as f:
                json.dump({'args': vars(args), **res}, f, indent=1)


def rescore(runs):
    from omegaconf import OmegaConf

    ev = OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'train.yaml')).eval
    for run in runs:
        agent, cfg = load_agent(run)
        res = {}
        for split, seeds in [('train', train_layout_seeds(cfg.num_train_layouts)), ('test', test_layout_seeds())]:
            kw = dict(context_len=cfg.context_len, num_contexts=ev.num_contexts, max_steps=ev.max_steps, seed=0,
                      online_belief=ev.online_belief, task_kappa=ev.task_kappa)
            res[f'eval/{split}_return'] = evaluate(agent, seeds, **kw)['return']
            res[f'eval/{split}_return_random_start'] = evaluate(agent, seeds, start='random', **kw)['return']
            res[f'eval/{split}_return_greedy_point_belief'] = evaluate(
                agent, seeds, **{**kw, 'online_belief': False, 'task_kappa': None})['return']
        path = os.path.join(run, 'final.json')
        with open(path) as f:
            data = json.load(f)
        data['final'] = {'step': data['final']['step'], **res}
        data['rescored_with_eval'] = OmegaConf.to_container(ev)
        with open(path, 'w') as f:
            json.dump(data, f, indent=1)
        print(f"{run}: rescored train={res['eval/train_return']:.3f} test={res['eval/test_return']:.3f}", flush=True)


if __name__ == '__main__':
    main()
