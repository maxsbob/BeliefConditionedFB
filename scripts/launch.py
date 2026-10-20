"""Run a grid of training jobs, a few at a time on each GPU.

Every job is one call of scripts/train_fourrooms.py with Hydra overrides. Jobs whose output directory already
contains ``final.json`` are skipped, so an interrupted grid can be resumed by re-running the same command.

Examples:
    # Fig. 5 (left): number of training layouts
    uv run scripts/launch.py --grid layouts --gpus 0 1 --jobs-per-gpu 3
    # Fig. 5 (right): context length
    uv run scripts/launch.py --grid context --gpus 0 1 --jobs-per-gpu 3
"""

from __future__ import annotations

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

GRIDS = {
    # Fig. 5, "# of Environments": all methods, context length 100.
    'layouts': dict(agent=['rfb', 'bfb'], num_train_layouts=[10, 20, 30, 50], context_len=[100]),
    # Same with the baselines (FB, HILP, LAP); not context-dependent.
    'baselines': dict(agent=['fb', 'hilp', 'lap'], num_train_layouts=[10, 20, 30, 50], context_len=[100]),
    # Fig. 5, "Context Length": BFB / RFB with 20 training layouts.
    'context': dict(agent=['rfb', 'bfb'], num_train_layouts=[20], context_len=[50, 150, 200]),
    # Fig. 6: kappa ablation for RFB.
    'kappa': dict(agent=['rfb'], num_train_layouts=[20], context_len=[100], **{'agent.kappa': [50, 200]}),
    # Pre-train (and cache) the encoders of all Fig. 5 settings before launching many jobs that share them.
    'encoders': dict(agent=['bfb'], num_train_layouts=[10, 20, 30, 50], context_len=[100]),
    'encoders_context': dict(agent=['bfb'], num_train_layouts=[20], context_len=[50, 150, 200]),
}


def build_jobs(grid: dict, seeds, extra, root_dir):
    keys = list(grid)
    jobs = []
    for values in itertools.product(*grid.values(), seeds):
        overrides = dict(zip(keys + ['seed'], values))
        name = '_'.join(f'{k.split(".")[-1]}{v}' for k, v in overrides.items() if k != 'agent')
        out_dir = os.path.join(root_dir, overrides['agent'], name)
        args = [f'{k}={v}' for k, v in overrides.items()] + [f'out_dir={out_dir}'] + list(extra)
        jobs.append((out_dir, args))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', choices=list(GRIDS), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--gpus', type=str, nargs='+', default=['0'])
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    parser.add_argument('--root', default='runs/fig5')
    parser.add_argument('overrides', nargs='*', help='extra Hydra overrides passed to every job')
    args = parser.parse_args()

    jobs = build_jobs(GRIDS[args.grid], args.seeds, args.overrides, args.root)
    todo = [j for j in jobs if not os.path.exists(os.path.join(j[0], 'final.json'))]
    print(f'{len(jobs)} jobs, {len(jobs) - len(todo)} already finished, {len(todo)} to run', flush=True)
    q = queue.Queue()
    for j in todo:
        q.put(j)
    os.makedirs('logs', exist_ok=True)

    def worker(gpu):
        while True:
            try:
                out_dir, job_args = q.get_nowait()
            except queue.Empty:
                return
            # Skip jobs another launcher is currently running (lock file with a live pid).
            lock = os.path.join(out_dir, '.running')
            if os.path.exists(lock):
                try:
                    os.kill(int(open(lock).read()), 0)
                    print(f'[gpu {gpu}] skipped (running elsewhere): {out_dir}', flush=True)
                    continue
                except (OSError, ValueError):
                    pass
            os.makedirs(out_dir, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_PREALLOCATE='false')
            log = os.path.join('logs', out_dir.replace('/', '_') + '.log')
            cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'train_fourrooms.py'), *job_args]
            with open(log, 'w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                with open(lock, 'w') as lf:
                    lf.write(str(proc.pid))
                ret = proc.wait()
            os.remove(lock)
            print(f'[gpu {gpu}] {"done" if ret == 0 else f"FAILED ({ret})"}: {out_dir}', flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus for _ in range(args.jobs_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
