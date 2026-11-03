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
import concurrent.futures
import fcntl
import hashlib
import itertools
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GRIDS = {
    # Fig. 5, "# of Environments": all methods, context length 100.
    "layouts": {"agent": ["rfb", "bfb"], "num_train_layouts": [10, 20, 30, 50], "context_len": [100]},
    # Same with the baselines (FB, HILP, LAP); not context-dependent.
    "baselines": {"agent": ["fb", "hilp", "lap"], "num_train_layouts": [10, 20, 30, 50], "context_len": [100]},
    # Fig. 5, "Context Length": BFB / RFB with 20 training layouts.
    "context": {"agent": ["rfb", "bfb"], "num_train_layouts": [20], "context_len": [50, 150, 200]},
    # Fig. 6: kappa ablation for RFB.
    "kappa": dict(agent=["rfb"], num_train_layouts=[20], context_len=[100], **{"agent.kappa": [50, 200]}),
    # Pre-train (and cache) the encoders of all Fig. 5 settings before launching many jobs that share them.
    "encoders": {"agent": ["bfb"], "num_train_layouts": [10, 20, 30, 50], "context_len": [100], "fb_steps": [0]},
    "encoders_context": {"agent": ["bfb"], "num_train_layouts": [20], "context_len": [50, 150, 200], "fb_steps": [0]},
}


def build_jobs(grid: dict, seeds, extra, root_dir):
    keys = list(grid)
    jobs = []
    for values in itertools.product(*grid.values(), seeds):
        overrides = dict(zip(keys + ["seed"], values))
        name = "_".join(f"{k.split('.')[-1]}{v}" for k, v in overrides.items() if k != "agent")
        out_dir = os.path.join(root_dir, overrides["agent"], name)
        if extra:
            suffix = hashlib.sha256(json.dumps(list(extra)).encode()).hexdigest()[:8]
            out_dir += f"_variant{suffix}"
        args = [f"{k}={v}" for k, v in overrides.items()] + [f"out_dir={out_dir}"] + list(extra)
        jobs.append((out_dir, args))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=list(GRIDS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpus", type=str, nargs="+", default=["0"])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--root", default="runs/fig5")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides passed to every job")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1 or any(x.startswith("out_dir=") for x in args.overrides):
        parser.error("positive jobs-per-gpu required; use --root to select output directories")

    jobs = build_jobs(GRIDS[args.grid], args.seeds, args.overrides, args.root)
    todo = [j for j in jobs if not os.path.exists(os.path.join(j[0], "final.json"))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already finished, {len(todo)} to run", flush=True)
    q = queue.Queue()
    for j in todo:
        q.put(j)
    os.makedirs("logs", exist_ok=True)

    def worker(gpu):
        failures = []
        while True:
            try:
                out_dir, job_args = q.get_nowait()
            except queue.Empty:
                return failures
            os.makedirs(out_dir, exist_ok=True)
            # OS locks are atomic and released automatically after a crash.
            with open(os.path.join(out_dir, ".lock"), "a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print(f"[gpu {gpu}] skipped (running elsewhere): {out_dir}", flush=True)
                    continue
                if (Path(out_dir) / "final.json").exists():
                    continue
                env = dict(
                    os.environ, CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2"
                )
                log = os.path.join(out_dir, "console.log")
                cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_fourrooms.py"), *job_args]
                (Path(out_dir) / "command.json").write_text(json.dumps(cmd, indent=2))
                with open(log, "a") as f:
                    ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False).returncode
                if ret:
                    failures.append(out_dir)
                print(f"[gpu {gpu}] {'done' if ret == 0 else f'FAILED ({ret})'}: {out_dir}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus) * args.jobs_per_gpu) as pool:
        futures = [pool.submit(worker, g) for g in args.gpus for _ in range(args.jobs_per_gpu)]
        failures = list(itertools.chain.from_iterable(f.result() for f in futures))
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; inspect console.log in each output directory")


if __name__ == "__main__":
    main()
