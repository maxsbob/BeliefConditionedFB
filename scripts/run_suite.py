"""Bounded multi-GPU experiment runner with OS locks, manifests and failure propagation.

uv run scripts/run_suite.py --suite continuous --gpus 0 1 --jobs-per-gpu 2
uv run scripts/run_suite.py --suite antwind --gpus 0 1 --seeds 0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import itertools
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["continuous", "antwind", "grid"], required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--agents", nargs="+", choices=["fb", "bfb", "rfb"], default=["fb", "bfb", "rfb"])
    parser.add_argument("--families", nargs="+", choices=["pointmass", "inertial"], default=["pointmass", "inertial"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be positive")
    if any(v.startswith("out_dir=") for v in args.overrides):
        parser.error("use --root to change output paths")
    families = args.families if args.suite == "continuous" else [args.suite]
    root = args.root or Path("runs") / args.suite
    jobs = []
    for family, seed, method in itertools.product(families, args.seeds, args.agents):
        out = root / family / method / f"seed{seed}" if args.suite == "continuous" else root / method / f"seed{seed}"
        overrides = [f"agent={method}", f"seed={seed}", f"out_dir={out}"]
        if args.suite == "continuous":
            overrides.append(f"family={family}")
        if args.suite == "grid":
            script = "train_fourrooms.py"
            overrides += ["num_train_layouts=30"]
        else:
            script = "train_continuous.py"
            if args.suite == "antwind":
                overrides = ["--config-name", "antwind", *overrides]
        command = [sys.executable, str(ROOT / "scripts" / script), *overrides, *args.overrides]
        jobs.append({"out": str(out), "command": command})
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    # Version the manifest rather than overwriting an earlier launch's record.
    manifest = root / f"launch_{os.getpid()}.json"
    manifest.write_text(json.dumps({"gpus": args.gpus, "jobs_per_gpu": args.jobs_per_gpu, "jobs": jobs}, indent=2))
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        failures = []
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return failures
            out = Path(job["out"])
            out.mkdir(parents=True, exist_ok=True)
            with (out / ".lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print(f"already running: {out}", flush=True)
                    continue
                job_file = out / "job.json"
                if job_file.exists() and json.loads(job_file.read_text())["command"] != job["command"]:
                    failures.append(f"configuration mismatch: {out}; choose a different --root")
                    continue
                if (out / "final.json").exists():
                    print(f"complete: {out}", flush=True)
                    continue
                job_file.write_text(json.dumps(job, indent=2))
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    OMP_NUM_THREADS="2",
                )
                print(f"gpu {gpu}: {out}", flush=True)
                with (out / "console.log").open("a") as log:
                    result = subprocess.run(
                        job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                if result.returncode:
                    failures.append(f"{out}: exit {result.returncode}")
                    print(f"FAILED: {out}", flush=True)
                else:
                    print(f"finished: {out}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus) * args.jobs_per_gpu) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
        failures = list(itertools.chain.from_iterable(f.result() for f in futures))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
