"""Audit existing grid checkpoints on paired tasks and declared dynamics shifts.

Example: uv run scripts/benchmark_grid.py --run runs/fig5/bfb/num_train_layouts30_context_len100_seed0
Results include every trial; historical training records are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import jax

from bfb.envs.grid_suites import SUITES, grid_suite
from bfb.evaluation import evaluate
from bfb.experiment import ROOT, atomic_json, file_sha256, load_agent, provenance

PROTOCOLS = {
    "greedy_static": {},
    "greedy_online": {"online_belief": True},
    "sampled_static": {"task_kappa": 200.0},
    "sampled_online": {"online_belief": True, "task_kappa": 200.0},
    "shuffled_online": {"online_belief": True, "task_kappa": 200.0, "context_mode": "shuffled"},
    "zero_online": {"online_belief": True, "task_kappa": 200.0, "context_mode": "zero"},
    "epsilon_online": {"online_belief": True, "epsilon": 0.1},
    "calibrated_online": {"online_belief": True, "task_kappa": 200.0, "action_calibration": True},
    "guarded_online": {"online_belief": True, "task_kappa": 200.0, "action_calibration": True, "guard_blocked": True},
}


def main():
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "data" / "jax_cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="omit to evaluate random and full-map oracle controls")
    parser.add_argument("--label", help="explicit variant label in result tables (e.g. bfb_compact)")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS)[:4])
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--num-layouts", type=int, default=20)
    parser.add_argument("--num-contexts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--out", type=Path, default=Path("results/grid"))
    args = parser.parse_args()
    agent, cfg = load_agent(args.run) if args.run else (None, None)
    meta = provenance()
    name = f"{cfg.agent.name}_n{cfg.num_train_layouts}_s{cfg.seed}" if cfg else "controls"
    checkpoint_hash = file_sha256(args.run / "checkpoint.pkl") if args.run else None
    for suite in args.suites:
        seeds, grids, env_kwargs = grid_suite(
            suite, args.num_layouts, args.split, exclude=max(50, cfg.num_train_layouts if cfg else 50)
        )
        protocols = args.protocols if agent is not None else ("random", "oracle")
        for protocol in protocols:
            start_time = time.monotonic()
            options = PROTOCOLS[protocol] if agent is not None else {"policy": protocol}
            result = evaluate(
                agent,
                seeds,
                cfg.context_len if cfg else 100,
                args.num_contexts,
                seed=args.seed,
                grids_override=grids,
                env_kwargs=env_kwargs,
                **options,
            )
            payload = {
                "schema_version": 1,
                "benchmark": suite,
                "split": args.split,
                "protocol": protocol,
                "algorithm": args.label or (cfg.agent.name if cfg else protocol),
                "training_seed": cfg.seed if cfg else None,
                "num_train_layouts": cfg.num_train_layouts if cfg else 0,
                "checkpoint": str(args.run) if args.run else None,
                "checkpoint_sha256": checkpoint_hash,
                "grid_sha256": hashlib.sha256(grids.tobytes()).hexdigest(),
                "evaluation_seed": args.seed,
                "num_contexts": args.num_contexts,
                "max_steps": 100,
                "context_len": cfg.context_len if cfg else 100,
                "env_kwargs": env_kwargs,
                "options": options,
                "provenance": meta,
                "result": result,
                "seconds": time.monotonic() - start_time,
            }
            # Include the checkpoint digest so different architectures with the same seed never overwrite.
            suffix = checkpoint_hash[:8] if checkpoint_hash else "none"
            path = args.out / f"{name}_{suffix}_{suite}_{args.split}_{protocol}_e{args.seed}.json"
            if path.exists():
                raise FileExistsError(f"{path}: choose another output directory to preserve the previous evaluation")
            atomic_json(path, payload)
            print(f"{name} {suite} {protocol}: {result['return']:.4f} ({payload['seconds']:.1f}s)", flush=True)
    jax.clear_caches()


if __name__ == "__main__":
    main()
