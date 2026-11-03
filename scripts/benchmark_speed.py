"""Measure steady-state grid training throughput, excluding compilation and belief-bank construction."""

import argparse
import time
from pathlib import Path

import jax
import numpy as np

from bfb.data import FourRoomsDataset, load_or_generate
from bfb.experiment import atomic_json, load_agent, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/throughput.json"))
    args = parser.parse_args()
    results = []
    for run in args.runs:
        agent, cfg = load_agent(run)
        dataset = FourRoomsDataset(
            load_or_generate(
                cfg.data.path, cfg.num_train_layouts, cfg.data.total_transitions, cfg.data.episode_length, cfg.data.seed
            )
        )
        bank = None
        if agent.use_context:
            mu, log_std = agent.build_belief_bank(
                agent.enc_state.params, dataset.arrays, jax.random.key(0), 128, cfg.context_len, dataset.episode_length
            )
            bank = (mu, log_std, agent.h_stats)
        state, info = agent.train_chunk(
            agent.state, dataset.arrays, bank, jax.random.key(1), args.updates, cfg.batch_size
        )
        jax.block_until_ready((state, info))
        times = []
        for trial in range(args.repeats):
            start = time.perf_counter()
            state, info = agent.train_chunk(
                state, dataset.arrays, bank, jax.random.key(trial + 2), args.updates, cfg.batch_size
            )
            jax.block_until_ready((state, info))
            times.append(time.perf_counter() - start)
        results.append(
            {
                "run": str(run),
                "batch_size": cfg.batch_size,
                "updates_per_repeat": args.updates,
                "seconds": times,
                "median_updates_per_second": args.updates / np.median(times),
                "median_transitions_per_second": args.updates * cfg.batch_size / np.median(times),
                "fb_parameters": sum(x.size for x in jax.tree.leaves(agent.state.params)),
                "encoder_parameters": sum(x.size for x in jax.tree.leaves(agent.enc_state.params))
                if agent.enc_state
                else 0,
            }
        )
        print(results[-1], flush=True)
        jax.clear_caches()
    atomic_json(
        args.out,
        {
            "provenance": provenance(),
            "results": results,
            "scope": "training updates only; different presets have different batch sizes and quality",
        },
    )


if __name__ == "__main__":
    main()
