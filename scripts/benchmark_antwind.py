"""Re-evaluate an AntWind checkpoint, or evaluate random and zero-action controls."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from bfb.antwind import evaluate_antwind, load_antwind
from bfb.continuous import ContinuousFBAgent
from bfb.experiment import atomic_json, file_sha256, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/antwind_official"))
    parser.add_argument("--out", type=Path, default=Path("results/antwind_controls"))
    args = parser.parse_args()
    arrays, normalization, labels, hashes = load_antwind(args.data)
    del arrays
    agent = None
    metadata = {"provenance": provenance(), "data_sha256": hashes}
    if args.run:
        cfg = OmegaConf.create(json.loads((args.run / "final.json").read_text())["config"])
        agent = ContinuousFBAgent(cfg.agent, 105, 8, cfg.seed)
        agent.load(args.run / "checkpoint.pkl")
        metadata.update(
            algorithm=cfg.agent.name, training_seed=cfg.seed, checkpoint_sha256=file_sha256(args.run / "checkpoint.pkl")
        )
    for split in ("train", "test"):
        for protocol in ("static", "online", "shuffled") if agent else ("random", "zero"):
            result = evaluate_antwind(
                agent,
                args.data,
                normalization,
                labels,
                split,
                online_belief=protocol != "static",
                context_mode="shuffled" if protocol == "shuffled" else "correct",
                policy="agent" if agent else protocol,
            )
            name = f"{cfg.agent.name}_s{cfg.seed}" if agent else protocol
            path = args.out / f"{name}_{split}_{protocol}.json"
            if path.exists():
                raise FileExistsError(path)
            atomic_json(
                path,
                {
                    "benchmark": "antwind",
                    "algorithm": cfg.agent.name if agent else protocol,
                    "split": split,
                    "protocol": protocol,
                    "result": result,
                    **metadata,
                },
            )
            print(name, split, protocol, result["return"], flush=True)


if __name__ == "__main__":
    main()
