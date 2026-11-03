"""Evaluate continuous checkpoints with an optional transition-fitted affine action adapter."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from bfb.affine_adaptation import fit_dynamics
from bfb.continuous import ContinuousFBAgent
from bfb.continuous_evaluation import evaluate_continuous
from bfb.envs.pointmass import generate_dataset
from bfb.experiment import atomic_json, file_sha256, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--family", choices=["pointmass", "inertial"], default="pointmass")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["iid", "rotation", "gain"])
    parser.add_argument("--out", type=Path, default=Path("results/continuous_adapted"))
    args = parser.parse_args()
    agent, reference = None, None
    if args.run:
        cfg = OmegaConf.create(json.loads((args.run / "final.json").read_text())["config"])
        family = cfg.family
        agent = ContinuousFBAgent(cfg.agent, 4 if family == "inertial" else 2, seed=cfg.seed)
        agent.load(args.run / "checkpoint.pkl")
        name = f"{family}_{cfg.agent.name}_s{cfg.seed}"
        if args.adapter:
            arrays = generate_dataset(family, cfg.num_train_contexts, cfg.episodes, cfg.episode_length, seed=0)
            reference = fit_dynamics(*[arrays[k][:1] for k in ("obs", "actions", "next_obs")])[0]
        context_len, num_contexts = cfg.context_len, cfg.eval_contexts
        metadata = {
            "training_seed": cfg.seed,
            "algorithm": cfg.agent.name,
            "checkpoint_sha256": file_sha256(args.run / "checkpoint.pkl"),
        }
    else:
        family = args.family
        name = f"{family}_controls"
        context_len, num_contexts = 64, 8
        metadata = {}
    for split in args.splits:
        protocols = ["static", "online"] if agent else ["random", "oracle"]
        for protocol in protocols:
            result = evaluate_continuous(
                agent,
                family,
                split,
                context_len,
                num_contexts,
                online_belief=protocol == "online",
                policy="agent" if agent else protocol,
                action_reference=reference,
            )
            path = args.out / f"{name}_{split}_{protocol}.json"
            if path.exists():
                raise FileExistsError(path)
            atomic_json(
                path,
                {
                    "family": family,
                    "split": split,
                    "protocol": protocol,
                    "adapter": args.adapter,
                    "result": result,
                    "provenance": provenance(),
                    **metadata,
                },
            )
            print(name, split, protocol, result["return"], flush=True)


if __name__ == "__main__":
    main()
