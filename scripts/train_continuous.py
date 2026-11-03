"""Train continuous FB/BFB/RFB and measure interpolation versus explicit dynamics extrapolation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import hydra
import jax
import numpy as np
from omegaconf import OmegaConf

from bfb.continuous import ContinuousFBAgent
from bfb.continuous_evaluation import evaluate_continuous
from bfb.envs.pointmass import generate_dataset
from bfb.experiment import ROOT, atomic_json, provenance, restore_training, save_training, training_fingerprint


@hydra.main(version_base="1.3", config_path=str(ROOT / "configs"), config_name="continuous")
def main(cfg):
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "data" / "jax_cache"))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.json").exists():
        raise FileExistsError(f"{out} already completed")
    metadata = provenance()
    config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.family == "antwind":
        from bfb.antwind import evaluate_antwind, load_antwind

        host_arrays, normalization, labels, hashes = load_antwind(cfg.data_path, cfg.num_train_contexts, cfg.episodes)
        metadata["official_data_sha256"] = hashes
        np.savez(out / "normalization.npz", **normalization)
    else:
        host_arrays = generate_dataset(cfg.family, cfg.num_train_contexts, cfg.episodes, cfg.episode_length, seed=0)
    metadata["data_sha256"] = hashlib.sha256(
        b"".join(host_arrays[k].tobytes() for k in sorted(host_arrays))
    ).hexdigest()
    arrays = jax.device_put(host_arrays)
    agent = ContinuousFBAgent(cfg.agent, arrays["obs"].shape[-1], action_dim=arrays["actions"].shape[-1], seed=cfg.seed)
    fingerprint = training_fingerprint(config, metadata["data_sha256"])
    resumed_history = restore_training(out / "training_state.pkl", agent, fingerprint)
    atomic_json(out / "config.json", config)
    atomic_json(out / "provenance.json", metadata)
    enc_key, bank_key, fb_key = jax.random.split(jax.random.key(cfg.seed + 1), 3)
    bank = None
    started = time.monotonic()
    log = (out / "metrics.jsonl").open("a")

    def record(step, info):
        metrics = {k: float(v) for k, v in jax.device_get(info).items()}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(metrics)
        log.write(json.dumps({"step": step, "elapsed_seconds": time.monotonic() - started, **metrics}) + "\n")
        log.flush()
        print(step, metrics, flush=True)

    if agent.use_context:
        cache_config = {
            "family": cfg.family,
            "num_contexts": cfg.num_train_contexts,
            "episodes": cfg.episodes,
            "length": cfg.episode_length,
            "context_len": cfg.context_len,
            "encoder_steps": cfg.encoder_steps,
            "encoder_batch": cfg.encoder_batch_size,
            "encoder": config["agent"]["encoder"],
            "seed": cfg.seed,
            "h_dim": cfg.agent.h_dim,
            "data": metadata["data_sha256"],
            "schema": 1,
        }
        digest = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:16]
        cache = ROOT / "data" / "continuous_encoders" / f"{digest}.pkl"
        if resumed_history is not None:
            print(f"resuming at FB step {int(agent.state.step)}", flush=True)
        elif cache.exists():
            with cache.open("rb") as f:
                agent.enc_state = agent.enc_state.replace(params=jax.device_put(pickle.load(f)))
        else:
            for start in range(0, cfg.encoder_steps, cfg.chunk):
                count = min(cfg.chunk, cfg.encoder_steps - start)
                agent.enc_state, info = agent.train_encoder_chunk(
                    agent.enc_state,
                    arrays,
                    jax.random.fold_in(enc_key, start),
                    count,
                    cfg.encoder_batch_size,
                    cfg.context_len,
                    cfg.episode_length,
                )
                record(start + count - cfg.encoder_steps, info)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(f".tmp{os.getpid()}")
            with temporary.open("wb") as f:
                pickle.dump(jax.device_get(agent.enc_state.params), f)
            os.replace(temporary, cache)
        mu, log_std = agent.build_belief_bank(
            agent.enc_state.params, arrays, bank_key, cfg.embedding_bank_size, cfg.context_len, cfg.episode_length
        )
        agent.h_stats = agent.belief_stats(mu)
        bank = (mu, log_std, agent.h_stats)

    history = [] if resumed_history is None else resumed_history
    for start in range(int(agent.state.step), cfg.fb_steps, cfg.chunk):
        count = min(cfg.chunk, cfg.fb_steps - start)
        agent.state, info = agent.train_chunk(
            agent.state, arrays, bank, jax.random.fold_in(fb_key, start), count, cfg.batch_size
        )
        record(start + count, info)
        if (start + count) // cfg.eval_interval > start // cfg.eval_interval or start + count == cfg.fb_steps:
            # Only validation feedback during training; held-out tests run once at the fixed final step below.
            if cfg.family == "antwind":
                result = evaluate_antwind(
                    agent,
                    cfg.data_path,
                    normalization,
                    labels,
                    "train",
                    cfg.context_len,
                    num_contexts=1,
                    seed=cfg.eval_seed,
                )
                metric = "train_return"
            else:
                result = evaluate_continuous(
                    agent, cfg.family, "validation", cfg.context_len, num_contexts=2, seed=cfg.eval_seed
                )
                metric = "validation_return"
            history.append({"step": start + count, metric: result["return"]})
            print("validation", history[-1], flush=True)
            agent.save(out / "checkpoint.pkl")
            save_training(out / "training_state.pkl", agent, fingerprint, history)
    evaluations = {}
    for split in ("train", "test") if cfg.family == "antwind" else ("train", "iid", "rotation", "gain"):
        for protocol in ("static", "online", "shuffled"):
            if cfg.family == "antwind":
                result = evaluate_antwind(
                    agent,
                    cfg.data_path,
                    normalization,
                    labels,
                    split,
                    cfg.context_len,
                    cfg.eval_contexts,
                    seed=cfg.eval_seed,
                    online_belief=protocol != "static",
                    context_mode="shuffled" if protocol == "shuffled" else "correct",
                )
            else:
                result = evaluate_continuous(
                    agent,
                    cfg.family,
                    split,
                    cfg.context_len,
                    cfg.eval_contexts,
                    num_dynamics=cfg.num_train_contexts,
                    seed=cfg.eval_seed,
                    online_belief=protocol != "static",
                    context_mode="shuffled" if protocol == "shuffled" else "correct",
                )
            evaluations[f"{split}/{protocol}"] = result
            print(split, protocol, result["return"], flush=True)
    agent.save(out / "checkpoint.pkl")
    atomic_json(
        out / "final.json",
        {
            "config": config,
            "provenance": metadata,
            "history": history,
            "evaluations": evaluations,
            "seconds": time.monotonic() - started,
        },
    )
    log.close()


if __name__ == "__main__":
    main()
