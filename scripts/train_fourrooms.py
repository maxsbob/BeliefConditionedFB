"""Train FB / BFB / RFB on Randomized Four-Rooms and evaluate zero-shot on train and held-out layouts.

Examples:
    uv run scripts/train_fourrooms.py agent=rfb num_train_layouts=30 seed=1
    uv run scripts/train_fourrooms.py agent=bfb context_len=50
    uv run scripts/train_fourrooms.py agent=fb wandb.mode=online
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import trange

from bfb.agent import FBAgent
from bfb.baselines import LatentQAgent
from bfb.data import FourRoomsDataset, load_or_generate
from bfb.diagnostics import belief_means, layout_identification_accuracy
from bfb.envs.fourrooms import NUM_ACTIONS, test_layout_seeds, train_layout_seeds
from bfb.evaluation import evaluate
from bfb.experiment import atomic_json, file_sha256, provenance

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")


class Logger:
    def __init__(self, out_dir, cfg):
        self.f = open(os.path.join(out_dir, "metrics.jsonl"), "a")  # noqa: SIM115 -- closed by Logger.close
        self.wandb = None
        if cfg.wandb.mode != "disabled":
            import wandb

            self.wandb = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                mode=cfg.wandb.mode,
                group=f"{cfg.agent.name}_layouts{cfg.num_train_layouts}_ctx{cfg.context_len}",
                name=f"{cfg.agent.name}_layouts{cfg.num_train_layouts}_ctx{cfg.context_len}_seed{cfg.seed}",
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=out_dir,
            )

    def log(self, metrics: dict, step: int):
        metrics = {k: float(v) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(metrics)
        self.f.write(json.dumps({"step": step, **metrics}) + "\n")
        self.f.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self):
        self.f.close()
        if self.wandb is not None:
            self.wandb.finish()


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig):
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "final.json")):
        raise FileExistsError(f"{out_dir} is already complete; choose a new out_dir")
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)

    data = load_or_generate(
        cfg.data.path, cfg.num_train_layouts, cfg.data.total_transitions, cfg.data.episode_length, cfg.data.seed
    )
    dataset = FourRoomsDataset(data)
    print(
        f"dataset: {dataset.num_layouts} layouts x {dataset.num_episodes} episodes x {dataset.episode_length} "
        f"steps = {dataset.size} transitions"
    )
    if cfg.agent.name in ("hilp", "lap"):
        agent = LatentQAgent(
            cfg.agent, obs_dim=2, num_actions=NUM_ACTIONS, episode_length=dataset.episode_length, seed=cfg.seed
        )
    else:
        agent = FBAgent(cfg.agent, obs_dim=2, num_actions=NUM_ACTIONS, seed=cfg.seed)
    logger = Logger(out_dir, cfg)
    root_key = jax.random.key(cfg.seed + 1)
    encoder_key, bank_key, key = jax.random.split(root_key, 3)
    metadata = {**provenance(), "data_sha256": file_sha256(cfg.data.path)}
    atomic_json(os.path.join(out_dir, "provenance.json"), metadata)

    train_seeds = train_layout_seeds(cfg.num_train_layouts)
    test_seeds = test_layout_seeds(cfg.eval.num_test_layouts, exclude=max(50, cfg.num_train_layouts))

    def run_eval(step):
        res = {}
        for split, seeds in [("train", train_seeds), ("test", test_seeds)]:
            kw = {
                "context_len": cfg.context_len,
                "num_contexts": cfg.eval.num_contexts,
                "max_steps": cfg.eval.max_steps,
                "seed": cfg.eval.get("seed", 20261103),
                "online_belief": cfg.eval.online_belief,
                "task_kappa": cfg.eval.task_kappa,
            }
            res[f"eval/{split}_return"] = evaluate(agent, seeds, **kw)["return"]
            if cfg.eval.random_start:
                res[f"eval/{split}_return_random_start"] = evaluate(agent, seeds, start="random", **kw)["return"]
            # Also report the protocol of the paper: greedy actions and a single (point) belief.
            res[f"eval/{split}_return_greedy_point_belief"] = evaluate(
                agent, seeds, **{**kw, "online_belief": False, "task_kappa": None}
            )["return"]
        logger.log(res, step)
        return res

    # ---------------------------------------------------------------- stage 1: context encoder
    chunk = cfg.log_interval
    bank = None
    if agent.use_context and cfg.agent.get("oracle_context", False):
        # Oracle-ID: h = one-hot index of the training layout (no encoder; held-out layouts get h = 0).
        eye = jnp.eye(dataset.num_layouts, cfg.agent.h_dim)
        bank = (eye[:, None], jnp.full((dataset.num_layouts, 1, cfg.agent.h_dim), -30.0), None)
        agent.oracle_bank = eye
    elif agent.use_context:
        # The encoder only depends on the data, the context length, the encoder config and the seed, so BFB and RFB
        # runs with the same settings share one pre-trained encoder.
        enc_key = hashlib.md5(
            json.dumps(
                {
                    "schema": 2,
                    "data_sha256": metadata["data_sha256"],
                    "encoder": OmegaConf.to_container(cfg.agent.encoder, resolve=True),
                    "h_dim": cfg.agent.h_dim,
                    "data": cfg.data.path,
                    "context_len": cfg.context_len,
                    "steps": cfg.encoder_steps,
                    "batch": cfg.encoder_batch_size,
                    "seed": cfg.seed,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        enc_path = os.path.join(cfg.encoder_cache_dir, f"encoder_{enc_key}.pkl") if cfg.encoder_cache_dir else None
        if enc_path and os.path.exists(enc_path):
            with open(enc_path, "rb") as f:
                agent.enc_state = agent.enc_state.replace(params=jax.device_put(pickle.load(f)))
            print(f"loaded pre-trained encoder {enc_path}")
        else:
            for start in trange(0, cfg.encoder_steps, chunk, desc="encoder", dynamic_ncols=True, mininterval=30):
                count = min(chunk, cfg.encoder_steps - start)
                step = start + count
                sub = jax.random.fold_in(encoder_key, start)
                agent.enc_state, info = agent.train_encoder_chunk(
                    agent.enc_state,
                    dataset.arrays,
                    sub,
                    count,
                    cfg.encoder_batch_size,
                    cfg.context_len,
                    dataset.episode_length,
                )
                logger.log(jax.device_get(info), step - cfg.encoder_steps)  # negative steps = encoder pre-training
            if enc_path:
                os.makedirs(cfg.encoder_cache_dir, exist_ok=True)
                tmp = f"{enc_path}.tmp{os.getpid()}"
                with open(tmp, "wb") as f:
                    pickle.dump(jax.device_get(agent.enc_state.params), f)
                os.replace(tmp, enc_path)
        bank_mu, bank_log_std = agent.build_belief_bank(
            agent.enc_state.params,
            dataset.arrays,
            bank_key,
            cfg.embedding_bank_size,
            cfg.context_len,
            dataset.episode_length,
        )
        agent.h_stats = agent.belief_stats(bank_mu)
        bank = (bank_mu, bank_log_std, agent.h_stats)
        acc = {
            split: layout_identification_accuracy(belief_means(agent, seeds, cfg.context_len, 10))
            for split, seeds in [("train", train_seeds[:20]), ("test", test_seeds)]
        }
        logger.log({"encoder/layout_id_acc_train": acc["train"], "encoder/layout_id_acc_test": acc["test"]}, 0)
        print(f"belief layout-identification accuracy: train {acc['train']:.2f}, test {acc['test']:.2f}")

    # ---------------------------------------------------------------- stage 2: FB (or baseline) training
    if min(chunk, cfg.eval.interval, cfg.batch_size) <= 0 or cfg.batch_size < 2:
        raise ValueError("positive intervals and batch_size >= 2 required")
    if cfg.fb_steps == 0:
        atomic_json(
            os.path.join(out_dir, "final.json"),
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "final": {"step": 0, "encoder_only": True},
                "history": [],
                "provenance": metadata,
            },
        )
        logger.close()
        return
    history = [{"step": 0, **run_eval(0)}] if cfg.eval.get("initial", True) else []
    t0 = time.time()
    pbar = trange(0, cfg.fb_steps, chunk, desc=cfg.agent.name, dynamic_ncols=True, mininterval=30)
    for start in pbar:
        count = min(chunk, cfg.fb_steps - start)
        step = start + count
        sub = jax.random.fold_in(key, start)
        agent.state, info = agent.train_chunk(agent.state, dataset.arrays, bank, sub, count, cfg.batch_size)
        info = jax.device_get(info)
        info["time/steps_per_sec"] = step / (time.time() - t0)
        logger.log(info, step)
        if step // cfg.eval.interval > start // cfg.eval.interval or step == cfg.fb_steps:
            res = run_eval(step)
            history.append({"step": step, **res})
            pbar.set_postfix(train=f"{res['eval/train_return']:.2f}", test=f"{res['eval/test_return']:.2f}")

    final = history[-1] if history else {"step": 0, **run_eval(0)}
    if cfg.save_checkpoint:
        agent.save(os.path.join(out_dir, "checkpoint.pkl"))
    atomic_json(
        os.path.join(out_dir, "final.json"),
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "final": final,
            "history": history,
            "provenance": metadata,
        },
    )
    logger.close()
    print(f"final: train={final['eval/train_return']:.3f} test={final['eval/test_return']:.3f}")


if __name__ == "__main__":
    main()
