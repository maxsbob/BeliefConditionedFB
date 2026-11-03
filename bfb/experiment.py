"""Small, shared helpers for provenance, atomic results, and inference checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import subprocess
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for folder in ("bfb", "scripts", "configs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in (".py", ".yaml"):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance():
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hash(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "devices": [f"{device}: {device.device_kind}" for device in jax.devices()],
        "uv_lock_sha256": file_sha256(ROOT / "uv.lock"),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")

    def convert(value):
        if isinstance(value, (np.ndarray, jax.Array)):
            return np.asarray(value).tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    tmp.write_text(json.dumps(payload, indent=2, default=convert, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_agent(run_dir):
    """Load an inference checkpoint (not a resumable training state). Never imports plotting code."""
    from bfb.agent import FBAgent
    from bfb.baselines import LatentQAgent
    from bfb.envs.fourrooms import NUM_ACTIONS

    run_dir = Path(run_dir)
    cfg = OmegaConf.create(json.loads((run_dir / "final.json").read_text())["config"])
    defaults_name = cfg.agent.name if cfg.agent.name in ("hilp", "lap") else "fb"
    defaults = OmegaConf.load(ROOT / "configs" / "agent" / f"{defaults_name}.yaml")
    cfg.agent = OmegaConf.merge(defaults, cfg.agent)
    if cfg.agent.name in ("hilp", "lap"):
        agent = LatentQAgent(cfg.agent, 2, NUM_ACTIONS, cfg.data.episode_length, cfg.seed)
    else:
        agent = FBAgent(cfg.agent, 2, NUM_ACTIONS, cfg.seed)
    agent.load(run_dir / "checkpoint.pkl")
    return agent, cfg


def training_fingerprint(config, data_hash):
    """All learning/data settings must stay unchanged when resuming interrupted training."""
    settings = {k: v for k, v in config.items() if k not in ("out_dir", "fb_steps")}
    return hashlib.sha256(
        json.dumps({"training_schema": 2, "config": settings, "data": data_hash}, sort_keys=True).encode()
    ).hexdigest()


def save_training(path, agent, fingerprint, history):
    path = Path(path)
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    payload = {
        "fingerprint": fingerprint,
        "state": agent.state,
        "encoder": agent.enc_state,
        "h_stats": agent.h_stats,
        "history": history,
    }
    with temporary.open("wb") as f:
        pickle.dump(jax.device_get(payload), f)
    os.replace(temporary, path)


def restore_training(path, agent, fingerprint):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload["fingerprint"] != fingerprint:
        raise ValueError(f"{path}: resume configuration/data mismatch; use a new output directory")
    agent.state = jax.device_put(payload["state"])
    agent.enc_state = jax.device_put(payload["encoder"])
    agent.h_stats = jax.device_put(payload["h_stats"])
    return payload["history"]
