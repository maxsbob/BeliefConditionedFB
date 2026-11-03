"""Vectorized continuous-control diagnostics, not the paper's MuJoCo PointMass.

Position is observed; actuator rotation/gain and wind are hidden. A second family adds inertia and observes
velocity too. Splits deliberately distinguish interpolation from held-out rotation quadrants and gain ranges.
No simulator internals are exposed to the learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dynamics:
    angle: float
    gain: float
    wind_x: float
    wind_y: float
    damping: float = 0.0


def contexts(family="pointmass", split="train", count=16):
    if family not in ("pointmass", "inertial") or split not in ("train", "validation", "iid", "rotation", "gain"):
        raise ValueError((family, split))
    # A fixed lattice makes exclusions inspectable and independent of training randomness.
    phase = {"train": 0.0, "validation": 0.25, "iid": 0.5, "rotation": 0.5, "gain": 0.5}[split]
    result = []
    for i in range(count):
        t = (i + phase) / count
        angle = (1.5 * np.pi + 0.5 * np.pi * t) if split == "rotation" else 1.5 * np.pi * t
        gain = (0.45 + 0.15 * ((i % 4) / 3)) if split == "gain" else (0.85 + 0.3 * ((i % 4) / 3))
        wind_angle = 2 * np.pi * ((i * 7 + phase) % count) / count
        damping = 0.65 + 0.2 * ((i % 3) / 2) if family == "inertial" else 0.0
        result.append(Dynamics(angle, gain, 0.12 * np.cos(wind_angle), 0.12 * np.sin(wind_angle), damping))
    return result


class PointMass:
    step_scale = 0.1

    def __init__(self, dynamics, family="pointmass"):
        self.family = family
        self.batch = len(dynamics)
        self.obs_dim = 4 if family == "inertial" else 2
        self.angle = np.array([d.angle for d in dynamics], dtype=np.float32)
        self.gain = np.array([d.gain for d in dynamics], dtype=np.float32)[:, None]
        self.wind = np.array([[d.wind_x, d.wind_y] for d in dynamics], dtype=np.float32)
        self.damping = np.array([d.damping for d in dynamics], dtype=np.float32)[:, None]
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.rotation = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)

    def reset(self, positions, velocity=None):
        self.pos = np.asarray(positions, dtype=np.float32).copy()
        self.velocity = np.zeros_like(self.pos) if velocity is None else np.asarray(velocity, dtype=np.float32).copy()
        return self.observation()

    def observation(self):
        return self.pos.copy() if self.obs_dim == 2 else np.concatenate([self.pos, self.velocity], -1)

    def step(self, actions):
        force = self.gain * np.einsum("bij,bj->bi", self.rotation, np.clip(actions, -1, 1)) + self.wind
        self.velocity = self.damping * self.velocity + (1 - self.damping) * force
        proposed = self.pos + self.step_scale * self.velocity
        self.pos = np.clip(proposed, -1, 1).astype(np.float32)
        if self.obs_dim == 4:
            self.velocity = np.where(np.abs(proposed) > 1, 0.0, self.velocity).astype(np.float32)
        return self.observation()


def generate_dataset(family="pointmass", num_contexts=16, episodes=256, length=64, seed=0):
    """Reward-free uniform random actions and random resets; no task labels or privileged state in observations."""
    rng = np.random.default_rng(seed)
    dynamics = [d for d in contexts(family, "train", num_contexts) for _ in range(episodes)]
    env = PointMass(dynamics, family)
    obs = env.reset(rng.uniform(-0.95, 0.95, (len(dynamics), 2)))
    observations, actions, next_observations = [], [], []
    for _ in range(length):
        action = rng.uniform(-1, 1, (len(dynamics), 2)).astype(np.float32)
        nxt = env.step(action)
        observations.append(obs)
        actions.append(action)
        next_observations.append(nxt)
        obs = nxt
    return {
        k: np.stack(v, 1).reshape(num_contexts, episodes * length, -1)
        for k, v in [("obs", observations), ("actions", actions), ("next_obs", next_observations)]
    }
