"""Observation-only inference adapters for grid action symmetries and observed collisions."""

import itertools

import numpy as np

from bfb.envs.fourrooms import GRID_SIZE, MOTIONS

PERMUTATIONS = np.array(list(itertools.permutations(range(4))))


def infer_action_map(observations, actions, next_observations):
    """Maximum-likelihood permutation from nonzero displacements; ties prefer the identity.

    No layout, simulator action map, reward or task information is accepted. Under slip this is a majority vote
    with a bijection constraint. Unobserved actions are resolved by the constraint and deterministic tie breaking.
    """
    delta = np.asarray(next_observations) - np.asarray(observations)
    direction_match = (delta[..., None, :] == MOTIONS).all(-1)
    action_match = np.asarray(actions)[..., None] == np.arange(4)
    counts = np.einsum("bta,btd->bad", action_match.astype(np.float32), direction_match.astype(np.float32))
    scores = counts[:, np.arange(4)[None, :], PERMUTATIONS].sum(-1)
    return PERMUTATIONS[scores.argmax(-1)]


class CollisionMemory:
    """Count observed successes/failures of each state-action pair. Unknown pairs remain available."""

    def __init__(self, batch):
        self.blocked = np.zeros((batch, GRID_SIZE, GRID_SIZE, 4), dtype=np.int32)
        self.moved = np.zeros_like(self.blocked)

    def update(self, obs, actions, nxt):
        if obs.ndim == 2:
            obs, actions, nxt = obs[:, None], actions[:, None], nxt[:, None]
        idx = (np.arange(len(obs))[:, None], obs[..., 0], obs[..., 1], actions)
        moved = np.any(obs != nxt, axis=-1)
        np.add.at(self.blocked, idx, ~moved)
        np.add.at(self.moved, idx, moved)

    def mask(self, positions):
        idx = (np.arange(len(positions)), positions[:, 0], positions[:, 1])
        # One self-loop is enough in deterministic worlds; any observed motion re-opens the action.
        forbidden = (self.blocked[idx] > 0) & (self.moved[idx] == 0)
        return forbidden & ~forbidden.all(-1, keepdims=True)
