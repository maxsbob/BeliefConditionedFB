"""Randomized Four-Rooms (Appendix C.2).

A discrete, deterministic grid world. Every layout is an 11x11 grid with a border, one horizontal and
one vertical inner wall at random positions and one door in each of the four wall segments, so all four rooms
are connected. The layout is the hidden context; the agent only observes its (row, col) position.

The layout generator reproduces the original implementation bit-for-bit, so layout ``seed=i`` is the same grid
that was used in the paper.

The environment is vectorised in numpy: a batch of episodes (each with its own layout, position and goal) is
stepped at once, which makes evaluation over many layouts/goals/contexts cheap.
"""

from __future__ import annotations

import numpy as np

GRID_SIZE = 11
# Actions: north, south, west, east (row/col deltas).
MOTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
NUM_ACTIONS = len(MOTIONS)
START_POS = (1, 1)


def make_layout(seed: int, size: int = GRID_SIZE) -> np.ndarray:
    """Return a (size, size) int grid, 1 = wall, 0 = free. Identical to the layouts used in the paper."""
    rng = np.random.RandomState(seed)
    grid = np.zeros((size, size), dtype=np.int32)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1

    vertical_wall = rng.randint(low=3, high=size - 3)
    horizontal_wall = rng.randint(low=3, high=size - 3)

    grid[horizontal_wall, :] = 1
    grid[horizontal_wall, rng.randint(low=1, high=vertical_wall)] = 0
    grid[horizontal_wall, rng.randint(low=vertical_wall + 1, high=size - 1)] = 0

    grid[:, vertical_wall] = 1
    grid[rng.randint(low=1, high=horizontal_wall), vertical_wall] = 0
    grid[rng.randint(low=horizontal_wall + 1, high=size - 1), vertical_wall] = 0
    return grid


def eval_goals(size: int = GRID_SIZE) -> np.ndarray:
    """The three fixed evaluation goals, one in each room that does not contain the start (1, 1)."""
    return np.array([(2, size - 3), (size - 3, 2), (size - 3, size - 3)], dtype=np.int64)


def train_layout_seeds(num_layouts: int) -> np.ndarray:
    return np.arange(num_layouts)


def test_layout_seeds(num_layouts: int = 20, first_seed: int = 10_000, exclude: int = 50) -> np.ndarray:
    """A fixed set of held-out layouts.

    The test set is the same for every training configuration (it does not depend on the number of training
    layouts) and never contains a grid identical to one of the first ``exclude`` training layouts.
    """
    train_grids = {make_layout(s).tobytes() for s in range(exclude)}
    seeds, seed = [], first_seed
    while len(seeds) < num_layouts:
        grid = make_layout(seed).tobytes()
        if grid not in train_grids:
            seeds.append(seed)
            train_grids.add(grid)
        seed += 1
    return np.array(seeds)


class VecFourRooms:
    """Batch of independent four-rooms episodes.

    Args:
        grids: (B, H, W) layouts.
        max_steps: episode horizon (100 in the paper).
    """

    def __init__(
        self,
        grids: np.ndarray,
        max_steps: int = 100,
        *,
        slip: float = 0.0,
        action_map=None,
        rng: np.random.Generator | None = None,
    ):
        self.grids = np.asarray(grids)
        self.batch = len(self.grids)
        self.max_steps = max_steps
        self._b = np.arange(self.batch)
        if not 0 <= slip <= 1:
            raise ValueError("slip must be in [0, 1]")
        self.slip = slip
        self.action_map = np.arange(NUM_ACTIONS) if action_map is None else np.asarray(action_map)
        if sorted(self.action_map.tolist()) != list(range(NUM_ACTIONS)):
            raise ValueError("action_map must be a permutation of the four actions")
        self.rng = np.random.default_rng() if rng is None else rng

    def free_cells(self, i: int) -> np.ndarray:
        return np.argwhere(self.grids[i] == 0)

    def reset(self, starts: np.ndarray, goals: np.ndarray | None = None) -> np.ndarray:
        self.pos = np.asarray(starts, dtype=np.int64).copy()
        self.goals = None if goals is None else np.asarray(goals, dtype=np.int64)
        self.t = 0
        self.reached = np.zeros(self.batch, dtype=bool)
        return self.pos.copy()

    def step(self, actions: np.ndarray):
        """Moves into walls leave the agent in place. Returns (next_pos, reward, done)."""
        actions = self.action_map[np.asarray(actions)]
        if self.slip:
            actions = np.where(
                self.rng.random(self.batch) < self.slip, self.rng.integers(NUM_ACTIONS, size=self.batch), actions
            )
        new = self.pos + MOTIONS[actions]
        new = np.clip(new, 0, np.asarray(self.grids.shape[1:]) - 1)
        blocked = self.grids[self._b, new[:, 0], new[:, 1]] == 1
        # Agents that already reached their goal stay there (the episode is over for them).
        stay = blocked | self.reached
        self.pos = np.where(stay[:, None], self.pos, new)
        self.t += 1
        reward = np.zeros(self.batch, dtype=np.float32)
        if self.goals is not None:
            hit = (self.pos == self.goals).all(-1) & ~self.reached
            reward[hit] = 1.0
            self.reached |= hit
        done = self.reached | (self.t >= self.max_steps)
        return self.pos.copy(), reward, done


def random_walk(grids: np.ndarray, starts: np.ndarray, length: int, rng: np.random.Generator, **env_kwargs) -> dict:
    """Uniform random exploration policy. Returns arrays of shape (B, length, ...)."""
    env = VecFourRooms(grids, max_steps=length, rng=rng, **env_kwargs)
    obs = env.reset(starts)
    observations, actions, next_observations = [], [], []
    for _ in range(length):
        a = rng.integers(NUM_ACTIONS, size=env.batch)
        next_obs, _, _ = env.step(a)
        observations.append(obs)
        actions.append(a)
        next_observations.append(next_obs)
        obs = next_obs
    return {
        "observations": np.stack(observations, 1),
        "actions": np.stack(actions, 1),
        "next_observations": np.stack(next_observations, 1),
    }


def random_free_cells(grids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((len(grids), 2), dtype=np.int64)
    for i, g in enumerate(grids):
        free = np.argwhere(g == 0)
        out[i] = free[rng.integers(len(free))]
    return out
