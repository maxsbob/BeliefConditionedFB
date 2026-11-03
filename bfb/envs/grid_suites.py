"""Declared grid shift families. Only `iid` is the paper's environment distribution."""

from collections import deque

import numpy as np

from bfb.envs.fourrooms import GRID_SIZE, MOTIONS, START_POS, eval_goals, make_layout, test_layout_seeds

SUITES = ("iid", "slip", "remap", "obstacles")


def distances(grid, goal):
    dist = np.full(grid.shape, np.inf)
    dist[tuple(goal)] = 0
    queue = deque([tuple(goal)])
    while queue:
        cell = queue.popleft()
        for delta in MOTIONS:
            nxt = tuple(np.asarray(cell) + delta)
            if min(nxt) < 0 or max(nxt) >= len(grid):
                continue
            if grid[nxt] == 0 and not np.isfinite(dist[nxt]):
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def obstacle_layout(seed):
    """Connected random obstacles: a structural shift from four axis-aligned rooms."""
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        grid = (rng.random((GRID_SIZE, GRID_SIZE)) < 0.22).astype(np.int32)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
        for pos in [START_POS, *eval_goals()]:
            grid[tuple(pos)] = 0
        if np.isfinite(distances(grid, START_POS)[grid == 0]).all():
            return grid
    raise RuntimeError("could not generate a connected obstacle layout")


def grid_suite(name, num_layouts=20, split="test", exclude=50):
    if name not in SUITES or split not in ("test", "validation"):
        raise ValueError((name, split))
    seeds = test_layout_seeds(num_layouts, first_seed=10_000 if split == "test" else 20_000, exclude=exclude)
    generator = obstacle_layout if name == "obstacles" else make_layout
    grids = np.stack([generator(int(s)) for s in seeds])
    kwargs = {"slip": 0.25} if name == "slip" else {"action_map": [3, 2, 0, 1]} if name == "remap" else {}
    return seeds, grids, kwargs
