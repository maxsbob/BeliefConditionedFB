"""Verify cached grid transition laws and inspect official AntWind episode continuity."""

import argparse
from pathlib import Path

import numpy as np

from bfb.envs.fourrooms import MOTIONS, make_layout
from bfb.experiment import atomic_json, file_sha256


def audit_grid(path):
    with np.load(path, allow_pickle=False) as data:
        obs, act, nxt, grids = [data[k] for k in ("observations", "actions", "next_observations", "grids")]
        candidate = obs + MOTIONS[act]
        blocked = grids[np.arange(len(grids))[:, None, None], candidate[..., 0], candidate[..., 1]] != 0
        expected = np.where(blocked[..., None], obs, candidate)
        errors = int(np.any(expected != nxt, axis=-1).sum())
        continuity = int(np.any(obs[:, :, 1:] != nxt[:, :, :-1], axis=-1).sum())
        layouts_match = all(np.array_equal(g, make_layout(int(s))) for g, s in zip(grids, data["layout_seeds"]))
        coverage = []
        for i, grid in enumerate(grids):
            visited = np.zeros((*grid.shape, 4), bool)
            visited[obs[i, ..., 0], obs[i, ..., 1], act[i]] = True
            coverage.append(float(visited[grid == 0].mean()))
        if errors or continuity or not layouts_match:
            raise ValueError(f"{path}: invalid transitions/layouts/episode boundaries")
        return {
            "sha256": file_sha256(path),
            "transitions": int(act.size),
            "layouts": len(grids),
            "unique_layouts": len({g.tobytes() for g in grids}),
            "transition_errors": errors,
            "episode_discontinuities": continuity,
            "state_action_coverage_per_layout": coverage,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/data_audit.json"))
    args = parser.parse_args()
    result = {path.name: audit_grid(path) for path in sorted(args.data.glob("fourrooms_*.npz"))}
    for path in sorted((args.data / "antwind_official").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            obs, nxt = data["observation"], data["next_observation"]
            discontinuities = np.max(np.abs(obs[:, 1:] - nxt[:, :-1]), axis=-1) > 1e-6
            result["antwind/" + path.name] = {
                "sha256": file_sha256(path),
                "shape": obs.shape,
                "episode_discontinuities": int(discontinuities.sum()),
                "mean_episode_reward": float(data["reward"].sum(-1).mean()),
                "max_abs_action": float(np.abs(data["actions"]).max()),
            }
    atomic_json(args.out, result)
    print(f"audited {len(result)} datasets: {args.out}")


if __name__ == "__main__":
    main()
