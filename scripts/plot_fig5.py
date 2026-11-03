"""Aggregate runs produced by scripts/launch.py and plot Fig. 5 (Randomized Four-Rooms part) + a results table.

uv run scripts/plot_fig5.py --root runs/fig5 --out figures/fig5_fourrooms.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "rfb": {"label": "RFB", "color": "#d95f02", "lw": 2.2},
    "bfb": {"label": "BFB", "color": "#fc8d62", "lw": 2.0},
    "hilp": {"label": "HILP", "color": "#4d4d4d", "lw": 1.6},
    "lap": {"label": "LAP", "color": "#878787", "lw": 1.6},
    "fb": {"label": "FB", "color": "#bababa", "lw": 1.6},
}


def load(root):
    """{(agent, layouts, ctx): {'train': [...], 'test': [...]}} with one entry per seed (final evaluation)."""
    out = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(root, "**", "final.json"), recursive=True):
        with open(path) as f:
            run = json.load(f)
        cfg, final = run["config"], run["final"]
        if final.get("encoder_only"):
            continue
        key = (cfg["agent"]["name"], cfg["num_train_layouts"], cfg["context_len"])
        out[key]["train"].append(final["eval/train_return"])
        out[key]["test"].append(final["eval/test_return"])
    return out


def random_policy_return(layouts, num_episodes_per_goal=50, seed=0):
    from bfb.envs.fourrooms import NUM_ACTIONS, START_POS, VecFourRooms, eval_goals, make_layout

    rng = np.random.default_rng(seed)
    goals = eval_goals()
    grids = np.repeat(np.stack([make_layout(s) for s in layouts]), len(goals) * num_episodes_per_goal, 0)
    goal_pos = np.tile(np.repeat(goals, num_episodes_per_goal, 0), (len(layouts), 1))
    env = VecFourRooms(grids)
    env.reset(np.tile(np.array(START_POS), (len(grids), 1)), goal_pos)
    ret = np.zeros(len(grids))
    for _ in range(100):
        _, r, _ = env.step(rng.integers(NUM_ACTIONS, size=len(grids)))
        ret += r
    return ret.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/fig5")
    parser.add_argument("--out", default="figures/fig5_fourrooms.png")
    parser.add_argument("--ablation-layouts", type=int, default=20, help="number of layouts in the context ablation")
    args = parser.parse_args()
    data = load(args.root)
    if not data:
        raise SystemExit(f"no finished runs under {args.root}")

    from bfb.envs.fourrooms import test_layout_seeds, train_layout_seeds

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharey="row")
    lines = ["| method | layouts | context | train | test | seeds |", "|---|---|---|---|---|---|"]
    for row, split in enumerate(["train", "test"]):
        # Left: number of training layouts (context 100).
        ax = axes[row, 0]
        for agent, style in STYLE.items():
            pts = sorted((n, v[split]) for (a, n, c), v in data.items() if a == agent and c == 100)
            if not pts:
                continue
            x = [p[0] for p in pts]
            m = np.array([np.mean(p[1]) for p in pts])
            s = np.array([np.std(p[1]) for p in pts])
            ax.plot(x, m, marker="o", ms=4, **style)
            ax.fill_between(x, m - s, m + s, color=style["color"], alpha=0.2)
        xs = sorted({n for (a, n, c) in data})
        rnd = [random_policy_return(train_layout_seeds(n) if split == "train" else test_layout_seeds()) for n in xs]
        ax.plot(xs, rnd, ls="--", color="#1b9e77", label="Random")
        ax.set_ylabel(f"{split.capitalize()} Return")
        # Right: context length (BFB / RFB).
        ax = axes[row, 1]
        for agent in ["rfb", "bfb"]:
            pts = sorted((c, v[split]) for (a, n, c), v in data.items() if a == agent and n == args.ablation_layouts)
            if len(pts) < 2:
                continue
            x = [p[0] for p in pts]
            m = np.array([np.mean(p[1]) for p in pts])
            s = np.array([np.std(p[1]) for p in pts])
            ax.plot(x, m, marker="o", ms=4, **STYLE[agent])
            ax.fill_between(x, m - s, m + s, color=STYLE[agent]["color"], alpha=0.2)
        for a in axes[row]:
            a.set_ylim(-0.02, 1.02)
            a.grid(alpha=0.3)
    axes[1, 0].set_xlabel("# of training layouts")
    axes[1, 1].set_xlabel("Context length")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Randomized Four-Rooms")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)

    for (a, n, c), v in sorted(data.items()):
        lines.append(
            f"| {STYLE.get(a, {'label': a})['label']} | {n} | {c} | {np.mean(v['train']):.2f} ± {np.std(v['train']):.2f}"
            f" | {np.mean(v['test']):.2f} ± {np.std(v['test']):.2f} | {len(v['train'])} |"
        )
    table = "\n".join(lines)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
