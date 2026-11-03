"""Create seed-level tables, paired comparisons, a figure, and compact auditable evidence.

Only completed experiments are read. Each training seed contributes one score, regardless of its trial count.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bfb.experiment import atomic_json, file_sha256


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(differences, rng, replicates=5000):
    """Paired bootstrap over training seeds, then shared layout/context IDs within each sampled seed."""
    values = np.stack(differences)
    seeds, layouts = values.shape
    seed_ids = rng.integers(seeds, size=(replicates, seeds))
    layout_ids = rng.integers(layouts, size=(replicates, seeds, layouts))
    samples = values[seed_ids[..., None], layout_ids].mean((1, 2))
    return np.quantile(samples, [0.025, 0.975])


def summarize(root=Path("."), out=Path("results"), archive=None):
    out.mkdir(parents=True, exist_ok=True)
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    records = []
    evidence = []
    sweep_records = []

    def artifacts(folder, pattern, recursive=False):
        if archive is not None:
            prefix = Path(folder)
            selected = []
            for item in archive:
                path = Path(item["path"])
                if not path.is_relative_to(prefix):
                    continue
                if not recursive and path.parent != prefix:
                    continue
                if path.match(pattern):
                    selected.append(item)
            yield from sorted(selected, key=lambda item: item["path"])
        else:
            directory = root / folder
            paths = directory.rglob(pattern) if recursive else directory.glob(pattern)
            for path in sorted(paths):
                yield {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "data": json.loads(path.read_text()),
                }

    def add_sweep(data):
        if (
            data["algorithm"] in ("bfb", "rfb")
            and data["benchmark"] == "iid"
            and data["protocol"] in ("greedy_static", "sampled_online")
        ):
            sweep_records.append(
                {
                    "algorithm": data["algorithm"],
                    "num_train_layouts": data["num_train_layouts"],
                    "context_len": data["context_len"],
                    "protocol": data["protocol"],
                    "seed": data["training_seed"],
                    "score": data["result"]["return"],
                }
            )

    # Exclude exploratory validation runs and duplicate preliminary seed-0 audits.
    for folder in ("grid_full", "grid_adapted", "grid_baselines", "grid_baselines_adapted", "grid_compact"):
        for item in artifacts(f"results/{folder}", "*.json"):
            data = item["data"]
            if data["split"] != "test":
                continue
            result = data["result"]
            records.append(
                {
                    "benchmark": "grid_" + data["benchmark"],
                    "split": "test",
                    "algorithm": data["algorithm"],
                    "protocol": data["protocol"],
                    "seed": data["training_seed"],
                    "score": result["return"],
                    "per_context": np.array(result["per_layout"]),
                    "pairing": (data["evaluation_seed"], data["num_contexts"], data["grid_sha256"]),
                }
            )
            evidence.append(item)
            add_sweep(data)
    for item in artifacts("results/grid_sweeps", "*.json"):
        data = item["data"]
        add_sweep(data)
        evidence.append(item)
    sweep_groups = defaultdict(list)
    for item in sweep_records:
        key = tuple(item[k] for k in ("algorithm", "num_train_layouts", "context_len", "protocol"))
        sweep_groups[key].append(item)
    sweep_rows = []
    for key, items in sorted(sweep_groups.items()):
        if len({i["seed"] for i in items}) != len(items):
            raise ValueError(f"duplicate sweep seeds for {key}")
        scores = [i["score"] for i in items]
        sweep_rows.append(
            dict(
                zip(("algorithm", "num_train_layouts", "context_len", "protocol"), key),
                n_seeds=len(items),
                mean=float(np.mean(scores)),
                seed_std=float(np.std(scores, ddof=1)) if len(items) > 1 else "",
            )
        )
    write_csv(out / "grid_sweeps.csv", sweep_rows)
    for folder in ("continuous", "antwind", "antwind_bc"):
        for item in artifacts(f"runs/{folder}", "final.json", recursive=True):
            data = item["data"]
            cfg = data["config"]
            name = cfg["agent"]["name"] + ("_bc1" if folder == "antwind_bc" else "")
            for key, result in data["evaluations"].items():
                split, protocol = key.split("/")
                per_context = result.get("per_dynamics", np.mean(result.get("returns", [[0]]), axis=-1))
                records.append(
                    {
                        "benchmark": cfg["family"],
                        "split": split,
                        "algorithm": name,
                        "protocol": protocol,
                        "seed": cfg["seed"],
                        "score": result["return"],
                        "per_context": np.array(per_context),
                        "pairing": (cfg["eval_seed"], cfg["eval_contexts"], cfg["num_train_contexts"]),
                    }
                )
            evidence.append(item)
    for folder in ("continuous_adapted", "continuous_controls", "antwind_controls"):
        for item in artifacts(f"results/{folder}", "*.json"):
            data = item["data"]
            result = data["result"]
            name = data.get("algorithm", data["protocol"])
            if data.get("adapter"):
                name += "_affine"
            records.append(
                {
                    "benchmark": data.get("family", data.get("benchmark")),
                    "split": data["split"],
                    "algorithm": name,
                    "protocol": data["protocol"],
                    "seed": data.get("training_seed"),
                    "score": result["return"],
                    "per_context": np.array(result.get("per_dynamics", np.mean(result.get("returns", [[0]]), axis=-1))),
                    "pairing": (20261103, 8, 16),
                }
            )
            evidence.append(item)

    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in ("benchmark", "split", "algorithm", "protocol"))].append(record)
    rows = []
    for key, items in sorted(groups.items()):
        seeds = [r["seed"] for r in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seed/config records for {key}: {seeds}")
        scores = [r["score"] for r in items]
        rows.append(
            dict(
                zip(("benchmark", "split", "algorithm", "protocol"), key),
                n_seeds=len(scores),
                mean=float(np.mean(scores)),
                seed_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else "",
                seeds=",".join(str(s) for s in seeds),
            )
        )
    write_csv(out / "summary.csv", rows)
    write_csv(
        out / "seed_scores.csv",
        [{k: r[k] for k in ("benchmark", "split", "algorithm", "protocol", "seed", "score")} for r in records],
    )
    comparisons = []
    rng = np.random.default_rng(0)
    for key, items in sorted(groups.items()):
        bench, split, method, protocol = key
        suffix = "_bc1" if method.endswith("_bc1") else "_affine" if method.endswith("_affine") else ""
        base_method = method.removesuffix(suffix) if suffix else method
        if base_method not in ("bfb", "rfb"):
            continue
        references = ["fb" + suffix] + (["bfb" + suffix] if base_method == "rfb" else [])
        for reference in references:
            other = {r["seed"]: r for r in groups.get((bench, split, reference, protocol), [])}
            pairs = [(r, other[r["seed"]]) for r in items if r["seed"] in other]
            if len(pairs) < 2:
                continue
            if any(a["pairing"] != b["pairing"] for a, b in pairs):
                raise ValueError(f"incompatible paired evaluation: {key}/{reference}")
            differences = [a["per_context"] - b["per_context"] for a, b in pairs]
            low, high = paired_interval(differences, rng)
            comparisons.append(
                {
                    "benchmark": bench,
                    "split": split,
                    "protocol": protocol,
                    "comparison": f"{method}-{reference}",
                    "paired_seeds": len(pairs),
                    "mean_difference": float(np.mean(differences)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(out / "paired_comparisons.csv", comparisons)
    # Deterministic gzip header; raw trial arrays/configurations/source fingerprints remain available.
    with (
        (evidence_dir / "experiments.jsonl.gz").open("wb") as file,
        gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed,
    ):
        for item in evidence:
            compressed.write((json.dumps(item, separators=(",", ":")) + "\n").encode())
    atomic_json(
        out / "manifest.json",
        {
            "files": len(evidence),
            "groups": len(rows),
            "evidence_sha256": file_sha256(evidence_dir / "experiments.jsonl.gz"),
            "uncertainty": "sample SD across training seeds; n=1 has no SD",
            "paired_CI": "paired hierarchical bootstrap over seeds and dynamics; 5000 draws",
        },
    )
    return rows


def plot(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    specs = [
        ("grid_iid", "test", ["greedy_static", "sampled_online", "guarded_online"], "Four-Rooms (held out)"),
        ("pointmass", "rotation", ["static", "online"], "Continuous: unseen rotation"),
        ("inertial", "rotation", ["static", "online"], "Inertial: unseen rotation"),
    ]
    colors = {"fb": "#666d75", "bfb": "#2878b5", "rfb": "#e58d35"}
    for ax, (benchmark, split, protocols, title) in zip(axes, specs):
        width = 0.24
        for j, method in enumerate(colors):
            selected = {
                r["protocol"]: r
                for r in rows
                if r["benchmark"] == benchmark and r["split"] == split and r["algorithm"] == method
            }
            for i, protocol in enumerate(protocols):
                if protocol not in selected:
                    continue
                r = selected[protocol]
                ax.bar(
                    i + (j - 1) * width,
                    r["mean"],
                    width,
                    color=colors[method],
                    label=method.upper() if i == 0 else None,
                    yerr=r["seed_std"] if r["seed_std"] != "" else None,
                    capsize=3,
                )
        ax.set_xticks(range(len(protocols)), [p.replace("_", "\n") for p in protocols], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Success rate (mean ± seed SD)")
    axes[0].legend(frameon=False, fontsize=9)
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--from-evidence", type=Path, help="Regenerate tables from a saved experiments.jsonl.gz archive"
    )
    args = parser.parse_args()
    archive = None
    if args.from_evidence:
        with gzip.open(args.from_evidence, "rt") as file:
            archive = [json.loads(line) for line in file]
        paths = [item["path"] for item in archive]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate source paths in evidence archive")
    rows = summarize(args.root, args.out, archive)
    plot(rows, args.root / "figures" / "reproduction.png")
    sweeps_path = args.out / "grid_sweeps.csv"
    sweeps = []
    if sweeps_path.exists():
        with sweeps_path.open() as file:
            sweeps = list(csv.DictReader(file))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained", sharey=True)
    for row, protocol in enumerate(("greedy_static", "sampled_online")):
        for col, x_name in enumerate(("num_train_layouts", "context_len")):
            ax = axes[row, col]
            for method, color in [("bfb", "#2878b5"), ("rfb", "#e58d35")]:
                selected = [
                    r
                    for r in sweeps
                    if r["algorithm"] == method
                    and r["protocol"] == protocol
                    and (int(r["context_len"]) == 100 if col == 0 else int(r["num_train_layouts"]) == 20)
                ]
                selected.sort(key=lambda r: int(r[x_name]))
                ax.errorbar(
                    [int(r[x_name]) for r in selected],
                    [float(r["mean"]) for r in selected],
                    yerr=[float(r["seed_std"] or 0) for r in selected],
                    marker="o",
                    capsize=3,
                    color=color,
                    label=method.upper(),
                )
            ax.set_xlabel("Training layouts" if col == 0 else "Context transitions")
            ax.set_title(protocol.replace("_", " "))
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_ylabel("Held-out success")
    axes[1, 0].set_ylabel("Held-out success")
    fig.savefig(args.root / "figures" / "grid_sweeps.png", dpi=180)
    plt.close(fig)
    print(f"{len(rows)} result groups; {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
