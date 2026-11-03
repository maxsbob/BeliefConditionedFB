"""Render the reproduction report from completed, seed-level result tables."""

import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "results" / "summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    results = {(r["benchmark"], r["split"], r["algorithm"], r["protocol"]): r for r in rows}

    def score(benchmark, split, method, protocol, digits=3, seeds=3):
        row = results[(benchmark, split, method, protocol)]
        if int(row["n_seeds"]) != seeds:
            raise ValueError(f"incomplete results: {benchmark}/{split}/{method}/{protocol}")
        mean = f"{float(row['mean']):.{digits}f}"
        if seeds == 1:
            return mean + " (1 seed)"
        return mean + f" ± {float(row['seed_std']):.{digits}f}"

    def table(header, entries):
        return "\n".join(
            [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * len(header)) + " |",
                *["| " + " | ".join(entry) + " |" for entry in entries],
            ]
        )

    grid = table(
        ["Protocol / shift", "FB", "BFB", "RFB"],
        [
            [label, *[score(bench, "test", method, protocol) for method in ("fb", "bfb", "rfb")]]
            for label, bench, protocol in [
                ("Held-out rooms; greedy, static belief", "grid_iid", "greedy_static"),
                ("Held-out rooms; sampled, online belief", "grid_iid", "sampled_online"),
                ("Held-out rooms; calibrated + collision memory", "grid_iid", "guarded_online"),
                ("Unseen action remap; sampled, online", "grid_remap", "sampled_online"),
                ("Unseen action remap; calibrated + memory", "grid_remap", "guarded_online"),
                ("Random obstacles; calibrated + memory", "grid_obstacles", "guarded_online"),
                ("Action slip; sampled, online", "grid_slip", "sampled_online"),
            ]
        ],
    )
    continuous = table(
        ["Dynamics / split (online beliefs)", "FB", "BFB", "RFB"],
        [
            [family + " / " + split, *[score(family, split, method, "online") for method in ("fb", "bfb", "rfb")]]
            for family in ("pointmass", "inertial")
            for split in ("iid", "rotation", "gain")
        ],
    )
    adapted = table(
        ["Held-out rotation; affine adapter", "FB", "BFB", "RFB"],
        [
            [family, *[score(family, "rotation", method + "_affine", "online") for method in ("fb", "bfb", "rfb")]]
            for family in ("pointmass", "inertial")
        ],
    )
    ant = table(
        ["Actor regularizer / protocol", "FB", "BFB", "RFB"],
        [
            [
                label,
                *[
                    score("antwind", "test", method + suffix, protocol, digits=1, seeds=n)
                    for method in ("fb", "bfb", "rfb")
                ],
            ]
            for label, suffix, protocol, n in [
                ("Weak BC (0.03), static belief; exploratory", "", "static", 1),
                ("BC 1.0, static belief", "_bc1", "static", 3),
                ("BC 1.0, online belief", "_bc1", "online", 3),
                ("BC 1.0, shuffled belief", "_bc1", "shuffled", 3),
            ]
        ],
    )
    compact = table(
        ["Preset", "Sampled + online", "Calibrated + memory"],
        [
            [
                method.upper() + " compact",
                score("grid_iid", "test", method + "_compact", "sampled_online", seeds=1),
                score("grid_iid", "test", method + "_compact", "guarded_online", seeds=1),
            ]
            for method in ("bfb", "rfb")
        ],
    )
    speed = json.loads((root / "results" / "throughput.json").read_text())["results"]
    speeds = table(
        ["Preset", "FB parameters", "Batch", "Updates/s", "Transitions/s"],
        [
            [
                ("Reference" if i == 0 else "Compact"),
                f"{r['fb_parameters']:,}",
                str(r["batch_size"]),
                f"{r['median_updates_per_second']:.0f}",
                f"{r['median_transitions_per_second']:.0f}",
            ]
            for i, r in enumerate(speed)
        ],
    )
    report = f"""# Reproduction and extension report — 3 November 2026

**The full set of paper claims is not reproduced.** Belief conditioning helps on the declared continuous rotation
shifts and the revised grid protocol. The paper-style fixed-belief grid scores are lower than the reported means;
RFB is not uniformly better than BFB; and a matched, strongly regularized FB baseline wins on AntWind.
The repo now contains tested implementations and the positive and negative evidence needed to inspect those findings.

## What was actually run

- 35 fresh research training runs: 3 full-size grid FB baselines; 18 continuous diagnostics (2 families × 3 methods ×
  3 seeds); 3 exploratory AntWind runs; 9 matched AntWind BC runs; and 2 compact-grid pilots.
- Independent re-evaluation of all 42 inherited v2 BFB/RFB grid checkpoints, including the layout-count and
  context-length sweeps. These are **re-evaluations**, not newly trained models. Their historical tuning used these
  test layouts, so the new evaluation seed does not make the layouts an untouched research holdout.
- Paired grid protocols, shuffled/zero-belief controls, random policies, full-map oracles, action-symmetry adapters,
  and affine adapters. Raw trials, checkpoint/configuration hashes, and provenance are in the evidence archive.
- Dataset audit of all five cached grid datasets and all 20 official AntWind files. Numerical regression tests and
  CLI smoke runs are separate from research results and are not counted as successful reproductions.

Except where explicitly marked as a pilot, tables show mean ± **sample SD over three training seeds**. Success rates
are fractions; AntWind uses raw return. [All scores](../results/summary.csv),
[paired bootstrap comparisons](../results/paired_comparisons.csv), and
[compressed trial evidence](../results/evidence/experiments.jsonl.gz) are machine-readable.

## Four-Rooms and distribution shifts

These evaluations use 30 training layouts, 20 held-out layouts, three goals, and 20 contexts per layout, with 100
exploration transitions and a 100-step control horizon. The learned weights remain fixed during evaluation.

{grid}

The uniform random policy scores about 0.143 on held-out rooms; the full-map oracle solves all trials. Static greedy
inference falls short of the paper's quoted BFB/RFB means of 0.40/0.61. The much higher sampled-online scores use
the changed v2 protocol and must not be presented as strict reproduction. Training also differs from the paper.

An unseen action permutation breaks the unmodified agents. Inferring the permutation from observed movements,
then avoiding repeatedly observed self-loops, substantially improves transfer. The identical adapters are also
evaluated with FB. Neither adapter reads the grid or hidden action map. Collision memory assumes repeatable
transitions; its benefit is smaller or can disappear with action slip.

![Main results](../figures/reproduction.png)

## Dataset diversity and context length

The full [sweep table](../results/grid_sweeps.csv) contains three seeds per setting. Under sampled-online inference,
increasing layout diversity gives a modest, noisy gain that saturates. Increasing context length is **not
monotonically beneficial** in this implementation; 150/200-step contexts often score worse than 50/100-step ones.
Long training contexts span multiple random-reset episodes, while evaluation contexts are contiguous exploration;
that distribution mismatch is a plausible contributor, not a demonstrated causal explanation.

![Re-evaluated sweeps](../figures/grid_sweeps.png)

## Continuous diagnostics

These are new lightweight continuous-control benchmarks, **not the paper's MuJoCo PointMass maze**. The IID split
interpolates among training dynamics; the rotation split holds out an actuator-rotation quadrant; the gain split
reduces actuation below the training range. Inertia adds a second dynamics family. All use 100K FB updates and
reward-free random-action data. Policies are evaluated on four goal tasks that are specified only at inference.

{continuous}

Belief conditioning preserves useful behavior on rotations where FB fails. Transfer is nevertheless incomplete,
and gain extrapolation is not uniformly solved. RFB does not consistently improve upon BFB.

{adapted}

The affine extension estimates a transition map from the reward-free context and expresses actions in a reference
frame fitted from training data. It performs **closed-form test-time system identification**, so these are extension
results, not evidence for the paper's no-test-time-fitting protocol. It assumes observed state, affine dynamics,
and a known integration step. It helps rotation transfer, but is not a general solution to low gain, clipping,
stochasticity, or nonlinear robot dynamics. It is not applied to AntWind.

## AntWind: important negative result

This uses the released data, the paper repository's Ant XML, and its wind magnitude/viscosity. Training IDs are
0–15 and test IDs are 16–19. The context consists of reward-free slices of the released expert trajectories;
network normalization and reward labels for task inference use training data only. The horizon is 256 steps.

{ant}

The initial weakly regularized runs did not approach the paper's BFB/RFB returns (roughly 550/640). Increasing the
offline behavior penalty restored high returns, but **FB with the same penalty is better**. Thus the strong
regularized BFB/RFB returns do not reproduce the paper's algorithm ranking. Expert behavior in the offline data
is a major confound: a forward-locomotion score alone does not establish arbitrary-task zero-shot competence.
Shuffled-belief results are reported rather than omitted.

Uniform random actions score about 97 on held-out winds; zero actions score about 281. Baseline action definitions
matter. The main paper, appendix, and released notebooks also differ in their stated data counts/splits; this
implementation explicitly fixes 16 training and 4 test winds, and records the actual 819,200 training transitions.

## Efficiency and a failed small-model shortcut

{speeds}

These are median steady-state rates over three timed blocks of 1,000 updates on one RTX 3090 Ti, after compilation
and belief-bank construction. The presets have different batch sizes; these rates are not full-run wall times or
an equal-quality comparison. [Full timing evidence](../results/throughput.json) includes each measurement.

{compact}

The compact preset is faster and smaller but loses substantial grid performance at its 100K-update/8K-encoder
budget. It is useful for development; the reference settings remain the quality baseline. The compact pilot is
not promoted as a successful reproduction. Frozen belief banks, on-device sampling, fused update loops, cached
encoders/compilation, and bounded multi-GPU workers make the broader suite practical.

## Limits and interpretation

- Original MuJoCo randomized PointMass and OGBench Scene/friction experiments were not reproduced. The continuous
  diagnostics are explicitly different tasks. No scores are invented or substituted for those missing experiments.
- The fresh main comparisons use three seeds, not the paper's five. HILP/LAP remain available and were smoke-tested;
  their full multi-benchmark claims were not freshly replicated.
- The grid RFB reference includes belief perturbation in addition to the vMF task prior. Its gains cannot be
  attributed solely to nonoverlapping policy clusters. With κ=10 in 100 dimensions the directional bias is weak.
- Gaussian encoder outputs are embeddings, not demonstrated calibrated posteriors. High reconstruction accuracy
  alone does not guarantee useful extrapolation or control.
- The observation-only grid adapters and the affine system-identification extension are separate contributions,
  not silent changes to the paper's evaluation. Structured priors help, and also restrict the scope of the claims.
- Continuous success means reaching a goal at least once within the horizon; the archived horizon-end distances
  are not used in the comparisons. No checkpoint is selected by maximizing test return.

See [benchmark definitions](benchmarks.md) for exact information budgets and [implementation notes](methods.md)
for corrections, checkpoint semantics, and the continuous pilot configuration compatibility note.

Sources: [paper, arXiv v2](https://arxiv.org/abs/2505.13150v2),
[official repository](https://github.com/maxsbob/BeliefConditionedFB), and local tag `v1.0-paper-original`.
"""
    (root / "docs" / "reproduction.md").write_text(report)
    print(root / "docs" / "reproduction.md")


if __name__ == "__main__":
    main()
