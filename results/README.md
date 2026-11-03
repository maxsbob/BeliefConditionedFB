# Measured evidence

`summary.csv` reports one mean and sample standard deviation per benchmark, split, method, and inference protocol.
`seed_scores.csv` retains each seed's score. `grid_sweeps.csv` separates training-layout count and context length.
`paired_comparisons.csv` reports paired hierarchical bootstrap intervals over training seeds and dynamics IDs;
these descriptive intervals are based on only three training seeds. One-seed pilots have no estimated seed SD.

`evidence/experiments.jsonl.gz` contains 546 source records. Each JSON line has a repository-relative `path`,
the original file's `sha256`, and its full parsed `data`. This preserves trial arrays, configurations, evaluation
settings, checkpoint hashes, and recorded provenance. `manifest.json` hashes the compressed archive. The archive
does not include model weights or datasets; those large generated files remain in ignored `runs/` and `data/`.
The source hash identifies the code used at execution time; the current working tree also includes subsequent
reporting and compatibility fixes, as described in [implementation notes](../docs/methods.md).

Regenerate tables and figures from the included archive, without training or downloading data:

```bash
JAX_PLATFORMS=cpu uv run scripts/summarize_results.py \
  --from-evidence results/evidence/experiments.jsonl.gz
uv run scripts/write_report.py
```

Omit `--from-evidence` to aggregate locally completed runs and evaluation files. The aggregator uses the explicit
run directories listed in its source, so custom output directories must be incorporated deliberately. It rejects
duplicate seeds and incompatible paired comparisons rather than silently averaging repeated evaluations.
The report generator requires all three seeds for its main comparisons and will reject incomplete results.

`data_audit.json` records dataset sizes, hashes, transition checks, and AntWind discontinuities.
`throughput.json` contains warm-up settings, parameter counts, and repeated steady-state GPU timings.
The [report](../docs/reproduction.md) explains which results reproduce a protocol, extend it, or fail to support
the original claim. In particular, higher AntWind returns with behavior regularization do not reproduce the
paper's algorithm ranking, and the affine adapter uses test-time model fitting.
