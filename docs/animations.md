# Reproducing the dynamics animations

The README GIFs show actual rollouts recorded through the same evaluators as the benchmark suite. Recording
does not introduce another policy implementation or select episodes according to success. Every checkpoint has
training seed 0, and the evaluation seed is 20261103. These are qualitative examples, not additional aggregate
performance claims.

| Animation | Fixed example selection | Inference |
|---|---|---|
| Grid | First held-out layout, first context, goal (8,8) | Sampled tasks, online belief, action calibration, collision memory |
| Point mass | First dynamics instance, first context, goal (0.75,0.75) | Online belief; affine adapter only in the explicitly labeled third scene |
| Inertial | Same selection, with agents trained on inertial dynamics | Same four inference scenes as point mass |
| AntWind | First episode of wind IDs 0, 16, 19 | Online belief, matched behavior penalty 1.0 |

Each scene starts a **new episode with a fresh context**. No clip demonstrates abrupt dynamics changes during a
rollout. Grid slip is stochastic within an episode, but its probability remains fixed. The affine adapter fits a
transition model at evaluation time; it is an extension to the strict no-fitting protocol. AntWind context comes
from the released offline expert trajectories, whereas the diagnostic environments use random exploration.

The videos retain failed trials. “Goal reached” means the goal was hit at least once; continuous policies keep
moving through the full horizon, matching the success metric. The grid simulator stops a successful episode.
Ant returns accumulate until physical termination or the 256-step horizon. Playback samples every two simulator
steps for grids/diagnostics and every eight steps for Ant; the step counter is authoritative, not playback time.

Each GIF has a [compressed JSON companion](../figures/dynamics/) containing checkpoint hashes, exact dynamics, displayed
trajectories, and full evaluation results. Ant trajectory positions are recorded only for visualization and are
not extra policy inputs. Its muted floor and agent colors change rendering materials only. The benchmark physics,
observations, actions, and rewards are unchanged.

## Regenerate

Generation requires the trained checkpoints and, for AntWind, the official dataset. These large artifacts are
kept locally outside Git; viewing the README GIFs needs neither. Use the training commands in the README to create
checkpoints, then point the generator at their run directories:

```bash
# The reference paths used for the published GIFs are the defaults.
CUDA_VISIBLE_DEVICES=0 uv run scripts/make_dynamics_gifs.py --suite grid
CUDA_VISIBLE_DEVICES=0 uv run scripts/make_dynamics_gifs.py --suite pointmass
CUDA_VISIBLE_DEVICES=0 uv run scripts/make_dynamics_gifs.py --suite inertial
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl uv run --extra mujoco scripts/make_dynamics_gifs.py --suite antwind

# Or generate all four, including headless MuJoCo rendering.
MUJOCO_GL=egl uv run --extra mujoco scripts/make_dynamics_gifs.py --suite all
```

Use `--overwrite` to explicitly replace published animations, or `--out` for a separate output directory.
`--grid-fb`, `--grid-root`, `--continuous-root`, `--ant-root`, and `--data-root` select artifact locations.
Each generation also writes a PNG contact sheet for reviewing all scenes. The GIF uses one palette across frames
to keep text and colors stable, and encodes only changed pixels to reduce download size. Readable JSON records
are retained locally; their deterministic `.json.gz` copies are committed. For the intended typography, install DejaVu Sans; a bundled Pillow font is the
fallback.
