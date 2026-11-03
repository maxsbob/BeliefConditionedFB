# Benchmark definitions

All learned policies receive observations, a reward-free transition context where applicable, and a task encoding.
They receive no test layout IDs, action permutations, wind parameters, or simulator state beyond the declared
observation. Full-information oracle controls are explicitly separate.

| Benchmark | Observations/actions | Training and test | Metric |
|---|---|---|---|
| Four-Rooms IID | (row, column); 4 discrete actions | Train layouts 0–29; 20 fixed excluded layouts starting at seed 10000 | Success within 100 steps |
| Four-Rooms slip | Same | Same held-out layouts; each commanded action is replaced by a uniform action with probability 0.25 | Success within 100 steps |
| Four-Rooms remap | Same | Same held-out layouts; action map `[3, 2, 0, 1]` | Success within 100 steps |
| Random obstacles | Same | Connected 11×11 grids, interior obstacle probability 0.22, no four-room constraint | Success within 100 steps |
| PointMass diagnostic | Continuous position; 2 continuous actions | Hidden action rotation, gain, and wind; split definitions below | Reach distance ≤0.12 within 80 steps |
| Inertial diagnostic | Position and velocity; 2 continuous actions | Same, plus hidden damping | Same |
| AntWind | 105-dimensional Ant observation; 8 continuous actions | Official data contexts 0–15; held-out wind IDs 16–19 | Unnormalized Gymnasium return over ≤256 steps |

**The two diagnostic point-mass tasks are new environments. They are not the paper's randomized MuJoCo maze.**
Four-Rooms IID means new dynamics instances drawn from the training generator. Slip, remapping, and obstacle
structure are actual distribution shifts. These categories should not be combined into one “OOD” score.

## Grid protocol

The start is (1,1), and goals are (2,8), (8,2), (8,8). Every test layout is evaluated with 20 reward-free exploration
contexts and all three goals: 1,200 trials per training seed. The initial context is a 100-step random walk from
(1,1), shared across goals. Separate random streams generate contexts, rollout sampling (including starts),
and environment noise. Algorithms receive paired contexts and environmental random draws.

- `greedy_static`: fixed initial belief, deterministic greedy action. This matches the paper's inference style,
  but inherited v2 checkpoints were trained with modified architectures/objectives/settings.
- `greedy_online`: refresh the belief from the latest 100 transitions; deterministic actions.
- `sampled_static`: fixed belief; task code drawn from vMF(B(goal), 200) each step.
- `sampled_online`: both changes.
- `shuffled_online` / `zero_online`: remove the association between belief and current dynamics, or remove belief.
  Shuffling shifts whole layout groups, so it does not accidentally retain the same layout's belief.
- `calibrated_online`: infer the discrete action permutation from observed nonzero movement; encode canonical
  actions and map selected actions back. The true action map is never passed to the adapter.
- `guarded_online`: calibration plus memory of observed self-loops. Actions that have only failed from the current
  cell are masked; unobserved actions remain available. Any observed successful move reopens an action. This is a
  deterministic-world prior, so it can hurt in stochastic environments.

Random-policy controls and shortest-path full-map oracles execute the same horizon and physical dynamics. Reaching
a goal gives reward once. Oracle scores are sanity checks, not learned-agent results. Validation layouts start at
seed 20000; test layouts start at 10000 and are excluded from the first max(50, N) training layouts by grid content.

## Continuous diagnostics

Position lies in [-1,1]². Non-inertial motion is `p' = clip(p + 0.1 * (gain * R(angle) * action + wind))`.
The inertial variant uses `v' = damping * v + (1 - damping) * force`, then `p' = clip(p + 0.1 * v')`;
velocity components are reset on boundary collisions. Actions are clipped to [-1,1]².

There are 16 training dynamics, 256 random-start episodes per dynamics, and 64 random-action steps per episode:
262,144 reward-free transitions. Each evaluation uses 16 dynamics, eight independent contexts, and four corner
waypoint tasks, totaling 512 trials per training seed. Policies start at the center; the exploratory context starts
at a random position. All task goals are given only at evaluation.

- Training rotation angles span [0, 3π/2), gains are 0.85–1.15, and wind magnitude is 0.12.
- Validation offsets the training angle lattice by a quarter-cell; IID test uses a half-cell offset.
- Rotation extrapolation uses the excluded last quadrant [3π/2, 2π).
- Gain extrapolation uses 0.45–0.60, below every training gain.
- Inertial damping is 0.65, 0.75, or 0.85; the same rotation/gain exclusions apply.

The affine adapter fits action, velocity (when observed), and intercept coefficients to the context, discarding
boundary-clipped transitions. It expresses actions in a reference coordinate frame fitted to **training context
0's transitions**, and inverts the map when acting. It assumes affine dynamics, a known integration step, and full
state observations. It is not applied to AntWind and is not evidence of general nonlinear system identification.
It performs closed-form test-time model fitting, so it does **not** satisfy a strict no-test-time-fitting
interpretation of the paper's zero-shot protocol. Neural network weights remain frozen.

## AntWind

The XML is copied from `v1.0-paper-original` under the repository's Apache-2.0 license. Wind is set directly on the
MuJoCo model: magnitude 4, angle `2π * id / 20`, viscosity 0.05. This avoids the original temporary-XML replacement
and the `env_id=0` truthiness bug. The observed state and frame skip match Gymnasium's Ant-v5 with the supplied XML.

Training uses 16 × 200 × 256 = 819,200 transitions. Observation scaling statistics and the 10,000 reward labels
used to infer the forward-locomotion task vector come exclusively from training contexts. The model is trained on
reward-free transitions; reward labels are used only for task inference. Actions are drawn from the learned actor.
Physical unhealthy states stop bootstrapping; an artificial dataset time limit does not.

Evaluation contexts are 100-step reward-free slices of the **provided expert trajectories**, including held-out
contexts for held-out evaluation. This is a different information budget from random exploration and is reported
in each result. Five simulator episodes per wind ID are used. Static, online, and shuffled-belief evaluations are
separate. The supplied datasets contain a small number of within-array episode discontinuities, recorded in
`results/data_audit.json`; each recorded transition remains intact, and no synthetic cross-reset transition is made.

The stronger `bc_coef=1.0` variant regularizes the actor toward the offline behavior actions. Because that dataset
was collected by task-trained policies, improved locomotion can come from imitation. Compare FB/BFB/RFB with the
same regularizer and inspect shuffled-belief controls before attributing a gain to dynamics inference.

## Uncertainty and selection

Report the mean and sample standard deviation across independent training seeds. Contexts, goals and layouts are
not counted as additional training seeds. A one-seed result has no estimated seed standard deviation. Paired
comparisons resample training seeds and dynamics IDs while preserving algorithm pairing (5,000 bootstrap draws).
Three seeds provide limited evidence; intervals are descriptive, not a broad robustness guarantee.

Continuous checkpoints are evaluated at a predeclared final update count; intermediate feedback uses validation
dynamics (or training dynamics for AntWind). Adapter designs were checked on validation before test evaluation.
All inference ablations are published, including worse results. No best-seed or best-test-checkpoint selection is
used. Inherited v2 checkpoints have historical tuning exposure to their test layouts; their fresh evaluations do
not make those layouts a previously untouched research test set.
