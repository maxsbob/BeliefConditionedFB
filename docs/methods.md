# Implementation and changes

## Shared FB structure

`F(s,a,z,h)` estimates successor features; `B(s')` is independent of dynamics. The discrete policy enumerates
four actions. The continuous policy uses a DDPG actor trained to maximize the twin forward networks' value
`F(s,a,z,h)ᵀ z`, with target-action smoothing. Backward features and task vectors have norm √d. Training uses the
FB off-diagonal Bellman objective, its diagonal occupancy term, and backward orthogonality regularization.

FB has no context input. BFB conditions on the inferred belief and uses a uniform task prior. RFB uses a vMF prior
around the inferred belief; half of training tasks still come from backward features. The inherited grid RFB also
perturbs the belief during training (`belief_kappa=100`). Continuous RFB disables this extra perturbation, isolating
the directional task prior. These are distinct variants, not an exact paper implementation.

The transformer encoder has no positional embedding and pools transition tokens symmetrically. It predicts
held-out query outcomes from the same training dynamics and is frozen during FB training. Grid outcomes are wall
collisions. Continuous outcomes are scaled state differences. Held-out query sampling uses known grouping of the
offline training data; the group index is never provided to the networks or used at test time.
The Gaussian output is a learned embedding distribution, not a demonstrated calibrated posterior. In particular,
the default zero KL coefficient allows its variance to shrink; vMF perturbation averages are sensitivity checks,
not a proof of Bayesian uncertainty calibration.

A finite bank of frozen-encoder beliefs avoids re-running the transformer on every FB update. It approximates the
context distribution rather than being mathematically identical to sampling fresh contexts. Training sampling
and update loops are fused with `jax.lax.scan`; vMF inverse-CDF tables avoid repeated rejection sampling on GPU.
The tables are approximate numerical samplers and are checked against analytic distribution moments.

## Corrections in this audit

- Cache hits no longer change the FB random-number stream: encoder, belief-bank, and FB streams are separated.
- Training executes non-multiple final chunks and encoder budgets smaller than the logging interval.
- Dataset caches check layout counts, episode dimensions and generation metadata; a nonzero seed cannot silently
  reuse an unversioned seed-0 dataset.
- Short training contexts remain inside a recorded episode. Long set-valued contexts may intentionally include
  multiple episodes of the same dynamics and are not claimed to be one uninterrupted trajectory.
- Held-out layout exclusion grows with the training set and compares actual grids, not just seed integers.
- Evaluations retain all trials and use paired random streams. Static/online beliefs and greedy/sampled actions
  are separated. Context ablations and random/oracle controls are explicit.
- HILP/LAP now support the task perturbation invoked by the default evaluation settings.
- Inference checkpoint loading initializes target parameters consistently. Inference and full training-state
  checkpoints have separate roles; continuous training-state checkpoints include optimizer state and counters.
- Re-scoring an old run creates a separate evaluation artifact instead of overwriting its training record.
- Launchers use OS locks, unique variant paths, command manifests, and nonzero failure exit statuses.
- Checkpoint loading is a library operation and no longer imports plotting scripts.

## Extensions

The discrete action adapter exploits permutation symmetry of the action interface. It finds the bijection with
maximum evidence from actual movements, then canonicalizes encoder inputs and policy outputs. Collision memory
prevents repeated attempts at actions observed only to self-loop. Both consume only the context and subsequent
observations. Neither reads the true layout or action map, changes network weights, or observes reward.

The continuous adapter estimates affine coefficients from transitions and transforms to/from one training
reference's action coordinates. This retains the trained FB actor and task representation but adds a strong
structural dynamics assumption. It helps distinguish representational extrapolation failure from lack of
information in the context. Its limits, including clipping and nonlinear dynamics, must remain explicit.
This is closed-form test-time system identification, not a reproduction of the paper's no-fitting protocol.

For AntWind, the optional actor behavior penalty reduces exploitation of poorly supported actions in the offline
dataset. It can also make behavior less task-dependent, so it is evaluated as a separate regularized variant and
should not be presented as proof of arbitrary-task zero-shot learning.

## Reproducibility

New training runs save resolved configs, a Git commit, a hash of current source files, dataset hashes, device and
library details. Dirty-worktree provenance is deliberate: preexisting research files are preserved. Evaluation
records include checkpoint hashes and exact task/seed settings. Compact evidence archives preserve configurations
and per-trial arrays while generated datasets and checkpoints remain outside Git.

The original `v1.0-paper-original` tag and inherited v2 reference are retained in Git history. The report labels
fresh training separately from re-evaluation of preexisting checkpoints. GPU wall times include other concurrent
experiments and are not isolated hardware speed benchmarks.

The earliest continuous pilot config files inherited `target_agg: mean` from the discrete defaults even though
the continuous kernel used twin minima. The current continuous preset explicitly declares `min`, and the kernel
honors this option. All recorded experiments in this report use minimum aggregation; old config files are retained
unaltered as evidence. The training-state schema prevents silently resuming across this implementation change.
