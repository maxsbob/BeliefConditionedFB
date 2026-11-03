"""Inference-only action coordinate change for locally affine continuous dynamics.

This extension assumes a known integration time step and full observed position (and velocity for inertial tasks).
It estimates coefficients from transitions, never from hidden environment parameters. It is not a generic solution
to nonlinear dynamics shifts such as AntWind.
"""

import numpy as np


def fit_dynamics(obs, actions, next_obs, ridge=1e-4):
    """Return batched least-squares coefficients mapping [action, optional velocity, 1] to motion."""
    inertial = obs.shape[-1] == 4
    features = [actions, obs[..., 2:]] if inertial else [actions]
    x = np.concatenate([*features, np.ones((*obs.shape[:2], 1))], -1).astype(np.float64)
    y = next_obs[..., 2:] if inertial else (next_obs - obs) / 0.1
    valid = (np.abs(next_obs[..., :2]) < 0.99999).all(-1).astype(np.float64)
    gram = np.einsum("bti,btj,bt->bij", x, x, valid) + ridge * np.eye(x.shape[-1])
    rhs = np.einsum("bti,btj,bt->bij", x, y, valid)
    return np.linalg.solve(gram, rhs).astype(np.float32)


class AffineActionAdapter:
    def __init__(self, obs, actions, nxt, reference):
        self.coefficients = fit_dynamics(obs, actions, nxt)
        self.reference = np.broadcast_to(reference, self.coefficients.shape)

    @staticmethod
    def _offset(coefficients, obs):
        offset = coefficients[:, -1]
        if obs.shape[-1] == 4:
            offset = offset + np.einsum("bi,bij->bj", obs[:, 2:], coefficients[:, 2:4])
        return offset

    def to_reference(self, obs, actions):
        # Vectorize over a context's time axis while keeping per-episode estimates independent.
        actual = np.einsum("bti,bij->btj", actions, self.coefficients[:, :2]) + self.coefficients[:, None, -1]
        reference_offset = self.reference[:, None, -1]
        if obs.shape[-1] == 4:
            actual += np.einsum("bti,bij->btj", obs[..., 2:], self.coefficients[:, 2:4])
            reference_offset = reference_offset + np.einsum("bti,bij->btj", obs[..., 2:], self.reference[:, 2:4])
        inverse = np.linalg.pinv(self.reference[:, :2])
        return np.einsum("bti,bij->btj", actual - reference_offset, inverse).astype(np.float32)

    def to_actual(self, obs, actions):
        desired = (
            np.einsum("bi,bij->bj", actions, self.reference[:, :2])
            + self._offset(self.reference, obs)
            - self._offset(self.coefficients, obs)
        )
        return np.clip(np.einsum("bi,bij->bj", desired, np.linalg.pinv(self.coefficients[:, :2])), -1, 1)
