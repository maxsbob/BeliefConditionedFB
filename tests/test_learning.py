import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from bfb.agent import FBAgent
from bfb.continuous import ContinuousFBAgent
from bfb.data import sample_contexts
from bfb.experiment import ROOT
from bfb.networks import DynamicsEncoder
from bfb.vmf import sample_vmf, w_quantile_table


def config(continuous=False, method="bfb"):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="continuous" if continuous else "train", overrides=[f"agent={method}"])
    cfg.agent.f_hidden_dims = [16]
    cfg.agent.f_preprocessor_dims = [16]
    cfg.agent.f_embed_dim = 16
    cfg.agent.b_hidden_dims = [16]
    cfg.agent.z_dim = cfg.agent.h_dim = 8
    cfg.agent.encoder.emb_dim = 8
    cfg.agent.encoder.mlp_dim = 16
    cfg.agent.encoder.num_layers = 1
    cfg.agent.encoder.predictor_hidden_dims = [16]
    if continuous:
        cfg.agent.actor_hidden_dims = [16]
    return cfg.agent


def test_encoder_is_permutation_invariant():
    model = DynamicsEncoder(8, 8, 1, 2, 16)
    obs = jax.random.normal(jax.random.key(1), (3, 12, 2))
    act = jax.nn.one_hot(jnp.arange(36).reshape(3, 12) % 4, 4)
    nxt = obs + 0.1
    params = model.init(jax.random.key(2), obs, act, nxt)
    a = model.apply(params, obs, act, nxt)
    perm = np.random.default_rng(1).permutation(12)
    b = model.apply(params, obs[:, perm], act[:, perm], nxt[:, perm])
    np.testing.assert_allclose(a[0], b[0], atol=1e-5)
    np.testing.assert_allclose(a[1], b[1], atol=1e-5)


def test_short_contexts_do_not_cross_random_resets():
    # Encode episode identity in observations, so crossing a reset is observable.
    obs = jnp.repeat(jnp.arange(5), 10).reshape(1, 50, 1)
    arrays = {"obs": obs, "next_obs": obs, "actions": jnp.zeros((1, 50), jnp.int32)}
    batch = sample_contexts(arrays, jax.random.key(0), jnp.zeros(100, jnp.int32), 7, 10)
    assert np.all(np.asarray(batch["obs"]).std(axis=1) == 0)


@pytest.mark.parametrize("kappa", [0.0, 10.0, 100.0])
def test_vmf_norm_and_analytic_first_moment(kappa):
    from scipy.special import ive

    dim = 8
    direction = jnp.zeros((12000, dim)).at[:, 2].set(1)
    samples = np.asarray(sample_vmf(jax.random.key(0), direction, kappa, w_quantile_table(kappa, dim)))
    np.testing.assert_allclose(np.linalg.norm(samples, axis=-1), 1, atol=1e-5)
    expected = 0 if kappa == 0 else ive(dim / 2, kappa) / ive(dim / 2 - 1, kappa)
    assert abs(samples[:, 2].mean() - expected) < 0.015
    assert np.max(np.abs(np.delete(samples.mean(0), 2))) < 0.015


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("method", ["fb", "bfb", "rfb"])
def test_update_is_finite_changes_parameters_and_roundtrips(tmp_path, continuous, method):
    cfg = config(continuous, method)
    agent = ContinuousFBAgent(cfg, 2) if continuous else FBAgent(cfg, 2, 4)
    rng = np.random.default_rng(1)
    batch = {
        "obs": jnp.asarray(rng.uniform(0, 1, (8, 2)), dtype=jnp.float32),
        "next_obs": jnp.asarray(rng.uniform(0, 1, (8, 2)), dtype=jnp.float32),
        "actions": jnp.asarray(rng.uniform(-1, 1, (8, 2)), dtype=jnp.float32)
        if continuous
        else jnp.asarray(rng.integers(4, size=8)),
    }
    if agent.use_context:
        batch["h"] = jnp.asarray(rng.normal(size=(8, 8)), dtype=jnp.float32)
    old = agent.state
    new, info = jax.jit(agent.update_fb)(old, batch, jax.random.key(4))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((new, info)))
    assert int(new.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(old.params), jax.tree.leaves(new.params)))
    agent.state = new
    from bfb.experiment import restore_training, save_training

    resume_path = tmp_path / "training.pkl"
    save_training(resume_path, agent, "configuration-a", [{"step": 1}])
    with pytest.raises(ValueError, match="mismatch"):
        restore_training(resume_path, agent, "configuration-b")
    assert restore_training(resume_path, agent, "configuration-a") == [{"step": 1}]
    resumed, _ = jax.jit(agent.update_fb)(agent.state, batch, jax.random.key(5))
    uninterrupted, _ = jax.jit(agent.update_fb)(new, batch, jax.random.key(5))
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(uninterrupted)):
        np.testing.assert_array_equal(a, b)
    checkpoint = tmp_path / "checkpoint.pkl"
    agent.save(checkpoint)
    agent.load(checkpoint)
    for expected, actual in zip(jax.tree.leaves(new.params), jax.tree.leaves(agent.state.params)):
        np.testing.assert_array_equal(expected, actual)


def test_continuous_oracle_handles_heldout_dynamics():
    from bfb.continuous_evaluation import evaluate_continuous

    for family in ("pointmass", "inertial"):
        for split in ("iid", "rotation", "gain"):
            result = evaluate_continuous(None, family, split, num_contexts=1, num_dynamics=4, policy="oracle")
            assert result["return"] == 1.0


def test_affine_adapter_preserves_observed_transition_equations():
    from bfb.affine_adaptation import AffineActionAdapter, fit_dynamics
    from bfb.envs.pointmass import generate_dataset

    for family in ("pointmass", "inertial"):
        data = generate_dataset(family, num_contexts=4, episodes=8, seed=31)
        obs, actions, nxt = [data[k] for k in ("obs", "actions", "next_obs")]
        reference = fit_dynamics(obs[:1], actions[:1], nxt[:1])[0]
        adapter = AffineActionAdapter(obs, actions, nxt, reference)
        canonical = adapter.to_reference(obs, actions)
        for t in (0, 10, 30):
            actual = adapter.to_actual(obs[:, t], canonical[:, t])
            np.testing.assert_allclose(actual, actions[:, t], atol=1e-4)


def test_antwind_zero_id_and_simulator_contract():
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from bfb.envs.antwind import make_antwind

    env = make_antwind(0, render_mode=None)
    try:
        obs, _ = env.reset(seed=123)
        assert obs.shape == (105,)
        np.testing.assert_allclose(env.model.opt.wind, [4, 0, 0])
        assert env.model.opt.viscosity == 0.05
        assert env.action_space.shape == (8,)
        nxt, reward, _, _, _ = env.step(np.zeros(8))
        assert np.isfinite(nxt).all() and np.isfinite(reward)
    finally:
        env.close()
