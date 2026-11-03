import numpy as np
import pytest

from bfb.data import generate_dataset, load_or_generate
from bfb.envs.fourrooms import VecFourRooms, make_layout
from bfb.envs.fourrooms import test_layout_seeds as heldout_seeds
from bfb.envs.grid_suites import grid_suite
from bfb.evaluation import evaluate


def test_holdouts_are_disjoint_even_for_large_training_set():
    training = {make_layout(s).tobytes() for s in range(200)}
    test = [make_layout(s).tobytes() for s in heldout_seeds(40, exclude=200)]
    assert not training.intersection(test)
    assert len(set(test)) == 40


def test_goal_is_rewarded_once_and_finished_agent_stays():
    grid = make_layout(0)
    env = VecFourRooms(grid[None])
    env.reset(np.array([[1, 1]]), np.array([[1, 2]]))
    pos, reward, done = env.step([3])
    assert reward.item() == 1 and done.item()
    pos2, reward, done = env.step([1])
    np.testing.assert_array_equal(pos, pos2)
    assert reward.item() == 0 and done.item()


def test_action_calibration_uses_displacements_and_collision_guard_only_observations():
    from bfb.adaptation import CollisionMemory, infer_action_map
    from bfb.envs.fourrooms import MOTIONS

    mapping = np.array([3, 2, 0, 1])
    obs = np.full((2, 4, 2), 5)
    actions = np.tile(np.arange(4), (2, 1))
    nxt = obs + MOTIONS[mapping][None]
    np.testing.assert_array_equal(infer_action_map(obs, actions, nxt), np.tile(mapping, (2, 1)))
    memory = CollisionMemory(2)
    pos = np.array([[1, 1], [5, 5]])
    assert not memory.mask(pos).any()
    memory.update(pos, np.array([0, 3]), pos)
    assert memory.mask(pos)[0, 0] and memory.mask(pos)[1, 3]
    memory.update(pos, np.array([0, 3]), pos + 1)
    assert not memory.mask(pos).any()


def test_cache_rejects_mismatched_seed_and_shape(tmp_path):
    path = str(tmp_path / "dataset.npz")
    load_or_generate(path, 2, 800, 100, seed=7)
    with pytest.raises(ValueError, match="different settings"):
        load_or_generate(path, 2, 800, 100, seed=8)
    with pytest.raises(ValueError, match="dimensions"):
        load_or_generate(path, 4, 800, 100, seed=7)
    with pytest.raises(ValueError, match="complete episode"):
        generate_dataset(4, total_transitions=10)


@pytest.mark.parametrize("suite", ["iid", "slip", "remap", "obstacles"])
def test_oracle_solves_declared_suites_and_random_is_reproducible(suite):
    seeds, grids, kwargs = grid_suite(suite, 3)
    options = {"context_len": 10, "num_contexts": 3, "seed": 1, "grids_override": grids, "env_kwargs": kwargs}
    oracle = evaluate(None, seeds, policy="oracle", **options)
    random1 = evaluate(None, seeds, policy="random", **options)
    random2 = evaluate(None, seeds, policy="random", **options)
    assert oracle["return"] == 1.0
    assert random1["return"] < oracle["return"]
    np.testing.assert_array_equal(random1["successes"], random2["successes"])
    assert oracle["successes"].shape == (3, 3, 3)


def test_animation_observers_do_not_change_rollouts():
    from bfb.continuous_evaluation import evaluate_continuous

    frames = []

    def observer(step, obs, scores, done):
        frames.append(step)
        # Observers receive copies, so even accidental array mutation cannot alter the rollout.
        obs[:] = 0
        scores[:] = 0
        done[:] = True

    seeds, grids, kwargs = grid_suite("slip", 2)
    options = {
        "context_len": 10,
        "num_contexts": 2,
        "max_steps": 20,
        "seed": 17,
        "grids_override": grids,
        "env_kwargs": kwargs,
        "policy": "random",
    }
    baseline = evaluate(None, seeds, **options)
    recorded = evaluate(None, seeds, trajectory_callback=observer, **options)
    np.testing.assert_array_equal(baseline["successes"], recorded["successes"])
    np.testing.assert_array_equal(baseline["steps_to_goal"], recorded["steps_to_goal"])
    assert frames == list(range(21))
    frames.clear()
    options = {
        "family": "inertial",
        "split": "rotation",
        "num_dynamics": 2,
        "num_contexts": 2,
        "max_steps": 20,
        "policy": "random",
    }
    baseline = evaluate_continuous(None, **options)
    recorded = evaluate_continuous(None, trajectory_callback=observer, **options)
    np.testing.assert_array_equal(baseline["successes"], recorded["successes"])
    np.testing.assert_array_equal(baseline["final_distance"], recorded["final_distance"])
    assert frames == list(range(21))
