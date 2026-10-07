import numpy as np

from postman.envs import make


def test_env_step_is_finite_and_reproducible():
    env_a = make("tiago_pp_bimanual_reach", seed=4)
    env_b = make("tiago_pp_bimanual_reach", seed=4)
    obs_a, _ = env_a.reset(seed=4)
    obs_b, _ = env_b.reset(seed=4)
    np.testing.assert_allclose(obs_a, obs_b)
    next_a = env_a.step(np.zeros(env_a.action_dim))
    next_b = env_b.step(np.zeros(env_b.action_dim))
    np.testing.assert_allclose(next_a[0], next_b[0])
    assert np.all(np.isfinite(next_a[0]))
