import numpy as np

from postman.envs import VectorizedManagerEnv, make


def test_vectorized_env_shapes():
    env = VectorizedManagerEnv(lambda seed: make("tiago_pp_bimanual_reach", seed=seed), 3, seed=2)
    observations, _ = env.reset(seed=2)
    assert observations.shape == (3, 74)
    result = env.step(np.zeros((3, env.action_dim)))
    assert result[0].shape == (3, 74)
    assert result[1].shape == (3,)
