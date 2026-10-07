from postman.envs import TASK_REGISTRY, make


def test_tasks_registered():
    assert "tiago_pp_bimanual_reach" in TASK_REGISTRY.names()
    env = make("tiago_pp_bimanual_reach", seed=1)
    observation, _ = env.reset(seed=1)
    assert observation.size == 25 + 25 + 24
    assert env.action_dim == 24
