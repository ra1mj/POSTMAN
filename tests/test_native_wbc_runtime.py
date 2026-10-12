import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("mjlab")


def test_native_wbc_action_and_command_contract():
    from mjlab.envs import ManagerBasedRlEnv

    from postman.tasks.arm_reach.config.tiago_pp.env_cfgs import (
        tiago_pp_arm_reach_env_cfg,
    )

    env = ManagerBasedRlEnv(tiago_pp_arm_reach_env_cfg(), device="cpu")
    assert env.step_dt == pytest.approx(0.01)
    assert env.action_manager.total_action_dim == 25
    assert env.observation_manager.group_obs_dim["actor"] == (123,)

    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 123)
    assert torch.isfinite(observations["actor"]).all()

    observations, reward, terminated, truncated, _ = env.step(
        torch.zeros((1, 25), device="cpu")
    )
    assert observations["actor"].shape == (1, 123)
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(reward).all()
    assert not bool(terminated.any())
    assert not bool(truncated.any())

    action_term = env.action_manager.get_term("whole_body")
    assert action_term.applied_targets.shape == (1, 25)
    assert torch.isfinite(action_term.applied_targets).all()
