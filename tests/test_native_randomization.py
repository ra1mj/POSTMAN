"""Small fake-model tests for the native reset domain randomization event."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from postman.tasks.arm_reach.mdp.randomization import (  # noqa: E402
    dynamics_observation,
    randomize_dynamics,
)


def _env(num_envs=3):
    # Two robot bodies (body 0 is furniture), two robot DOFs and two actuators.
    baseline = SimpleNamespace(
        body_mass=torch.tensor([20.0, 2.0, 3.0]),
        body_inertia=torch.arange(9.0).reshape(3, 3) + 1,
        dof_frictionloss=torch.tensor([.1, .2, .3, .4]),
        dof_damping=torch.tensor([1., 2., 3., 4.]),
        actuator_gainprm=torch.ones(2, 4),
        actuator_biasprm=torch.ones(2, 4) * -2,
    )
    model = SimpleNamespace(**{k: v.unsqueeze(0).repeat(num_envs, *([1] * v.ndim)).clone()
                               for k, v in vars(baseline).items()})
    indexing = SimpleNamespace(
        body_ids=torch.tensor([1, 2]),
        joint_v_adr=torch.tensor([1, 2]),
        free_joint_v_adr=torch.tensor([3, 0]),
        ctrl_ids=torch.tensor([0, 1]),
    )
    robot = SimpleNamespace(indexing=indexing)
    sim = SimpleNamespace(model=model, mj_model=baseline, device="cpu")
    return SimpleNamespace(num_envs=num_envs, device="cpu", sim=sim, scene={"robot": robot})


def test_native_event_is_robot_only_correlated_and_non_cumulative():
    env = _env()
    before = env.sim.model.body_mass.clone()
    obs0 = dynamics_observation(env)
    assert torch.all(obs0 == 1)
    torch.manual_seed(7)
    randomize_dynamics(env, torch.tensor([0, 2]))
    state = env._postman_dr
    assert torch.all(env.sim.model.body_mass[:, 0] == before[:, 0])  # furniture
    assert torch.allclose(
        state["mass"][0, state["body_ids"]], state["inertia"][0, state["body_ids"]]
    )
    assert torch.allclose(state["gain"], state["bias"])
    assert torch.all(
        (state["mass"][:, state["body_ids"]] >= 0.9)
        & (state["mass"][:, state["body_ids"]] <= 1.1)
    )
    assert torch.all(
        (state["friction"][:, state["dof_ids"]] >= 0.8)
        & (state["friction"][:, state["dof_ids"]] <= 1.2)
    )
    first = env.sim.model.body_mass[0].clone()
    torch.manual_seed(8)
    randomize_dynamics(env, torch.tensor([0]))
    expected = env.sim.mj_model.body_mass[env._postman_dr["body_ids"]] * env._postman_dr[
        "mass"
    ][0, env._postman_dr["body_ids"]]
    assert torch.allclose(env.sim.model.body_mass[0, env._postman_dr["body_ids"]], expected)
    assert not torch.allclose(first, env.sim.model.body_mass[0])


def test_partial_reset_leaves_other_world_factors_untouched():
    env = _env(2)
    torch.manual_seed(3)
    randomize_dynamics(env)
    old = env._postman_dr["mass"][1].clone()
    torch.manual_seed(4)
    randomize_dynamics(env, torch.tensor([0]))
    assert torch.allclose(old, env._postman_dr["mass"][1])
    assert dynamics_observation(env).shape[0] == 2
