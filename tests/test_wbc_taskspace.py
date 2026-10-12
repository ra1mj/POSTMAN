import numpy as np
import pytest

from postman.representations import AxisAnglePose, WholeBodyCommand
from postman.wbc import TiagoTaskSpaceWbc, wheel_to_base_matrix


def test_wheel_mapping_is_forward_and_yaw_without_lateral_motion():
    mapping = wheel_to_base_matrix()
    np.testing.assert_allclose(mapping @ np.ones(4), [0.0762, 0, 0])
    turning = mapping @ np.array([1, -1, 1, -1])
    assert turning[2] > 0
    assert turning[0] == turning[1] == 0


@pytest.fixture
def controller():
    mujoco = pytest.importorskip("mujoco")
    from postman.robots import TiagoPPScene

    try:
        path = TiagoPPScene().resolve_robot_mjcf()
    except FileNotFoundError:
        pytest.skip("TIAGo++ model is not installed")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wbc = TiagoTaskSpaceWbc(model)
    wbc.capture_reference(data)
    return mujoco, model, data, wbc


def test_taskspace_reference_is_stable_and_targets_are_mixed(controller):
    _, _, data, wbc = controller
    reference = wbc.reference
    command = WholeBodyCommand(left_wrist=AxisAnglePose(position=[0.05, 0, 0]))
    first = wbc.compute(data, command)
    second = wbc.compute(data, command)
    assert wbc.reference is reference
    assert first.actuator_targets.shape == (25,)
    assert first.solver_status == "solved_projected"
    assert np.isfinite(first.actuator_targets).all()
    np.testing.assert_allclose(first.actuator_targets[:4], first.qd_des[:4])
    assert first.constraint_violation <= 1e-10
    assert np.max(np.abs(second.qd_des - first.qd_des)
                  - wbc.config.dt * wbc.acceleration_limits) <= 1e-10
    assert first.diagnostics["task_errors"]["left_wrist"] == pytest.approx(0.05)


def test_joint_jacobian_matches_finite_difference_and_wheels_move_base(controller):
    mujoco, model, data, wbc = controller
    jacobian, _ = wbc.frame_jacobian(data, "left_wrist")
    body = wbc.body_ids["left_wrist"]
    initial = data.xpos[body].copy()
    epsilon = 1e-7
    index = 10
    data.qpos[wbc.qpos_indices[index]] += epsilon
    mujoco.mj_forward(model, data)
    derivative = (data.xpos[body] - initial) / epsilon
    np.testing.assert_allclose(derivative, jacobian[:, index], atol=1e-6)
    assert np.linalg.norm(jacobian[:, :4]) > 0


def test_taskspace_tracking_improves_under_kinematic_integration(controller):
    mujoco, model, data, wbc = controller
    command = WholeBodyCommand(left_wrist=AxisAnglePose(position=[0, 0, 0.03]))
    initial_error = None
    for _ in range(150):
        result = wbc.compute(data, command)
        error = result.diagnostics["task_errors"]["left_wrist"]
        initial_error = error if initial_error is None else initial_error
        data.qpos[wbc.qpos_indices[4:]] = result.actuator_targets[4:]
        # Integrate the virtual differential base, not wheel angle as pose.
        yaw = np.arctan2(data.xmat[wbc.body_ids["footprint"]].reshape(3, 3)[1, 0],
                         data.xmat[wbc.body_ids["footprint"]].reshape(3, 3)[0, 0])
        data.qpos[0] += result.base_cmd[0] * np.cos(yaw) * wbc.config.dt
        data.qpos[1] += result.base_cmd[0] * np.sin(yaw) * wbc.config.dt
        yaw += result.base_cmd[2] * wbc.config.dt
        data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
    assert error < initial_error * 0.65
