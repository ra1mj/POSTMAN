import numpy as np

from postman.representations import RobotState, WholeBodyCommand
from postman.wbc import KinematicQpWbc, KinematicQpWbcConfig, LinearTask


def test_kinematic_qp_tracks_identity_task_without_osqp():
    state = RobotState(np.zeros(4), np.zeros(4))
    controller = KinematicQpWbc(KinematicQpWbcConfig(dof=4, use_osqp=False, max_delta_q=0.5))
    command = WholeBodyCommand()
    task = LinearTask("joint", np.eye(4), np.ones(4))
    result = controller.compute(state, command, [task])
    np.testing.assert_allclose(result.q_des, 0.5 * np.ones(4), atol=1e-6)
    assert result.solver_status == "solved_numpy"
    assert np.all(np.isfinite(result.tau_cmd))
