import numpy as np
import pytest

from postman.representations import RobotState, WholeBodyCommand
from postman.wbc import (
    BatchedBoundedLeastSquares,
    BatchedLeastSquaresConfig,
    KinematicQpWbc,
    KinematicQpWbcConfig,
    mixed_actuator_targets,
)


def test_missing_tasks_holds_without_identity_mapping():
    state = RobotState(np.arange(25, dtype=float), np.zeros(25))
    result = KinematicQpWbc(KinematicQpWbcConfig(dof=25, use_osqp=False)).compute(
        state, WholeBodyCommand()
    )
    np.testing.assert_allclose(result.q_des, state.q)
    np.testing.assert_allclose(result.qd_des, 0.0)
    assert result.solver_status == "hold_missing_tasks"


def test_mixed_targets_keep_wheels_as_velocity_commands():
    q = np.zeros((2, 25))
    velocity = np.ones((2, 25))
    out = mixed_actuator_targets(q, velocity, 0.1)
    np.testing.assert_allclose(out[:, :4], 1.0)
    np.testing.assert_allclose(out[:, 4:], 0.1)


def test_batched_solver_respects_bounds_and_weights():
    torch = pytest.importorskip("torch")
    J = torch.eye(3).expand(2, 3, 3)
    target = torch.tensor([[2.0, -2.0, 0.3], [1.0, -1.0, 0.2]])
    lower = torch.full((2, 3), -0.5)
    upper = torch.full((2, 3), 0.5)
    result = BatchedBoundedLeastSquares(BatchedLeastSquaresConfig(iterations=64)).solve(
        J, target, lower, upper
    )
    assert result.shape == (2, 3)
    assert torch.isfinite(result).all()
    assert torch.all(result <= upper + 1e-6)
    assert torch.all(result >= lower - 1e-6)
    assert torch.allclose(result[0], torch.tensor([0.5, -0.5, 0.3]), atol=2e-3)
