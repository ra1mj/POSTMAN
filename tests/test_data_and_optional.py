import numpy as np

from postman.data import UMMRTrajectory
from postman.rl.distillation import mse_distillation_loss


def test_trajectory_roundtrip(tmp_path):
    trajectory = UMMRTrajectory(np.arange(3), np.zeros((3, 24)), np.ones((3, 4)))
    path = tmp_path / "trajectory.npz"
    trajectory.save_npz(path)
    restored = UMMRTrajectory.load_npz(path)
    np.testing.assert_allclose(restored.commands, trajectory.commands)


def test_distillation_loss():
    assert mse_distillation_loss(np.zeros((2, 3)), np.ones((2, 3))) == 1.0
