import numpy as np

from postman.representations import AxisAnglePose, WholeBodyAction, WholeBodyCommand


def test_ummr_roundtrip():
    command = WholeBodyCommand(
        base_twist=np.array([1.0, 2.0, 3.0]),
        torso=np.array([0.4]),
        left_wrist=AxisAnglePose(np.ones(3), np.arange(3)),
        right_wrist=AxisAnglePose(-np.ones(3), -np.arange(3)),
        left_elbow=np.ones(3),
        right_elbow=-np.ones(3),
        grippers=np.array([0.1, 0.9]),
    )
    restored = WholeBodyCommand.from_vector(command.as_vector())
    np.testing.assert_allclose(restored.as_vector(), command.as_vector())


def test_action_applies_bounded_residual():
    command = WholeBodyCommand()
    action = WholeBodyAction(np.ones(command.dim)).clipped(0.25)
    assert np.max(action.apply(command).as_vector()) == 0.25
