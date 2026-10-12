import numpy as np
import pytest

from postman.representations import (
    COMMAND_OFFSET_LIMITS,
    AxisAnglePose,
    TrajectoryAnchor,
    WholeBodyCommand,
    WholeBodyTrajectory,
    make_synthetic_trajectory,
    sample_synthetic_offset,
)


@pytest.mark.parametrize("mode", ("stand", "reach", "side", "height", "base", "bimanual"))
def test_synthetic_offsets_are_seeded_finite_and_bounded(mode):
    first = sample_synthetic_offset(np.random.default_rng(11), mode)
    second = sample_synthetic_offset(np.random.default_rng(11), mode)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (24,)
    assert np.all(np.isfinite(first))
    assert np.all(np.abs(first) <= COMMAND_OFFSET_LIMITS + 1e-12)


def test_synthetic_trajectory_interpolates_rotation_and_shortest_base_yaw():
    start = np.zeros(24)
    end = np.zeros(24)
    end[2] = np.deg2rad(179.0)
    end[7:10] = [0.0, 0.0, np.pi]
    trajectory = WholeBodyTrajectory(np.array([0.0, 1.0]), np.stack([start, end]))
    midpoint = trajectory.sample(0.5).as_vector()
    assert np.isclose(midpoint[2], np.deg2rad(89.5))
    assert np.isclose(np.linalg.norm(midpoint[7:10]), np.pi / 2.0, atol=1e-6)


def test_relative_trajectory_resolves_from_explicit_anchor():
    anchor_command = WholeBodyCommand(
        base_twist=[1.0, 2.0, 0.5],
        left_wrist=AxisAnglePose([0.2, 0.0, 0.4], [0.0, 0.0, 0.2]),
    )
    anchor = TrajectoryAnchor("odom_segment", AxisAnglePose([4.0, 5.0, 0.0]), anchor_command)
    endpoint = np.zeros(24)
    endpoint[:4] = [0.1, -0.2, 0.3, 0.05]
    endpoint[4:7] = [0.1, 0.0, 0.0]
    endpoint[7:10] = [0.0, 0.0, 0.2]
    trajectory = WholeBodyTrajectory([0.0, 0.5], np.stack([np.zeros(24), endpoint]), anchor)
    resolved = trajectory.resolve(0.5)
    np.testing.assert_allclose(resolved.base_twist[:2], [1.1, 1.8])
    assert np.isclose(resolved.base_twist[2], 0.8)
    np.testing.assert_allclose(resolved.left_wrist.position, [0.3, 0.0, 0.4])
    assert np.isclose(resolved.left_wrist.axis_angle[2], 0.4)


def test_current_next_window_and_synthetic_modes():
    trajectory = make_synthetic_trajectory(np.random.default_rng(7), mode="bimanual")
    current, next_target = trajectory.current_next(0.1, lookahead_s=0.2)
    assert current.as_vector().shape == (24,)
    assert next_target.as_vector().shape == (24,)
    assert np.all(np.isfinite(current.as_vector()))
    assert trajectory.mode == "bimanual"


def test_trajectory_rejects_bad_shapes_and_non_monotonic_times():
    with pytest.raises(ValueError, match="shape"):
        WholeBodyTrajectory([0.0, 1.0], np.zeros((2, 23)))
    with pytest.raises(ValueError, match="strictly increasing"):
        WholeBodyTrajectory([0.0, 0.0], np.zeros((2, 24)))
