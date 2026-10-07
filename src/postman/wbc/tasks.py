"""Helpers for turning robot kinematics into WBC linear tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import LinearTask


def cartesian_task(
    name: str,
    jacobian: Any,
    error: Any,
    *,
    gain: float = 1.0,
    weight: float = 1.0,
) -> LinearTask:
    """Create ``J dq ~= K e`` from a task-space error and Jacobian."""

    return LinearTask(
        name=name,
        jacobian=np.asarray(jacobian, dtype=np.float64),
        target=gain * np.asarray(error, dtype=np.float64).reshape(-1),
        weight=weight,
    )


def joint_limit_tasks(
    dof: int,
    q: Any,
    lower: Any,
    upper: Any,
    *,
    margin: float = 0.05,
    weight: float = 10.0,
) -> list[LinearTask]:
    """Return active one-dimensional tasks that push joints away from limits."""

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if any(array.size != dof for array in (q, lower, upper)):
        raise ValueError("joint limit arrays must have the configured dof")
    tasks = []
    for index in range(dof):
        row = np.zeros((1, dof))
        row[0, index] = 1.0
        if q[index] < lower[index] + margin:
            tasks.append(LinearTask(f"joint_{index}_lower_limit", row, np.array([margin]), weight))
        elif q[index] > upper[index] - margin:
            tasks.append(LinearTask(f"joint_{index}_upper_limit", row, np.array([-margin]), weight))
    return tasks


def pinocchio_frame_task(*args, **kwargs) -> LinearTask:
    """Build a frame task when Pinocchio is installed.

    The function keeps Pinocchio optional and deliberately accepts the model,
    data and frame name as positional arguments so robot plugins can select the
    exact Pinocchio API version they support.
    """

    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError("Install postman-rl-wbc[control] for Pinocchio tasks") from exc
    model, data, q, frame_name, target_pose = args[:5]
    frame_id = model.getFrameId(frame_name)
    pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
    pin.updateFramePlacements(model, data)
    current = data.oMf[frame_id]
    error = pin.log6(current.actInv(target_pose)).vector
    jacobian = pin.computeFrameJacobian(
        model, data, np.asarray(q, dtype=np.float64), frame_id, pin.LOCAL
    )
    return cartesian_task(frame_name, jacobian, error, **kwargs)
