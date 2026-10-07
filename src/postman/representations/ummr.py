"""Typed representations shared by policies, WBCs, datasets and deployments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

UMMR_SCHEMA_VERSION = "UMMR-1.0"


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values, got {array.size}")
    return array.copy()


@dataclass(frozen=True)
class AxisAnglePose:
    """A compact SE(3) pose represented by position and axis-angle rotation."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis_angle: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        object.__setattr__(self, "axis_angle", _vector(self.axis_angle, 3, "axis_angle"))

    def as_vector(self) -> np.ndarray:
        return np.concatenate((self.position, self.axis_angle))

    @classmethod
    def from_vector(cls, value: Any) -> AxisAnglePose:
        vector = _vector(value, 6, "pose")
        return cls(vector[:3], vector[3:])


@dataclass
class WholeBodyCommand:
    """Task-space command consumed by a whole-body controller."""

    base_twist: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torso: np.ndarray = field(default_factory=lambda: np.zeros(1))
    left_wrist: AxisAnglePose = field(default_factory=AxisAnglePose)
    right_wrist: AxisAnglePose = field(default_factory=AxisAnglePose)
    left_elbow: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right_elbow: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grippers: np.ndarray = field(default_factory=lambda: np.zeros(2))
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_twist = _vector(self.base_twist, 3, "base_twist")
        self.torso = _vector(self.torso, 1, "torso")
        self.left_elbow = _vector(self.left_elbow, 3, "left_elbow")
        self.right_elbow = _vector(self.right_elbow, 3, "right_elbow")
        self.grippers = _vector(self.grippers, 2, "grippers")

    @property
    def dim(self) -> int:
        return 24

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.base_twist,
                self.torso,
                self.left_wrist.as_vector(),
                self.right_wrist.as_vector(),
                self.left_elbow,
                self.right_elbow,
                self.grippers,
            )
        )

    @classmethod
    def from_vector(cls, value: Any) -> WholeBodyCommand:
        vector = _vector(value, 24, "whole-body command")
        index = 0
        base_twist = vector[index : index + 3]
        index += 3
        torso = vector[index : index + 1]
        index += 1
        left_wrist = AxisAnglePose.from_vector(vector[index : index + 6])
        index += 6
        right_wrist = AxisAnglePose.from_vector(vector[index : index + 6])
        index += 6
        left_elbow = vector[index : index + 3]
        index += 3
        right_elbow = vector[index : index + 3]
        index += 3
        grippers = vector[index : index + 2]
        return cls(base_twist, torso, left_wrist, right_wrist, left_elbow, right_elbow, grippers)


@dataclass
class WholeBodyAction:
    """Normalized residual applied to a reference :class:`WholeBodyCommand`."""

    residual: np.ndarray
    scale: float = 1.0

    def __post_init__(self) -> None:
        self.residual = _vector(self.residual, 24, "residual")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def clipped(self, limit: float = 1.0) -> WholeBodyAction:
        return WholeBodyAction(np.clip(self.residual, -limit, limit), self.scale)

    def apply(self, command: WholeBodyCommand) -> WholeBodyCommand:
        return WholeBodyCommand.from_vector(command.as_vector() + self.scale * self.residual)


@dataclass
class RobotState:
    """Simulator or hardware state passed to the WBC."""

    q: np.ndarray
    qd: np.ndarray
    base_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    base_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ee_poses: Mapping[str, AxisAnglePose] = field(default_factory=dict)
    privileged: Mapping[str, Any] = field(default_factory=dict)
    raw_qpos: np.ndarray | None = None
    raw_qvel: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float64).reshape(-1).copy()
        self.qd = np.asarray(self.qd, dtype=np.float64).reshape(-1).copy()
        if self.q.shape != self.qd.shape:
            raise ValueError("q and qd must have the same shape")
        self.base_pose = _vector(self.base_pose, 7, "base_pose")
        self.base_twist = _vector(self.base_twist, 6, "base_twist")
        if self.raw_qpos is not None:
            self.raw_qpos = np.asarray(self.raw_qpos, dtype=np.float64).reshape(-1).copy()
        if self.raw_qvel is not None:
            self.raw_qvel = np.asarray(self.raw_qvel, dtype=np.float64).reshape(-1).copy()


@dataclass
class WbcResult:
    """Controller output and diagnostics."""

    q_des: np.ndarray
    qd_des: np.ndarray
    tau_ff: np.ndarray
    tau_cmd: np.ndarray
    base_cmd: np.ndarray
    solver_status: str = "unknown"
    constraint_violation: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q_des = np.asarray(self.q_des, dtype=np.float64).reshape(-1)
        self.qd_des = np.asarray(self.qd_des, dtype=np.float64).reshape(-1)
        self.tau_ff = np.asarray(self.tau_ff, dtype=np.float64).reshape(-1)
        self.tau_cmd = np.asarray(self.tau_cmd, dtype=np.float64).reshape(-1)
        self.base_cmd = np.asarray(self.base_cmd, dtype=np.float64).reshape(-1)
        if self.constraint_violation < 0:
            raise ValueError("constraint_violation cannot be negative")
