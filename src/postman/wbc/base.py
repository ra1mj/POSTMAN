"""WBC interfaces and reusable impedance helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from postman.representations import RobotState, WbcResult, WholeBodyCommand


@dataclass(frozen=True)
class ImpedanceConfig:
    kp: float | np.ndarray = 30.0
    kd: float | np.ndarray = 2.0
    torque_limit: float | np.ndarray = 100.0


@dataclass(frozen=True)
class LinearTask:
    """Linearized task ``J * dq ~= target`` used by the QP backend."""

    name: str
    jacobian: np.ndarray
    target: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        jacobian = np.asarray(self.jacobian, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64).reshape(-1)
        if jacobian.ndim != 2 or jacobian.shape[0] != target.size:
            raise ValueError("task Jacobian and target dimensions do not match")
        if self.weight <= 0:
            raise ValueError("task weight must be positive")
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "target", target)


class WbcController(Protocol):
    def reset(self) -> None: ...

    def compute(
        self,
        state: RobotState,
        command: WholeBodyCommand,
        tasks: Sequence[LinearTask] | None = None,
    ) -> WbcResult: ...


def impedance_torque(
    state: RobotState,
    result: WbcResult,
    config: ImpedanceConfig,
) -> np.ndarray:
    kp = np.broadcast_to(config.kp, state.q.shape)
    kd = np.broadcast_to(config.kd, state.q.shape)
    limit = np.broadcast_to(config.torque_limit, state.q.shape)
    torque = result.tau_ff + kp * (result.q_des - state.q) + kd * (result.qd_des - state.qd)
    return np.clip(torque, -limit, limit)


@dataclass
class DynamicWbcBackend:
    """Reserved dynamic backend with an explicit, safe failure mode."""

    reason: str = "TSID/ARC-OPT backend is not enabled in the pure-Python v1 core"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def reset(self) -> None:
        self.diagnostics.clear()

    def compute(self, state: RobotState, command: WholeBodyCommand, tasks=None) -> WbcResult:
        zeros = np.zeros_like(state.q)
        self.diagnostics["requested_command_dim"] = command.dim
        return WbcResult(
            q_des=state.q,
            qd_des=zeros,
            tau_ff=zeros,
            tau_cmd=zeros,
            base_cmd=command.base_twist,
            solver_status="backend_unavailable",
            diagnostics={"reason": self.reason},
        )
