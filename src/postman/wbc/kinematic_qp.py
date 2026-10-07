"""A dependency-light kinematic QP WBC with optional OSQP acceleration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from postman.representations import RobotState, WbcResult, WholeBodyCommand

from .base import ImpedanceConfig, LinearTask, impedance_torque


@dataclass(frozen=True)
class KinematicQpWbcConfig:
    dof: int
    dt: float = 0.01
    max_delta_q: float = 0.25
    regularization: float = 1e-4
    impedance: ImpedanceConfig = ImpedanceConfig()
    use_osqp: bool = True
    lower_q: np.ndarray | None = None
    upper_q: np.ndarray | None = None


class KinematicQpWbc:
    """Solve a stack of linearized task objectives under per-joint step limits.

    The default task is deliberately identity-mapped so the backend is usable
    without a robot-specific kinematics dependency. Robot adapters should pass
    Cartesian ``LinearTask`` objects built from Pinocchio Jacobians.
    """

    def __init__(self, config: KinematicQpWbcConfig):
        if config.dof <= 0 or config.dt <= 0:
            raise ValueError("dof and dt must be positive")
        self.config = config
        self.last_status = "initialized"

    def reset(self) -> None:
        self.last_status = "reset"

    def _default_tasks(self, state: RobotState, command: WholeBodyCommand) -> list[LinearTask]:
        target = command.as_vector()
        rows = min(target.size, self.config.dof)
        jacobian = np.zeros((rows, self.config.dof), dtype=np.float64)
        jacobian[:, :rows] = np.eye(rows)
        return [LinearTask("ummr_residual_fallback", jacobian, target[:rows], 1.0)]

    def _solve(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        n = hessian.shape[0]
        if self.config.use_osqp:
            try:
                import osqp
                from scipy import sparse

                solver = osqp.OSQP()
                solver.setup(
                    P=sparse.csc_matrix((hessian + hessian.T) / 2),
                    q=gradient,
                    A=sparse.eye(n, format="csc"),
                    l=lower,
                    u=upper,
                    verbose=False,
                    polish=True,
                )
                result = solver.solve()
                if result.x is not None and result.info.status_val in (1, 2):
                    return np.asarray(result.x), "solved_osqp"
            except ImportError:
                pass
            except Exception as exc:  # pragma: no cover - solver-specific fallback
                self.last_status = f"osqp_error:{type(exc).__name__}"

        try:
            delta = np.linalg.solve(hessian, -gradient)
            return np.clip(delta, lower, upper), "solved_numpy"
        except np.linalg.LinAlgError:
            return np.zeros(n), "singular_fallback"

    def compute(
        self,
        state: RobotState,
        command: WholeBodyCommand,
        tasks: Sequence[LinearTask] | None = None,
    ) -> WbcResult:
        if state.q.size != self.config.dof:
            raise ValueError(f"expected state dof {self.config.dof}, got {state.q.size}")
        active_tasks = list(tasks) if tasks is not None else self._default_tasks(state, command)
        if not active_tasks:
            raise ValueError("at least one WBC task is required")
        hessian = self.config.regularization * np.eye(self.config.dof)
        gradient = np.zeros(self.config.dof)
        for task in active_tasks:
            if task.jacobian.shape[1] != self.config.dof:
                raise ValueError(f"task {task.name!r} has the wrong number of columns")
            weight = task.weight
            hessian += weight * task.jacobian.T @ task.jacobian
            gradient += -weight * task.jacobian.T @ task.target
        step_lower = -self.config.max_delta_q * np.ones(self.config.dof)
        step_upper = self.config.max_delta_q * np.ones(self.config.dof)
        if self.config.lower_q is not None:
            step_lower = np.maximum(step_lower, np.asarray(self.config.lower_q) - state.q)
        if self.config.upper_q is not None:
            step_upper = np.minimum(step_upper, np.asarray(self.config.upper_q) - state.q)
        if np.any(step_lower > step_upper):
            return WbcResult(
                q_des=state.q,
                qd_des=np.zeros_like(state.q),
                tau_ff=np.zeros_like(state.q),
                tau_cmd=np.zeros_like(state.q),
                base_cmd=command.base_twist,
                solver_status="infeasible_bounds",
                constraint_violation=float(np.maximum(step_lower - step_upper, 0).max()),
            )
        delta_q, status = self._solve(hessian, gradient, step_lower, step_upper)
        q_des = state.q + delta_q
        qd_des = delta_q / self.config.dt
        provisional = WbcResult(
            q_des=q_des,
            qd_des=qd_des,
            tau_ff=np.zeros_like(state.q),
            tau_cmd=np.zeros_like(state.q),
            base_cmd=command.base_twist,
            solver_status=status,
            constraint_violation=float(
                np.maximum(np.abs(delta_q) - self.config.max_delta_q, 0).max()
            ),
            diagnostics={"task_names": [task.name for task in active_tasks]},
        )
        provisional.tau_cmd = impedance_torque(state, provisional, self.config.impedance)
        self.last_status = status
        return provisional
