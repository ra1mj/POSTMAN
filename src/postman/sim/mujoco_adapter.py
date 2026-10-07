"""Optional MuJoCo runtime adapter.

This module is intentionally independent from the manager environment so the
same UMMR/WBC code can be used in tests, mjlab and future hardware bridges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from postman.representations import RobotState, WbcResult


@dataclass(frozen=True)
class MuJoCoAdapterConfig:
    xml_path: Path
    actuator_mode: str = "torque"


class MuJoCoAdapter:
    def __init__(self, config: MuJoCoAdapterConfig):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[sim] to use MuJoCoAdapter") from exc
        self.mujoco = mujoco
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(config.xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nu:
            actuator_joints = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int64)
            self._actuator_qpos_indices = np.asarray(
                self.model.jnt_qposadr[actuator_joints], dtype=np.int64
            )
            self._actuator_qvel_indices = np.asarray(
                self.model.jnt_dofadr[actuator_joints], dtype=np.int64
            )
        else:
            self._actuator_qpos_indices = np.zeros(0, dtype=np.int64)
            self._actuator_qvel_indices = np.zeros(0, dtype=np.int64)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self, qpos=None, qvel=None) -> None:
        self.data.qpos[:] = 0.0 if qpos is None else np.asarray(qpos)
        self.data.qvel[:] = 0.0 if qvel is None else np.asarray(qvel)
        self.mujoco.mj_forward(self.model, self.data)

    def state(self) -> RobotState:
        raw_qpos = np.asarray(self.data.qpos).copy()
        raw_qvel = np.asarray(self.data.qvel).copy()
        q = raw_qpos[self._actuator_qpos_indices]
        qd = raw_qvel[self._actuator_qvel_indices]
        base_pose = np.zeros(7)
        base_pose[: min(7, q.size)] = q[: min(7, q.size)]
        base_twist = np.zeros(6)
        base_twist[: min(6, qd.size)] = qd[: min(6, qd.size)]
        return RobotState(
            q=q,
            qd=qd,
            base_pose=base_pose,
            base_twist=base_twist,
            raw_qpos=raw_qpos,
            raw_qvel=raw_qvel,
        )

    def apply_wbc_result(self, result: WbcResult) -> None:
        if self.config.actuator_mode == "torque":
            ctrl = result.tau_cmd
        elif self.config.actuator_mode == "position":
            ctrl = result.q_des
        elif self.config.actuator_mode == "velocity":
            ctrl = result.qd_des
        else:
            raise ValueError(f"unknown actuator mode {self.config.actuator_mode!r}")
        count = min(self.model.nu, ctrl.size)
        self.data.ctrl[:count] = ctrl[:count]

    def step(self, substeps: int = 1) -> None:
        for _ in range(substeps):
            self.mujoco.mj_step(self.model, self.data)
