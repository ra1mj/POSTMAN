"""Interactive hand-target whole-body following for the TIAGo++ demo.

This is the deterministic reference controller used by the drag demo.  It is
deliberately separate from the learned policy: the same local-frame UMMR target
that the RL-WBC consumes is generated directly from a draggable wrist target,
then the task-space WBC coordinates the base, torso, elbow posture and arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from postman.representations import AxisAnglePose, WholeBodyCommand
from postman.robots import TIAGO_PP, TiagoPPScene

from .tiago_taskspace import TiagoTaskSpaceWbc, TiagoTaskSpaceWbcConfig


@dataclass(frozen=True)
class HandFollowConfig:
    """Mapping from wrist displacement to supporting-body motion."""

    control_dt: float = 0.01
    base_xy_gain: float = 0.35
    base_yaw_gain: float = 0.65
    torso_z_gain: float = 0.25
    hand_deadzone: float = 0.06
    max_base_x: float = 0.35
    max_base_y: float = 0.22
    max_base_yaw: float = 0.45
    max_torso: float = 0.12
    target_filter_alpha: float = 0.6
    wheel_servo_gain: float = 20.0
    physics_substeps: int | None = None
    safe_start_pose: bool = True
    zero_gravity: bool = False

    def __post_init__(self) -> None:
        values = (
            self.control_dt,
            self.base_xy_gain,
            self.base_yaw_gain,
            self.torso_z_gain,
            self.max_base_x,
            self.max_base_y,
            self.max_base_yaw,
            self.max_torso,
            self.target_filter_alpha,
            self.wheel_servo_gain,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("hand-follow gains, limits and dt must be finite and positive")
        if not 0.0 <= self.hand_deadzone < 1.0:
            raise ValueError("hand_deadzone must be in [0, 1)")
        if self.target_filter_alpha > 1.0:
            raise ValueError("target_filter_alpha must be <= 1")
        if self.physics_substeps is not None and self.physics_substeps <= 0:
            raise ValueError("physics_substeps must be positive when provided")


class HandFollowController:
    """Run a real MuJoCo TIAGo++ simulation from a draggable wrist target.

    The target is expressed in world coordinates.  The controller converts it
    to a local UMMR offset anchored at reset, adds a small support motion for
    footprint and torso, and solves the full task stack at 100 Hz.  MuJoCo is
    advanced with actuator controls on every call to :meth:`step`; state is only
    written during :meth:`reset`.
    """

    sides = ("left", "right")

    def __init__(
        self,
        model=None,
        data=None,
        *,
        xml_path: str | Path | None = None,
        config: HandFollowConfig | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("Install postman-rl-wbc[sim] for the hand demo") from exc

        self.mujoco = mujoco
        if model is None:
            path = Path(xml_path) if xml_path is not None else TiagoPPScene().resolve_mjcf()
            spec = mujoco.MjSpec.from_file(str(path))
            if xml_path is None:
                # Menagerie's scene floor is at -0.99 m while the wheel bottoms
                # and the free-root home pose are at z=0. Correct the demo scene
                # before compiling, including static collision broadphase data.
                spec.geom("floor").pos[2] = 0.0
            model = spec.compile()
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.config = config or HandFollowConfig()
        if self.config.zero_gravity:
            self.model.opt.gravity[:] = 0.0
        self.wbc = TiagoTaskSpaceWbc(
            model,
            TiagoTaskSpaceWbcConfig(
                dt=self.config.control_dt,
                position_gain=5.0,
                wrist_weight=40.0,
                orientation_weight=0.4,
                elbow_weight=0.1,
                torso_weight=2.0,
                base_weight=4.0,
                posture_weight=0.02,
                wheel_sync_weight=10.0,
            ),
        )
        # The interactive reference must remain responsive and bounded.  The
        # learned controller can later retune these limits in simulation.
        self.wbc.velocity_limits[:4] = 4.0
        self.wbc.acceleration_limits[:4] = 12.0
        self._canonical_joint_ids = {
            name: index for index, name in enumerate(TIAGO_PP.joint_names)
        }
        self._actuator_canonical_ids = self._resolve_actuator_order()
        # kv=2000 from the asset makes small WBC speed changes kick the chassis
        # under contact. This compliant wheel servo prevents pitch feedback
        # without pinning the free base or changing the ground friction.
        for actuator_id, canonical_id in enumerate(self._actuator_canonical_ids):
            kp = model.actuator_gainprm[actuator_id, 0]
            if canonical_id >= 4 and (
                kp <= 0 or not np.isclose(model.actuator_biasprm[actuator_id, 1], -kp)
            ):
                raise ValueError("hand demo requires position actuators for non-wheel joints")
            if canonical_id < 4:
                if model.actuator_biasprm[actuator_id, 1] != 0:
                    raise ValueError("hand demo requires wheel velocity actuators")
                model.actuator_gainprm[actuator_id, 0] = self.config.wheel_servo_gain
                model.actuator_biasprm[actuator_id, 2] = -self.config.wheel_servo_gain
        ratio = self.config.control_dt / model.opt.timestep
        self._substeps = self.config.physics_substeps or int(round(ratio))
        if self._substeps < 1 or not np.isclose(
            self._substeps * model.opt.timestep, self.config.control_dt
        ):
            raise ValueError("physics substeps must exactly cover one control interval")
        self._targets: dict[str, np.ndarray] = {}
        self._filtered_targets: dict[str, np.ndarray] = {}
        self._target_rotations: dict[str, np.ndarray] = {}
        self._active_sides = self.sides
        self._last_command = WholeBodyCommand()
        self._last_result = None
        self.reset()

    def set_active_hands(self, sides: tuple[str, ...] | list[str]) -> None:
        """Choose which wrist targets contribute supporting base/torso motion."""
        selected = tuple(dict.fromkeys(sides))
        if not selected or any(side not in self.sides for side in selected):
            raise ValueError(f"active hands must be a non-empty subset of {self.sides}")
        self._active_sides = selected

    def _resolve_actuator_order(self) -> np.ndarray:
        ids = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = self.model.joint(joint_id).name
            if joint_name not in self._canonical_joint_ids:
                raise ValueError(f"actuator {actuator_id} targets unknown joint {joint_name!r}")
            ids.append(self._canonical_joint_ids[joint_name])
        ids = np.asarray(ids, dtype=np.int64)
        if self.model.nu != TIAGO_PP.dof or not np.array_equal(np.sort(ids), np.arange(25)):
            raise ValueError("the demo requires one actuator for each canonical TIAGo++ joint")
        return ids

    def reset(self) -> None:
        """Reset MuJoCo state and re-anchor all UMMR targets."""
        self.mujoco.mj_resetData(self.model, self.data)
        if self.config.safe_start_pose:
            # Start in front of the chassis with lift travel available in both
            # directions and bent elbows, clear of singular straight-arm poses.
            self.data.qpos[self.model.joint("torso_lift_joint").qposadr] = 0.17
            arm_pose = (1.3, 0.3, 0.0, 1.4, 0.0, 0.0, 0.0)
            for side in self.sides:
                for index, value in enumerate(arm_pose, start=1):
                    self.data.qpos[
                        self.model.joint(f"arm_{side}_{index}_joint").qposadr
                    ] = value
                for finger in ("left", "right"):
                    self.data.qpos[
                        self.model.joint(
                            f"gripper_{side}_{finger}_finger_joint"
                        ).qposadr
                    ] = 0.02
        self.mujoco.mj_forward(self.model, self.data)
        self.wbc.reset()
        reference = self.wbc.capture_reference(self.data)
        self._targets = {
            side: reference.positions[f"{side}_wrist"].copy() for side in self.sides
        }
        self._filtered_targets = {side: p.copy() for side, p in self._targets.items()}
        self._position_setpoint = self.data.qpos[self.wbc.qpos_indices].copy()
        self._target_rotations = {
            side: reference.rotations[f"{side}_wrist"].copy() for side in self.sides
        }
        self._last_command = WholeBodyCommand()
        self._last_result = None

        # A paused reset must hold this posture rather than command zeros.
        self.data.ctrl[:] = self._position_setpoint[self._actuator_canonical_ids]
        self.data.ctrl[self._actuator_canonical_ids < 4] = 0.0

    @property
    def target(self) -> dict[str, np.ndarray]:
        return {side: target.copy() for side, target in self._targets.items()}

    def set_target(
        self,
        side: str,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
    ) -> None:
        """Set a left/right wrist target in world coordinates."""
        if side not in self.sides:
            raise ValueError(f"side must be one of {self.sides}, got {side!r}")
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.size != 3 or not np.isfinite(position).all():
            raise ValueError("position must be a finite 3-vector")
        self._targets[side] = position.copy()
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            if (
                not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0)
            ):
                raise ValueError("rotation must be a proper rotation matrix")
            self._target_rotations[side] = rotation.copy()

    def _local_offset(self, side: str) -> np.ndarray:
        reference = self.wbc.reference
        assert reference is not None
        yaw = reference.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        local_to_world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return local_to_world.T @ (
            self._filtered_targets[side] - reference.positions[f"{side}_wrist"]
        )

    def command(self) -> WholeBodyCommand:
        """Build the local UMMR command and distribute reach across the body."""
        offsets = {side: self._local_offset(side) for side in self.sides}
        side = max(self._active_sides, key=lambda item: np.linalg.norm(offsets[item]))
        offset = offsets[side]
        magnitude = float(np.linalg.norm(offset))
        active = max(0.0, magnitude - self.config.hand_deadzone)
        support = min(1.0, active / max(self.config.hand_deadzone, 1e-6))
        base = np.array(
            [
                np.clip(self.config.base_xy_gain * support * offset[0],
                        -self.config.max_base_x, self.config.max_base_x),
                np.clip(self.config.base_xy_gain * support * offset[1],
                        -self.config.max_base_y, self.config.max_base_y),
                np.clip(self.config.base_yaw_gain * support * offset[1],
                        -self.config.max_base_yaw, self.config.max_base_yaw),
            ]
        )
        torso = np.array(
            [np.clip(self.config.torso_z_gain * support * offset[2],
                     -self.config.max_torso, self.config.max_torso)]
        )
        reference = self.wbc.reference
        c, s = np.cos(reference.yaw), np.sin(reference.yaw)
        local_to_world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        poses = {
            f"{side}_wrist": AxisAnglePose(
                position=offsets[side],
                axis_angle=local_to_world.T @ self.wbc._rotation_error(
                    self._target_rotations[side], reference.rotations[f"{side}_wrist"]
                ),
            )
            for side in self.sides
        }
        wrist_kwargs = {
            **poses,
            "base_twist": base,
            "torso": torso,
            "grippers": np.array([0.02, 0.02]),
        }
        return WholeBodyCommand(**wrist_kwargs)

    def _apply_result(self, result) -> None:
        assert result.actuator_targets is not None
        # The checked-in scene_position asset has velocity wheel actuators and
        # position actuators for all remaining joints. Resolve by joint name so
        # the demo remains safe if XML declaration order changes.
        q = self.data.qpos[self.wbc.qpos_indices]
        # Integrate commanded velocity. Replacing this with q_measured+dt*v on
        # every tick collapses the servo error and slows damped arms drastically.
        self._position_setpoint[4:] += self.config.control_dt * result.qd_des[4:]
        lead = np.full(25, 0.12)
        lead[4] = 0.02
        lead[[14, 15, 23, 24]] = 0.005
        self._position_setpoint = np.clip(self._position_setpoint, q - lead, q + lead)
        self._position_setpoint = np.clip(
            self._position_setpoint, self.wbc.lower_q, self.wbc.upper_q
        )
        for actuator_id, canonical_id in enumerate(self._actuator_canonical_ids):
            if canonical_id < 4:
                value = result.qd_des[canonical_id]
            else:
                kp = self.model.actuator_gainprm[actuator_id, 0]
                bias = self.data.qfrc_bias[self.wbc.qvel_indices[canonical_id]]
                value = self._position_setpoint[canonical_id] + bias / kp
            if self.model.actuator_ctrllimited[actuator_id]:
                value = np.clip(value, *self.model.actuator_ctrlrange[actuator_id])
            self.data.ctrl[actuator_id] = value

    def step(self) -> dict[str, float]:
        """Track the current target for one 100 Hz control interval."""
        alpha = self.config.target_filter_alpha
        for side in self.sides:
            self._filtered_targets[side] += alpha * (
                self._targets[side] - self._filtered_targets[side]
            )
        command = self.command()
        result = self.wbc.compute(self.data, command)
        self._apply_result(result)
        for _ in range(self._substeps):
            self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        self._last_command = command
        self._last_result = result
        return self.metrics()

    def metrics(self) -> dict[str, float]:
        if self.wbc.reference is None:
            return {}
        errors = {
            f"{side}_hand_error": float(
                np.linalg.norm(
                    self._targets[side] - self.data.xpos[self.wbc.body_ids[f"{side}_wrist"]]
                )
            )
            for side in self.sides
        }
        base = self.data.xpos[self.wbc.body_ids["footprint"]]
        base_rotation = self.data.xmat[self.wbc.body_ids["footprint"]].reshape(3, 3)
        base_start = self.wbc.reference.positions["footprint"]
        q = self.data.qpos[self.wbc.qpos_indices]
        errors.update(
            {
                "base_motion": float(np.linalg.norm(base[:2] - base_start[:2])),
                "torso_motion": float(abs(q[4] - self.wbc.reference.joint_position[4])),
                "base_target_x": float(self._last_command.base_twist[0]),
                "torso_target": float(self._last_command.torso[0]),
                "base_height_m": float(base[2]),
                "base_tilt_rad": float(np.arccos(np.clip(base_rotation[2, 2], -1, 1))),
            }
        )
        return errors


__all__ = ["HandFollowConfig", "HandFollowController"]
