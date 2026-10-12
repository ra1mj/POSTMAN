"""MuJoCo task-space QP baseline for the checked-in TIAGo++ model contract.

This controller is an explicit kinematic baseline for RL comparison. Targets
are offsets from a captured segment reference: wrist/elbow translations and
axis-angle rotations use its yaw-aligned local frame; ``base_twist`` is a
legacy name for the segment's finite SE(2) offset. Torso is a lift offset and
grippers are absolute individual-finger openings in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from postman.representations import WbcResult, WholeBodyCommand
from postman.robots import TIAGO_PP

from .batched import mixed_actuator_targets

TIAGO_TASK_FRAMES = {
    "left_wrist": "arm_left_7_link",
    "right_wrist": "arm_right_7_link",
    "left_elbow": "arm_left_4_link",
    "right_elbow": "arm_right_4_link",
    "chest": "torso_lift_link",
    "footprint": "base_link",
}


def wheel_to_base_matrix(radius: float = 0.0762, half_track: float = 0.223175):
    """Map FR, FL, RR, RL wheel rates to local ``[vx, vy, yaw_rate]``.

    Menagerie's four cylinders have parallel axles, no roller or steering DOFs.
    It is a skid-steer approximation: lateral velocity is deliberately zero.
    """
    if radius <= 0 or half_track <= 0:
        raise ValueError("wheel radius and half track must be positive")
    return np.array(
        [[1, 1, 1, 1], [0, 0, 0, 0],
         [1 / half_track, -1 / half_track, 1 / half_track, -1 / half_track]],
        dtype=float,
    ) * (radius / 4)


def _yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rotation_exp(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle < 1e-10:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class TiagoTaskSpaceWbcConfig:
    dt: float = 0.01
    regularization: float = 1e-3
    iterations: int = 64
    position_gain: float = 4.0
    orientation_gain: float = 3.0
    base_gain: float = 2.0
    wrist_weight: float = 20.0
    orientation_weight: float = 4.0
    elbow_weight: float = 2.0
    torso_weight: float = 2.0
    base_weight: float = 2.0
    gripper_weight: float = 5.0
    posture_weight: float = 0.05
    wheel_sync_weight: float = 10.0
    wheel_radius: float = 0.0762
    half_track: float = 0.223175

    def __post_init__(self):
        if self.dt <= 0 or self.regularization <= 0 or self.iterations <= 0:
            raise ValueError("dt, regularization and iterations must be positive")
        values = (
            self.position_gain, self.orientation_gain, self.base_gain, self.wrist_weight,
            self.orientation_weight, self.elbow_weight, self.torso_weight,
            self.base_weight, self.gripper_weight, self.posture_weight, self.wheel_sync_weight,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("task gains and weights must be finite and positive")


@dataclass(frozen=True)
class TiagoReference:
    """Persistent anchor for one target segment, all poses in world coordinates."""

    joint_position: np.ndarray
    positions: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    yaw: float


class TiagoTaskSpaceWbc:
    frame_names = TIAGO_TASK_FRAMES

    def __init__(self, model, config: TiagoTaskSpaceWbcConfig | None = None):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.config = config or TiagoTaskSpaceWbcConfig()
        self.joint_ids = np.array([model.joint(name).id for name in TIAGO_PP.joint_names])
        self.qpos_indices = np.asarray(model.jnt_qposadr[self.joint_ids])
        self.qvel_indices = np.asarray(model.jnt_dofadr[self.joint_ids])
        self.body_ids = {name: model.body(frame).id for name, frame in self.frame_names.items()}
        self.lower_q = np.where(model.jnt_limited[self.joint_ids],
                                model.jnt_range[self.joint_ids, 0], -np.inf)
        self.upper_q = np.where(model.jnt_limited[self.joint_ids],
                                model.jnt_range[self.joint_ids, 1], np.inf)
        self.velocity_limits = np.ones(25)
        self.velocity_limits[:4] = 5.0
        self.velocity_limits[4] = 0.15
        self.velocity_limits[5:7] = 0.7
        self.velocity_limits[[14, 15, 23, 24]] = 0.05
        self.acceleration_limits = np.full(25, 4.0)
        self.acceleration_limits[:4] = 20.0
        self.acceleration_limits[4] = 1.0
        self.acceleration_limits[[14, 15, 23, 24]] = 0.3
        self.wheel_map = wheel_to_base_matrix(self.config.wheel_radius, self.config.half_track)
        self.reference: TiagoReference | None = None
        self.last_velocity = np.zeros(25)
        self.task_errors: dict[str, float] = {}

    def reset(self):
        self.reference = None
        self.last_velocity.fill(0)

    def capture_reference(self, data) -> TiagoReference:
        """Capture once per segment/reset, never once per control tick."""
        self.mujoco.mj_forward(self.model, data)
        rotations = {key: data.xmat[body].reshape(3, 3).copy()
                     for key, body in self.body_ids.items()}
        base_rotation = rotations["footprint"]
        reference = TiagoReference(
            data.qpos[self.qpos_indices].copy(),
            {key: data.xpos[body].copy() for key, body in self.body_ids.items()},
            rotations,
            float(np.arctan2(base_rotation[1, 0], base_rotation[0, 0])),
        )
        self.reference = reference
        self.last_velocity.fill(0)
        return reference

    def _rotation_error(self, target, current):
        quaternion = np.zeros(4)
        self.mujoco.mju_mat2Quat(quaternion, (target @ current.T).reshape(9))
        if quaternion[0] < 0:
            quaternion *= -1
        length = np.linalg.norm(quaternion[1:])
        if length < 1e-10:
            return 2 * quaternion[1:]
        return quaternion[1:] * (2 * np.arctan2(length, quaternion[0]) / length)

    def frame_jacobian(self, data, frame: str):
        """World translational/rotational Jacobians in canonical 25-D rates.

        Wheel columns contain the virtual mobile-base derivative, not the
        unhelpful wheel-spin derivative of an arm body in a floating-base tree.
        """
        body = self.body_ids[frame]
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        self.mujoco.mj_jacBody(self.model, data, jp, jr, body)
        jp = jp[:, self.qvel_indices].copy()
        jr = jr[:, self.qvel_indices].copy()
        base = self.body_ids["footprint"]
        base_rotation = data.xmat[base].reshape(3, 3)
        yaw = np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
        forward = np.array([np.cos(yaw), np.sin(yaw), 0])
        yaw_axis = np.array([0.0, 0.0, 1.0])
        lever = data.xpos[body] - data.xpos[base]
        jp[:, :4] = (
            forward[:, None] * self.wheel_map[0]
            + np.cross(yaw_axis, lever)[:, None] * self.wheel_map[2]
        )
        jr[:, :4] = yaw_axis[:, None] * self.wheel_map[2]
        return jp, jr

    def build_tasks(self, data, command: WholeBodyCommand, reference=None):
        """Return ``J[M,25], target_velocity[M], weights[M]``."""
        if not np.isfinite(command.as_vector()).all():
            raise ValueError("whole-body command must be finite")
        reference = reference or self.reference or self.capture_reference(data)
        cfg = self.config
        q = data.qpos[self.qpos_indices]
        local_to_world = _yaw_rotation(reference.yaw)
        rows, targets, weights = [], [], []
        self.task_errors = {}

        def append(name, jacobian, error, gain, weight):
            error = np.asarray(error).reshape(-1)
            rows.append(jacobian)
            targets.append(gain * error)
            # UMMR command weights are optional multiplicative priorities. A
            # zero value intentionally disables a task for one segment.
            weights.append(np.full(error.size, weight * command.weights.get(name, 1.0)))
            self.task_errors[name] = float(np.linalg.norm(error))

        for side in ("left", "right"):
            name = f"{side}_wrist"
            pose = getattr(command, name)
            body = self.body_ids[name]
            jp, jr = self.frame_jacobian(data, name)
            target_position = reference.positions[name] + local_to_world @ pose.position
            target_rotation = (
                local_to_world @ _rotation_exp(pose.axis_angle)
                @ local_to_world.T @ reference.rotations[name]
            )
            append(name, jp, target_position - data.xpos[body],
                   cfg.position_gain, cfg.wrist_weight)
            append(f"{name}_orientation", jr,
                   self._rotation_error(target_rotation, data.xmat[body].reshape(3, 3)),
                   cfg.orientation_gain, cfg.orientation_weight)
            name = f"{side}_elbow"
            body = self.body_ids[name]
            jp, _ = self.frame_jacobian(data, name)
            target_position = reference.positions[name] + local_to_world @ getattr(command, name)
            append(name, jp, target_position - data.xpos[body],
                   cfg.position_gain, cfg.elbow_weight)

        base = self.body_ids["footprint"]
        jp, jr = self.frame_jacobian(data, "footprint")
        offset = command.base_twist
        base_target = reference.positions["footprint"] + local_to_world @ np.r_[offset[:2], 0.0]
        rotation = data.xmat[base].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        append("footprint", np.vstack((jp[:2], jr[2:3])),
               np.r_[base_target[:2] - data.xpos[base, :2],
                     _wrap_angle(reference.yaw + offset[2] - yaw)],
               cfg.base_gain, cfg.base_weight)
        identity = np.eye(25)
        append("torso", identity[4:5],
               [reference.joint_position[4] + command.torso[0] - q[4]],
               cfg.position_gain, cfg.torso_weight)
        fingers = np.array([14, 15, 23, 24])
        gripper_target = np.repeat(command.grippers, 2)
        append("grippers", identity[fingers], gripper_target - q[fingers],
               cfg.position_gain, cfg.gripper_weight)
        append("posture", identity[4:], reference.joint_position[4:] - q[4:],
               1.0, cfg.posture_weight)
        append("wheel_sync", np.stack((identity[0] - identity[2], identity[1] - identity[3])),
               np.zeros(2), 1.0, cfg.wheel_sync_weight)
        return np.vstack(rows), np.concatenate(targets), np.concatenate(weights)

    def compute(self, data, command: WholeBodyCommand, reference=None) -> WbcResult:
        jacobian, target, weights = self.build_tasks(data, command, reference)
        cfg = self.config
        q = data.qpos[self.qpos_indices].copy()
        lower = np.maximum(-self.velocity_limits, (self.lower_q - q) / cfg.dt)
        upper = np.minimum(self.velocity_limits, (self.upper_q - q) / cfg.dt)
        # Rate limits constrain changes in commanded velocity. Joint feasibility
        # wins if the previous command has become incompatible with a hard limit.
        previous = np.clip(self.last_velocity, lower, upper)
        lower = np.maximum(lower, previous - cfg.dt * self.acceleration_limits)
        upper = np.minimum(upper, previous + cfg.dt * self.acceleration_limits)
        hessian = jacobian.T @ (weights[:, None] * jacobian) + cfg.regularization * np.eye(25)
        rhs = jacobian.T @ (weights * target)
        if np.any(lower > upper):
            velocity, status = np.zeros(25), "infeasible_bounds"
        else:
            velocity = np.clip(np.linalg.solve(hessian, rhs), lower, upper)
            step = 1 / np.abs(hessian).sum(axis=1).max()
            for _ in range(cfg.iterations):
                velocity = np.clip(velocity - step * (hessian @ velocity - rhs), lower, upper)
            status = "solved_projected"
        self.last_velocity = velocity.copy()
        actuator_targets = mixed_actuator_targets(q, velocity, cfg.dt, self.lower_q, self.upper_q)
        q_des = q + cfg.dt * velocity
        q_des[:4] = q[:4]
        violation = float(np.maximum(np.maximum(lower - velocity, velocity - upper), 0).max())
        result = WbcResult(
            q_des=q_des, qd_des=velocity, tau_ff=np.zeros(25), tau_cmd=np.zeros(25),
            base_cmd=self.wheel_map @ velocity[:4], solver_status=status,
            constraint_violation=violation,
            diagnostics={"task_errors": dict(self.task_errors),
                         "weighted_residual": float(
                             np.sum(weights * (jacobian @ velocity - target) ** 2)
                         ),
                         "base_model": "skid_steer_no_lateral_dof",
                         "command_frame": "segment_start_yaw_frame"},
            actuator_targets=actuator_targets,
        )
        return result
