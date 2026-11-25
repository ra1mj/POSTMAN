"""Shared Cartesian tracking, policy observations and action decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import ACTION_SIZE, BASE_RATE, JOINT_RATE, JOINTS, LunaModel

OBSERVATION_SIZE = 89
POLICY_VERSION = "gento-luna-v3-680-residual-wbc-v1"
POSITION_OFFSET = np.array([0.15, 0.12, 0.12])
ROTATION_OFFSET = 0.30
HEAD_OFFSET = 0.25
BASE_DEADBAND = 1e-5
LEAD_TIME = 0.10


def default_policy() -> Path:
    return Path(__file__).with_name("luna_680_policy.npz")


@dataclass(frozen=True)
class Target:
    hands: np.ndarray
    rotations: np.ndarray
    base: np.ndarray
    head: np.ndarray


def rotation_xyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    sr, sp, sy = np.sin([roll, pitch, yaw])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    return np.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ], axis=-1).reshape(*rpy.shape[:-1], 3, 3)


def rotation_error(goal: np.ndarray, actual: np.ndarray) -> np.ndarray:
    relative = goal @ actual.swapaxes(-1, -2)
    skew = 0.5 * (relative - relative.swapaxes(-1, -2))
    return skew[..., (2, 0, 1), (1, 2, 0)]


def offset_target(anchor: Target, xyz: np.ndarray, rpy: np.ndarray, base: np.ndarray,
                  head: np.ndarray) -> Target:
    return Target(anchor.hands + xyz, rotation_xyz(rpy) @ anchor.rotations,
                  base, anchor.head + head)


def anchor_target(model: LunaModel, q: np.ndarray) -> Target:
    hands, rotations, _ = model.kinematics(q)
    return Target(hands, rotations, np.zeros((len(q), 3)), q[:, -2:].copy())


@dataclass(frozen=True)
class Tracking:
    observation: np.ndarray
    reference: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    head_error: np.ndarray


def tracking(model: LunaModel, q: np.ndarray, qd: np.ndarray, base_vel: np.ndarray,
             goal: Target) -> Tracking:
    hands, rotations, jacobian = model.kinematics(q)
    xyz = goal.hands - hands
    rpy = rotation_error(goal.rotations, rotations)
    head = goal.head - q[:, -2:]
    task_velocity = np.concatenate((2.2 * xyz, 1.4 * rpy), axis=-1).reshape(-1, 12)
    weights = np.tile([1, 1, 1, 0.36, 0.36, 0.36], 2)
    # Penalize the load-bearing BODY axes more strongly than the arm axes.
    freedom = np.repeat([0.14, 1., 1., 0.], [6, 7, 7, 2])
    weighted = jacobian * weights[None, :, None]
    transpose = weighted.swapaxes(1, 2) * freedom[None, :, None]
    normal = weighted @ transpose + np.eye(12) * 0.0225
    correction = np.linalg.solve(normal, (task_velocity * weights)[..., None])
    rate = (transpose @ correction).squeeze(axis=-1)
    rate[:, -2:] = 2.0 * head
    # Slow the complete coupled update when any joint saturates.
    normalized = rate / JOINT_RATE
    normalized /= np.maximum(1., np.abs(normalized).max(axis=1, keepdims=True))
    reference = np.concatenate((goal.base / BASE_RATE, normalized), axis=1).clip(-1., 1.)
    observation = np.concatenate((q, qd / JOINT_RATE, base_vel / BASE_RATE,
                                  goal.base / BASE_RATE, xyz.reshape(-1, 6) * 5,
                                  rpy.reshape(-1, 6) * 2, head * 4, reference), axis=1)
    return Tracking(observation.clip(-5., 5.).astype(np.float32), reference, xyz, rpy, head)


def decode(control: Tracking, action: np.ndarray, q: np.ndarray, bounds: np.ndarray,
           base_cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a bounded residual into RDK body twist and FX angle targets."""
    action = np.asarray(action)
    if action.shape != control.reference.shape or not np.isfinite(action).all():
        raise ValueError("invalid Luna actor output")
    residual = action.clip(-1., 1.)
    reference = control.reference
    # Preserve a zero base request and the operator's requested direction.
    base = reference[:, :3] * (1 + 0.18 * residual[:, :3]) * BASE_RATE
    base = np.clip(base, -base_cap, base_cap)
    hand_error = np.linalg.norm(control.position_error, axis=-1)
    hand_error += 0.18 * np.linalg.norm(control.orientation_error, axis=-1)
    gate = np.concatenate((np.repeat(hand_error.max(axis=1, keepdims=True), 6, axis=1),
                           np.repeat(hand_error, 7, axis=1),
                           np.abs(control.head_error) * 0.35), axis=1)
    rate = (reference[:, 3:] + 0.50 * residual[:, 3:] * (gate / 0.10).clip(0., 1.))
    angles = np.clip(q + LEAD_TIME * rate.clip(-1., 1.) * JOINT_RATE,
                     bounds[:, 0], bounds[:, 1])
    return base, angles


class LunaPolicy:
    """CPU inference from a versioned RSL-RL actor export without pickle or Torch."""

    def __init__(self, checkpoint: Path, model: LunaModel):
        with np.load(checkpoint, allow_pickle=False) as data:
            if (data["format"].item() != POLICY_VERSION
                    or data["urdf_sha256"].item() != model.fingerprint
                    or tuple(data["joints"].tolist()) != JOINTS):
                raise ValueError("Luna checkpoint format, URDF hash or joint order does not match")
            width = OBSERVATION_SIZE
            self.layers = []
            for i in range(3):
                weight, bias = data[f"weight_{i}"], data[f"bias_{i}"]
                if (weight.ndim != 2 or weight.shape[1] != width
                        or bias.shape != (weight.shape[0],)
                        or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                    raise ValueError(f"invalid Luna actor layer {i}")
                self.layers.append((weight.astype(np.float32), bias.astype(np.float32)))
                width = len(bias)
            if width != ACTION_SIZE:
                raise ValueError("incorrect Luna actor action width")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        for index, (weight, bias) in enumerate(self.layers):
            value = value @ weight.T + bias
            if index < len(self.layers) - 1:
                np.expm1(np.minimum(value, 0.), out=value, where=value < 0.)
        if not np.isfinite(value).all():
            raise ValueError("non-finite Luna actor prediction")
        return value
