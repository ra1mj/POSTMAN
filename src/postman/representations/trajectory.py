"""Local task-space trajectory targets shared by simulated and learned controls.

The legacy 24-value UMMR layout is preserved. In this trajectory format the
historically named ``base_twist`` slot stores a finite local (x, y, yaw)
displacement, not a velocity. Torso, wrists, elbows and grippers store offsets
from a captured segment start. Wrist rotation offsets are rotations in the
local frame, composed on the left of their start orientations. ``resolve``
recovers targets in that frame; ``world_from_local`` retains its world anchor.
The interface keeps task-space targets stable while the controller handles
embodiment-specific joint coordination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ummr import AxisAnglePose, WholeBodyCommand

SYNTHETIC_MODES = ("stand", "reach", "side", "height", "base", "bimanual")

# Numerical target envelopes in metres/radians and normalized gripper opening.
# They bound generated requests; they do not establish robot IK feasibility.
COMMAND_OFFSET_LIMITS = np.array(
    [0.4, 0.3, 0.6, 0.15]
    + [0.3, 0.25, 0.25, 0.5, 0.5, 0.5] * 2
    + [0.2, 0.2, 0.2] * 2
    + [0.5, 0.5],
    dtype=np.float64,
)
COMMAND_OFFSET_LIMITS.setflags(write=False)


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _axis_angle_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation))
    if angle < 1e-12:
        quaternion = np.concatenate(([1.0], 0.5 * rotation))
    else:
        quaternion = np.concatenate(
            ([np.cos(angle / 2.0)], rotation * (np.sin(angle / 2.0) / angle))
        )
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    sin_half = float(np.linalg.norm(quaternion[1:]))
    if sin_half < 1e-12:
        return 2.0 * quaternion[1:]
    angle = 2.0 * np.arctan2(sin_half, quaternion[0])
    return quaternion[1:] * (angle / sin_half)


def _compose_rotations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = _axis_angle_to_quaternion(left), _axis_angle_to_quaternion(right)
    quaternion = np.concatenate(
        (
            [a[0] * b[0] - np.dot(a[1:], b[1:])],
            a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]),
        )
    )
    return _quaternion_to_axis_angle(quaternion)


def _interpolate_rotation(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    left, right = _axis_angle_to_quaternion(start), _axis_angle_to_quaternion(end)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quaternion = left + alpha * (right - left)
    else:
        angle = np.arccos(dot)
        quaternion = (
            np.sin((1.0 - alpha) * angle) * left + np.sin(alpha * angle) * right
        ) / np.sin(angle)
    return _quaternion_to_axis_angle(quaternion)


@dataclass(frozen=True, init=False)
class TrajectoryAnchor:
    """Frame and measured start targets needed to interpret relative commands.

    ``start.base_twist`` stores the start footprint pose (x, y, yaw) in the
    named local frame. ``world_from_local`` is that frame's world pose.
    The wrist/elbow positions in ``start`` use the same local frame.
    """

    frame_id: str = "segment_start"
    world_from_local: AxisAnglePose = field(default_factory=AxisAnglePose)
    start: WholeBodyCommand = field(default_factory=WholeBodyCommand)

    def __init__(
        self,
        frame_id: str = "segment_start",
        world_from_local: AxisAnglePose | None = None,
        start: WholeBodyCommand | None = None,
        *,
        start_command: WholeBodyCommand | None = None,
    ) -> None:
        if start is not None and start_command is not None:
            raise ValueError("provide only one of start and start_command")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "world_from_local", world_from_local or AxisAnglePose())
        object.__setattr__(self, "start", start_command or start or WholeBodyCommand())
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        if not isinstance(self.world_from_local, AxisAnglePose):
            raise TypeError("world_from_local must be an AxisAnglePose")
        if not isinstance(self.start, WholeBodyCommand):
            raise TypeError("start must be a WholeBodyCommand")
        world_from_local = AxisAnglePose(
            self.world_from_local.position, self.world_from_local.axis_angle
        )
        object.__setattr__(self, "world_from_local", world_from_local)
        # Capture the anchor independently of later mutations to robot buffers.
        copied = WholeBodyCommand.from_vector(self.start.as_vector())
        copied.weights = dict(self.start.weights)
        object.__setattr__(self, "start", copied)

    @property
    def start_command(self) -> WholeBodyCommand:
        """Compatibility spelling for callers that prefer explicit semantics."""
        return self.start

    @property
    def local_frame(self) -> str:
        return self.frame_id


@dataclass(frozen=True)
class WholeBodyTrajectory:
    """Short-horizon relative commands, sampled with linear/SO(3) interpolation."""

    timestamps: np.ndarray
    commands: np.ndarray
    anchor: TrajectoryAnchor = field(default_factory=TrajectoryAnchor)
    controller_id: str = "postman_rl_wbc"
    mode: str = "recorded"

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if isinstance(self.commands, (list, tuple)) and self.commands and all(
            isinstance(command, WholeBodyCommand) for command in self.commands
        ):
            commands = np.stack([command.as_vector() for command in self.commands])
        else:
            commands = np.asarray(self.commands, dtype=np.float64)
        if timestamps.ndim != 1 or timestamps.size < 2:
            raise ValueError("timestamps must be a vector containing at least two samples")
        if commands.shape != (timestamps.size, 24):
            raise ValueError("commands must have shape (number of timestamps, 24)")
        if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(commands)):
            raise ValueError("trajectory timestamps and commands must be finite")
        if timestamps[0] < 0.0 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timestamps must be non-negative and strictly increasing")
        if not isinstance(self.anchor, TrajectoryAnchor):
            raise TypeError("anchor must be a TrajectoryAnchor")
        if not isinstance(self.controller_id, str) or not self.controller_id.strip():
            raise ValueError("controller_id must be a non-empty string")
        timestamps, commands = timestamps.copy(), commands.copy()
        timestamps.setflags(write=False)
        commands.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "commands", commands)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    def sample(self, time_s: float) -> WholeBodyCommand:
        """Return relative targets; hold the first/last knot outside the interval."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if time_s <= self.timestamps[0]:
            return WholeBodyCommand.from_vector(self.commands[0])
        if time_s >= self.timestamps[-1]:
            return WholeBodyCommand.from_vector(self.commands[-1])
        index = int(np.searchsorted(self.timestamps, time_s, side="right")) - 1
        interval = self.timestamps[index + 1] - self.timestamps[index]
        alpha = float((time_s - self.timestamps[index]) / interval)
        start, end = self.commands[index], self.commands[index + 1]
        target = (1.0 - alpha) * start + alpha * end
        target[2] = _wrap_angle(start[2] + alpha * _wrap_angle(end[2] - start[2]))
        for rotation_slice in (slice(7, 10), slice(13, 16)):
            target[rotation_slice] = _interpolate_rotation(
                start[rotation_slice], end[rotation_slice], alpha
            )
        return WholeBodyCommand.from_vector(target)

    def interpolate(self, time_s: float) -> WholeBodyCommand:
        """Alias for :meth:`sample`, retained for trajectory consumers."""
        return self.sample(time_s)

    def sample_window(
        self, time_s: float, lookahead_s: float = 0.1
    ) -> tuple[WholeBodyCommand, WholeBodyCommand]:
        """Expose current/next references to the controller observation builder."""
        if not np.isfinite(lookahead_s) or lookahead_s <= 0.0:
            raise ValueError("lookahead_s must be finite and positive")
        return self.sample(time_s), self.sample(time_s + lookahead_s)

    def current_next(
        self, time_s: float, lookahead_s: float = 0.1
    ) -> tuple[WholeBodyCommand, WholeBodyCommand]:
        """Alias used by observation builders that expose current/next targets."""
        return self.sample_window(time_s, lookahead_s)

    def resolve(self, time_s: float) -> WholeBodyCommand:
        """Compose offsets with the captured start, returning local-frame poses.

        Translational offsets are expressed along the named frame axes, not
        the moving body axes. The returned base slot is a footprint pose.
        """
        start, relative = self.anchor.start, self.sample(time_s)
        target = WholeBodyCommand.from_vector(start.as_vector() + relative.as_vector())
        target.base_twist[2] = _wrap_angle(start.base_twist[2] + relative.base_twist[2])
        for side in ("left", "right"):
            name = f"{side}_wrist"
            original, offset = getattr(start, name), getattr(relative, name)
            setattr(
                target,
                name,
                AxisAnglePose(
                    original.position + offset.position,
                    _compose_rotations(offset.axis_angle, original.axis_angle),
                ),
            )
        target.weights = dict(start.weights)
        return target


def clip_command_offset(value: np.ndarray) -> np.ndarray:
    """Apply a numerical envelope without labelling a target reachable or safe."""
    vector = WholeBodyCommand.from_vector(value).as_vector()
    vector = np.clip(vector, -COMMAND_OFFSET_LIMITS, COMMAND_OFFSET_LIMITS)
    for rotation_slice in (slice(7, 10), slice(13, 16)):
        magnitude = float(np.linalg.norm(vector[rotation_slice]))
        if magnitude > 0.75:
            vector[rotation_slice] *= 0.75 / magnitude
    return vector


def _coerce_rng(rng: np.random.Generator | int) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError("rng must be a numpy.random.Generator or integer seed")


def sample_synthetic_offset(
    rng: np.random.Generator | int, mode: str = "mixed", scale: float = 1.0
) -> np.ndarray:
    """Sample deterministic, coordinated 24-value offsets from a supplied RNG or seed.

    ``scale`` supports an amplitude curriculum. Values above one are accepted
    but still clipped to the numerical envelope. No kinematic solver is used.
    """
    rng = _coerce_rng(rng)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and non-negative")
    if mode == "mixed":
        mode = str(rng.choice(SYNTHETIC_MODES))
    if mode not in SYNTHETIC_MODES:
        raise ValueError(f"unknown synthetic mode {mode!r}; expected {SYNTHETIC_MODES}")
    vector = np.zeros(24, dtype=np.float64)
    if mode == "stand":
        return vector
    if mode == "reach":
        shift = rng.uniform([0.06, -0.04, -0.06], [0.22, 0.04, 0.10])
        vector[0] = 0.2 * shift[0]
        vector[3] = 0.3 * shift[2]
        vector[4:7] = shift
        vector[10:13] = shift + rng.uniform(-0.02, 0.02, 3)
    elif mode == "side":
        shift = np.array([0.0, rng.uniform(0.06, 0.20) * rng.choice([-1, 1]), 0.0])
        vector[1] = 0.35 * shift[1]
        vector[4:7] = vector[10:13] = shift
    elif mode == "height":
        height = rng.uniform(0.08, 0.24) * rng.choice([-1, 1])
        vector[3] = 0.5 * height
        vector[6] = vector[12] = height
    elif mode == "base":
        vector[:3] = rng.uniform([-0.25, -0.15, -0.35], [0.25, 0.15, 0.35])
        # Fixed hand targets make the arms compensate for footprint movement.
    elif mode == "bimanual":
        shift = rng.uniform([-0.1, -0.12, -0.12], [0.18, 0.12, 0.18])
        rotation = rng.uniform(-0.18, 0.18, 3)
        vector[4:7] = vector[10:13] = shift
        vector[7:10] = vector[13:16] = rotation
        vector[3] = 0.3 * shift[2]
        vector[22:] = rng.uniform(-0.25, 0.25)
    vector[16:19] = 0.5 * vector[4:7]
    vector[19:22] = 0.5 * vector[10:13]
    return clip_command_offset(vector * scale)


def make_synthetic_trajectory(
    rng: np.random.Generator | int,
    mode: str = "mixed",
    duration: float = 2.0,
    anchor: TrajectoryAnchor | None = None,
    scale: float = 1.0,
) -> WholeBodyTrajectory:
    """Create a zero-to-target segment for reaching or whole-body coordination."""
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    rng = _coerce_rng(rng)
    selected_mode = str(rng.choice(SYNTHETIC_MODES)) if mode == "mixed" else mode
    endpoint = sample_synthetic_offset(rng, selected_mode, scale)
    return WholeBodyTrajectory(
        timestamps=np.array([0.0, duration]),
        commands=np.stack([np.zeros(24), endpoint]),
        anchor=anchor or TrajectoryAnchor(),
        mode=selected_mode,
    )


# Descriptive aliases used by command/trajectory managers.
sample_relative_offset = sample_synthetic_offset
Trajectory = WholeBodyTrajectory
