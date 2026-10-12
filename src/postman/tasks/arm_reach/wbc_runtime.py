"""Native mjlab command, action, observation and reward terms for RL-WBC.

The policy sees a compact task-space command and emits a mixed actuator target:
four wheel velocities followed by twenty-one joint positions.  All stateful
buffers are torch tensors owned by mjlab, so this module remains batched and
device resident in the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from postman.robots import TIAGO_PP
from postman.wbc.action_filter import BatchedActionFilter

COMMAND_DIM = 24
ACTION_DIM = 25
WHEEL_DIM = 4


def _all_env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long)


def _sample_urr(count: int, device: str, scale: float = 1.0) -> torch.Tensor:
    """Sample bounded synthetic local-frame UMMR offsets.

    The six target families are deliberately sampled together so the policy
    learns base/torso/arms coordination instead of independent joint tracks.
    ``base_twist`` is a finite local x/y/yaw displacement for this segment.
    """
    command = torch.zeros((count, COMMAND_DIM), device=device)
    modes = torch.randint(0, 6, (count,), device=device)
    command[:, 0:2] = torch.empty((count, 2), device=device).uniform_(-0.25, 0.25)
    command[:, 2] = torch.empty(count, device=device).uniform_(-0.45, 0.45)
    command[:, 3] = torch.empty(count, device=device).uniform_(-0.12, 0.12)
    # Both wrists and elbows are local Cartesian offsets.  The mirrored signs
    # provide a useful bimanual prior while preserving per-env variation.
    command[:, 4:7] = torch.empty((count, 3), device=device).uniform_(-0.18, 0.18)
    command[:, 10:13] = torch.empty((count, 3), device=device).uniform_(-0.18, 0.18)
    command[:, 16:19] = torch.empty((count, 3), device=device).uniform_(-0.12, 0.12)
    command[:, 19:22] = torch.empty((count, 3), device=device).uniform_(-0.12, 0.12)
    command[:, 22:24] = torch.empty((count, 2), device=device).uniform_(0.005, 0.04)

    # Mode gates keep the distribution structured: stand, reach, side, height,
    # base, and bimanual.  The zeroed components remain explicit targets.
    gate = torch.zeros((count, 6), device=device)
    gate.scatter_(1, modes[:, None], 1.0)
    command[:, 0:3] *= (gate[:, 4:5] + gate[:, 2:3] * 0.3 + gate[:, 5:6] * 0.4)
    command[:, 3:4] *= gate[:, 3:4] + gate[:, 1:2] * 0.25
    command[:, 4:22] *= gate[:, 1:2] + gate[:, 2:3] * 0.8 + gate[:, 3:4] * 0.8 + gate[:, 5:6]
    command[:, 22:24] *= gate[:, 1:2] + gate[:, 5:6]
    return command * float(scale)


@dataclass(kw_only=True)
class SyntheticUrrCommandCfg(CommandTermCfg):
    """Synthetic local-frame command trajectory configuration."""

    scale: float = 1.0
    command_name: str = "urr"

    def build(self, env):
        return SyntheticUrrCommand(self, env)


class SyntheticUrrCommand(CommandTerm):
    """A bounded two-waypoint command trajectory at the policy frequency."""

    def __init__(self, cfg: SyntheticUrrCommandCfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, COMMAND_DIM), device=self.device)
        self._next = torch.zeros_like(self._command)
        self.anchor = torch.zeros((self.num_envs, 6, 3), device=self.device)
        self.anchor_q = torch.zeros((self.num_envs, 25), device=self.device)
        self._duration = float(cfg.resampling_time_range[1])

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def next_command(self) -> torch.Tensor:
        return self._next

    @property
    def command_delta(self) -> torch.Tensor:
        return self._next - self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        robot = self._env.scene["robot"]
        body_names = (
            "base_link",
            "torso_lift_link",
            "arm_left_4_link",
            "arm_left_7_link",
            "arm_right_4_link",
            "arm_right_7_link",
        )
        body_ids = torch.tensor(
            [robot.body_names.index(name) for name in body_names],
            device=self.device,
            dtype=torch.long,
        )
        pose = robot.data.body_link_pose_w[env_ids][:, body_ids]
        self.anchor[env_ids] = pose[..., :3]
        q = robot.data.joint_pos[env_ids]
        self.anchor_q[env_ids] = q
        self._command[env_ids] = 0.0
        self._next[env_ids] = _sample_urr(len(env_ids), self.device, self.cfg.scale)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
        # CommandTerm updates time_left before this callback.  A cubic smoothstep
        # avoids discontinuous target velocity at a waypoint.
        phase = 1.0 - torch.clamp(self.time_left / self._duration, 0.0, 1.0)
        smooth = phase * phase * (3.0 - 2.0 * phase)
        self._command = self._next * smooth[:, None]

    def _update_metrics(self) -> None:
        self.metrics["command_magnitude"] = torch.linalg.vector_norm(self._command, dim=-1)


def urr_command(env, command_name: str = "urr") -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def urr_command_delta(env, command_name: str = "urr") -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return term.command_delta


def privileged_command_context(env, command_name: str = "urr") -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    # The teacher context is intentionally deployment independent: current q,
    # command waypoints, the bounded QP reference, and randomized dynamics.
    q = robot.data.joint_pos
    teacher = env.action_manager.get_term("whole_body").teacher_targets
    from postman.tasks.arm_reach.mdp.randomization import dynamics_observation

    factors = dynamics_observation(env)
    return torch.cat((q, term.command, term.command_delta, teacher, factors), dim=-1)


def actor_last_action(env, action_name: str = "whole_body") -> torch.Tensor:
    return env.action_manager.get_term(action_name).raw_action


@dataclass(kw_only=True)
class RlWbcActionCfg(ActionTermCfg):
    """Configuration for the canonical 25-dimensional RL-WBC action."""

    dt: float = 0.01
    alpha: float = 0.25
    max_delay: int = 2

    def build(self, env):
        return RlWbcAction(self, env)


class RlWbcAction(ActionTerm):
    """Apply a safe mixed wheel-velocity/joint-position policy target."""

    def __init__(self, cfg: RlWbcActionCfg, env):
        super().__init__(cfg, env)
        if cfg.dt <= 0 or not 0 < cfg.alpha <= 1 or cfg.max_delay != 2:
            raise ValueError("RL-WBC requires dt>0, alpha in (0,1], and max_delay=2")
        robot = env.scene[cfg.entity_name]
        names = tuple(robot.joint_names)
        try:
            ids = [names.index(name) for name in TIAGO_PP.joint_names]
        except ValueError as exc:
            raise ValueError("TIAGo++ canonical joints are missing from the entity") from exc
        self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        self._wheel_ids = self._joint_ids[:WHEEL_DIM]
        self._position_ids = self._joint_ids[WHEEL_DIM:]
        self._raw_actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self._applied_targets = torch.zeros_like(self._raw_actions)

        # The model's non-wheel joint ranges are finite; the wheels have no
        # positional range and are excluded from these 21-element bounds.
        limits = robot.data.joint_pos_limits[:, self._position_ids]
        lower = limits[..., 0]
        upper = limits[..., 1]
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("TIAGo++ position joints must have finite limits")
        lower = lower[0]
        upper = upper[0]
        velocity_limits = torch.full((21,), 4.0, device=self.device)
        velocity_limits[0] = 1.0
        velocity_limits[1:3] = 2.0
        velocity_limits[[10, 11, 19, 20]] = 0.3
        self._filter = BatchedActionFilter(
            self.num_envs,
            lower,
            upper,
            velocity_limits,
            dt=cfg.dt,
            alpha=cfg.alpha,
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def applied_targets(self) -> torch.Tensor:
        return self._applied_targets

    @property
    def teacher_targets(self) -> torch.Tensor:
        return self._teacher_targets()

    @property
    def diagnostics(self) -> dict[str, torch.Tensor]:
        return self._filter.diagnostics

    def _teacher_targets(self) -> torch.Tensor:
        robot = self._env.scene[self.cfg.entity_name]
        q = robot.data.joint_pos[:, self._joint_ids]
        target = torch.cat((torch.zeros_like(q[:, :4]), q[:, 4:]), dim=-1)
        # A conservative QP-compatible teacher: hold the articulated posture
        # while projecting the requested planar twist to skid-steer wheel rates.
        command = self._env.command_manager.get_command("urr")
        radius = 0.0762
        half_track = 0.223175
        vx = command[:, 0]
        yaw = command[:, 2]
        target[:, 0] = (vx / radius) + (half_track * yaw / radius)
        target[:, 2] = (vx / radius) + (half_track * yaw / radius)
        target[:, 1] = (vx / radius) - (half_track * yaw / radius)
        target[:, 3] = (vx / radius) - (half_track * yaw / radius)
        target[:, 4] = torch.clamp(
            q[:, 4] + command[:, 3], self._filter.lower[0], self._filter.upper[0]
        )
        return target

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        robot = self._env.scene[self.cfg.entity_name]
        measured = robot.data.joint_pos[:, self._joint_ids]
        teacher = self._teacher_targets()
        self._applied_targets = self._filter.step(actions, measured, teacher)

    def apply_actions(self) -> None:
        robot = self._env.scene[self.cfg.entity_name]
        robot.set_joint_velocity_target(self._applied_targets[:, :4], joint_ids=self._wheel_ids)
        robot.set_joint_position_target(self._applied_targets[:, 4:], joint_ids=self._position_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = _all_env_ids(self._env, env_ids)
        robot = self._env.scene[self.cfg.entity_name]
        measured = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        measured[:, 4:] = robot.data.joint_pos[:, self._joint_ids[4:]]
        delays = torch.randint(0, 3, (ids.numel(),), device=self.device)
        self._filter.reset(ids, measured[ids], delays=delays)
        dr_state = getattr(self._env, "_postman_dr", None)
        if dr_state is not None and "delay" in dr_state:
            dr_state["delay"][ids, 0] = delays.to(dtype=dr_state["delay"].dtype)
        self._raw_actions[ids] = 0.0
        self._applied_targets[ids] = measured[ids]


def command_tracking_reward(env, command_name: str = "urr") -> torch.Tensor:
    """Task-space reward for wrist/elbow/base/torso coordination."""
    term: SyntheticUrrCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    names = (
        "base_link",
        "torso_lift_link",
        "arm_left_4_link",
        "arm_left_7_link",
        "arm_right_4_link",
        "arm_right_7_link",
    )
    ids = torch.tensor([robot.body_names.index(name) for name in names], device=env.device)
    positions = robot.data.body_link_pose_w[:, ids, :3]
    target = term.anchor + torch.stack(
        (
            torch.cat((term.command[:, 0:2], torch.zeros_like(term.command[:, 0:1])), dim=-1),
            torch.cat((torch.zeros_like(term.command[:, 0:2]), term.command[:, 3:4]), dim=-1),
            term.command[:, 16:19],
            term.command[:, 4:7],
            term.command[:, 19:22],
            term.command[:, 10:13],
        ),
        dim=1,
    )
    errors = positions - target
    weighted = torch.cat(
        (
            errors[:, 0].square().sum(dim=-1, keepdim=True),
            errors[:, 1].square().sum(dim=-1, keepdim=True),
            1.5 * errors[:, 2].square().sum(dim=-1, keepdim=True),
            4.0 * errors[:, 3].square().sum(dim=-1, keepdim=True),
            1.5 * errors[:, 4].square().sum(dim=-1, keepdim=True),
            4.0 * errors[:, 5].square().sum(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    return torch.exp(-weighted.sum(dim=-1))


def action_smoothness_reward(env, action_name: str = "whole_body") -> torch.Tensor:
    term: RlWbcAction = env.action_manager.get_term(action_name)
    delta = term.raw_action - env.action_manager.prev_action
    return -delta.square().mean(dim=-1)


def joint_limit_penalty(env) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.joint_pos
    limits = robot.data.joint_pos_limits
    distance = torch.minimum(q - limits[..., 0], limits[..., 1] - q)
    return -torch.relu(0.05 - distance[:, 4:]).square().mean(dim=-1)


def no_oscillation_penalty(env, action_name: str = "whole_body") -> torch.Tensor:
    term: RlWbcAction = env.action_manager.get_term(action_name)
    delta = term.raw_action - env.action_manager.prev_action
    return -delta[:, 4:].diff(dim=-1).square().mean(dim=-1)


__all__ = [
    "RlWbcAction",
    "RlWbcActionCfg",
    "SyntheticUrrCommand",
    "SyntheticUrrCommandCfg",
    "action_smoothness_reward",
    "actor_last_action",
    "command_tracking_reward",
    "joint_limit_penalty",
    "no_oscillation_penalty",
    "privileged_command_context",
    "urr_command",
    "urr_command_delta",
]
