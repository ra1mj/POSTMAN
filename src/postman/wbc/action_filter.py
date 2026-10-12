"""Batched safety filter for canonical wheel/position policy actions.

The canonical action has four wheel velocities followed by 21 joint positions.
All state and filtering operations stay on the configured torch device.
"""

from __future__ import annotations

import torch


class BatchedActionFilter:
    """Map normalized actions to physical targets with delay and rate protection.

    ``lower``, ``upper`` and ``velocity_limits`` describe the 21 position
    joints. Canonical 25-element vectors are also accepted; their first four
    wheel entries are ignored. The wheel action range is fixed at ±5 rad/s.
    """

    action_dim = 25
    wheel_dim = 4
    max_delay = 2

    def __init__(
        self,
        num_envs: int,
        lower,
        upper,
        velocity_limits,
        dt: float = 0.01,
        max_wheel_acceleration: float = 10.0,
        alpha: float = 0.25,
        device=None,
    ) -> None:
        if num_envs < 1 or dt <= 0 or max_wheel_acceleration < 0 or not 0 < alpha <= 1:
            raise ValueError("invalid filter size, timestep, acceleration, or alpha")
        self.num_envs = int(num_envs)
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.max_wheel_acceleration = float(max_wheel_acceleration)

        self.lower = self._position_limits(lower, "lower")
        self.upper = self._position_limits(upper, "upper")
        self.velocity_limits = self._position_limits(velocity_limits, "velocity_limits")
        if not torch.all(torch.isfinite(self.lower)) or not torch.all(torch.isfinite(self.upper)):
            raise ValueError("position bounds must be finite")
        if not torch.all(torch.isfinite(self.velocity_limits)):
            raise ValueError("position velocity limits must be finite")
        if torch.any(self.lower >= self.upper) or torch.any(self.velocity_limits < 0):
            raise ValueError("position bounds or velocity limits are invalid")

        self.midpoint = (self.lower + self.upper) * 0.5
        self.half_span = (self.upper - self.lower) * 0.5
        self._previous = torch.cat(
            (torch.zeros(self.num_envs, 4, device=self.device),
             self.midpoint.expand(self.num_envs, -1).clone()), dim=1
        )
        self._queue = self._previous.unsqueeze(0).expand(3, -1, -1).clone()
        self._delays = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._write_index = 0
        self.fallback_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.diagnostics: dict[str, torch.Tensor] = {}

    def _position_limits(self, values, name: str) -> torch.Tensor:
        result = torch.as_tensor(values, dtype=self.dtype, device=self.device).flatten()
        if result.numel() == self.action_dim:
            result = result[self.wheel_dim:]
        if result.numel() != self.action_dim - self.wheel_dim:
            raise ValueError(f"{name} must have 21 or 25 elements")
        return result.clone()

    def _batch(self, values, count: int, name: str) -> torch.Tensor:
        result = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if result.shape != (count, self.action_dim):
            raise ValueError(f"{name} must have shape ({count}, 25)")
        return result

    def reset(self, env_ids, measured, delays=None) -> None:
        """Reset selected environments to measured positions and safe wheel zero.

        ``measured`` may contain all environments or only ``env_ids``. Delays
        are integer cycles in [0, 2], supplied per selected environment or as
        a scalar. Other environments' queue and applied action are untouched.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        if torch.any((ids < 0) | (ids >= self.num_envs)) or (
            torch.unique(ids).numel() != ids.numel()
        ):
            raise ValueError("env_ids must be unique valid environment indices")
        measured_t = torch.as_tensor(measured, dtype=self.dtype, device=self.device)
        if measured_t.shape == (self.num_envs, self.action_dim):
            selected = measured_t[ids]
        elif measured_t.shape == (ids.numel(), self.action_dim):
            selected = measured_t
        else:
            raise ValueError("measured must be [num_envs,25] or [len(env_ids),25]")
        positions = selected[:, self.wheel_dim:]
        if not torch.all(torch.isfinite(positions)):
            raise ValueError("measured position targets must be finite")
        initial = torch.cat(
            (torch.zeros(ids.numel(), self.wheel_dim, device=self.device),
             torch.clamp(positions, self.lower, self.upper)), dim=1
        )
        self._previous[ids] = initial
        self._queue[:, ids] = initial.unsqueeze(0)
        self.fallback_mask[ids] = False
        if delays is not None:
            delay_t = torch.as_tensor(delays, device=self.device)
            if delay_t.ndim == 0:
                delay_t = delay_t.expand(ids.numel())
            elif delay_t.shape == (self.num_envs,):
                delay_t = delay_t[ids]
            elif delay_t.shape != (ids.numel(),):
                raise ValueError("delays must be scalar or one value per environment")
            if torch.any(delay_t != delay_t.to(torch.long)) or torch.any(
                (delay_t < 0) | (delay_t > 2)
            ):
                raise ValueError("delays must be integer cycles in [0, 2]")
            self._delays[ids] = delay_t.to(torch.long)

    def step(self, normalized_action, measured, teacher_targets=None) -> torch.Tensor:
        """Return safe physical targets for all environments.

        Invalid policy rows (non-finite or magnitude > 1e6) use finite,
        in-bounds physical teacher targets when available. Otherwise they hold
        measured positions and command zero wheel speed. Fallback skips the
        delay queue and low-pass filter while still obeying rate limits.
        """
        raw = self._batch(normalized_action, self.num_envs, "normalized_action")
        measured_t = self._batch(measured, self.num_envs, "measured")
        positions = measured_t[:, self.wheel_dim:]
        measured_ok = torch.isfinite(positions).all(dim=1)
        measured_safe = torch.where(torch.isfinite(positions), positions, self._previous[:, 4:])
        measured_safe = torch.clamp(measured_safe, self.lower, self.upper)

        invalid = (~torch.isfinite(raw) | (torch.abs(raw) > 1e6)).any(dim=1)
        raw_safe = torch.where(torch.isfinite(raw) & (torch.abs(raw) <= 1e6), raw, 0.0)
        normalized = torch.clamp(raw_safe, -1.0, 1.0)
        saturation = (raw_safe != normalized).any(dim=1) & ~invalid
        physical = torch.cat(
            (normalized[:, :4] * 5.0,
             self.midpoint + normalized[:, 4:] * self.half_span), dim=1
        )

        teacher_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        teacher_safe = torch.zeros_like(physical)
        if teacher_targets is not None:
            teacher = self._batch(teacher_targets, self.num_envs, "teacher_targets")
            teacher_ok = torch.isfinite(teacher).all(dim=1)
            teacher_ok &= (torch.abs(teacher[:, :4]) <= 5.0).all(dim=1)
            teacher_ok &= (
                (teacher[:, 4:] >= self.lower) & (teacher[:, 4:] <= self.upper)
            ).all(dim=1)
            teacher_safe = torch.where(torch.isfinite(teacher), teacher, 0.0)

        hold = torch.cat((torch.zeros_like(physical[:, :4]), measured_safe), dim=1)
        # Sanitize before writing history. A fallback flushes every delayed
        # command for that environment so stale policy output cannot resurface.
        self._queue = torch.where(invalid[None, :, None], hold[None, :, :], self._queue)
        queued = torch.where(invalid[:, None], hold, physical)
        self._queue[self._write_index] = queued
        delayed_index = torch.remainder(self._write_index - self._delays, 3)
        rows = torch.arange(self.num_envs, device=self.device)
        delayed = self._queue[delayed_index, rows]
        fallback = torch.where(teacher_ok[:, None], teacher_safe, hold)
        desired = torch.where(
            invalid[:, None], fallback,
            self._previous + self.alpha * (delayed - self._previous),
        )
        max_delta = torch.cat(
            (torch.full((4,), self.max_wheel_acceleration * self.dt, device=self.device),
             self.velocity_limits * self.dt)
        )
        delta = desired - self._previous
        limited_delta = torch.clamp(delta, -max_delta, max_delta)
        rate_limited = (delta != limited_delta).any(dim=1)
        applied = self._previous + limited_delta
        applied = torch.cat(
            (torch.clamp(applied[:, :4], -5.0, 5.0),
             torch.clamp(applied[:, 4:], self.lower, self.upper)), dim=1
        )
        # No safe teacher means an urgent measured-position hold and wheel stop.
        # This is deliberately immediate, even if it exceeds a command slew
        # limit; the next policy step resumes from this safe applied target.
        hold_used = invalid & ~teacher_ok
        applied = torch.where(hold_used[:, None], hold, applied)
        self._previous = applied
        self._write_index = (self._write_index + 1) % 3
        self.fallback_mask = invalid
        self.diagnostics = {
            "fallback": invalid,
            "teacher_used": invalid & teacher_ok,
            "hold_used": hold_used,
            "saturation": saturation,
            "policy_saturated": saturation,
            "rate_limited": rate_limited,
            "measured_invalid": ~measured_ok,
            "delay_cycles": self._delays.clone(),
        }
        return applied

    __call__ = step
