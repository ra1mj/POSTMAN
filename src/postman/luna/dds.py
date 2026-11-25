"""POSTMAN client for the native gento_sdk_dds_ui_bridge executable."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import BASE_RATE, FX_TO_MODEL, GROUP_SLICES, GROUPS, JOINT_RATE

MOTION_AGE_NS = 200_000_000
CONTROLLER_AGE_NS = 1_600_000_000  # The SDK publishes ControllerState at 1 Hz.


def vector(value, count: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"invalid {label}: expected {count} finite values")
    return result


@dataclass(frozen=True)
class Limits:
    position: np.ndarray
    rate: np.ndarray
    base: np.ndarray

    @classmethod
    def from_gateway(cls, path: Path) -> Limits:
        config = json.loads(path.read_text())
        position, rate = [], []
        for group, axes in GROUPS.items():
            limits = np.asarray(config["joint_limits"][group], dtype=float)
            if (limits.shape != (len(axes), 2) or not np.isfinite(limits).all()
                    or np.any(limits[:, 0] >= limits[:, 1])):
                raise ValueError(f"invalid gateway limit range for {group}")
            position.extend(limits)
            trajectory = config.get("trajectory_limits", {}).get(group, {})
            cap = trajectory.get("velocity_rad_s", JOINT_RATE[GROUP_SLICES[group]])
            rate.extend(vector(cap, len(axes), f"{group} rate limits"))
        bounds = np.sort(np.array(position) * FX_TO_MODEL[:, None], axis=1)
        base = vector(config["base_velocity_limits"], 3, "base_velocity_limits")
        if np.any(base < 0) or np.any(np.array(rate) <= 0):
            raise ValueError("invalid gateway velocity limits")
        return cls(bounds, np.minimum(rate, JOINT_RATE), np.minimum(base, BASE_RATE))

    def check_position(self, q: np.ndarray) -> None:
        for group, section in GROUP_SLICES.items():
            for axis, (value, (low, high)) in enumerate(zip(q[section], self.position[section]), 1):
                if not low <= value <= high:
                    raise ValueError(f"{group} axis {axis} outside [{low:.6f}, {high:.6f}] rad "
                                     f"at {value:.6f} rad in model coordinates")


@dataclass(frozen=True)
class Feedback:
    stamp: int
    q: np.ndarray
    velocity: np.ndarray
    controller: dict
    base_pose: np.ndarray | None = None

    @classmethod
    def parse(cls, snapshot: dict, *, hardware: bool, command: bool) -> Feedback:
        now = time.time_ns()
        stamp = snapshot.get("motion_stamp_ns")
        if type(stamp) is not int or not 0 <= now - stamp <= MOTION_AGE_NS:
            raise ValueError("fresh native MotionState unavailable")
        if command and (snapshot.get("command_endpoints") is not True
                        or snapshot.get("command_deadline_supported") is not True):
            raise ValueError("native DDS command readers or bridge command deadlines unavailable")
        positions = []
        groups = snapshot.get("joints", {})
        for name, axes in GROUPS.items():
            positions.extend(vector(groups.get(name, {}).get("position"), len(axes), name))
        velocity = vector(snapshot.get("base"), 3, "RDK velocity")
        controller = snapshot.get("controller", {})
        if hardware:
            status_stamp = controller.get("stamp_ns")
            fx_age = controller.get("frame_age_ms")
            if (controller.get("model") != 3 or controller.get("frame_fresh") is not True
                    or type(status_stamp) is not int
                    or not 0 <= now - status_stamp <= CONTROLLER_AGE_NS
                    or type(fx_age) is not int or not 0 <= fx_age <= 500
                    or type(controller.get("frame_serial")) is not int):
                raise ValueError("fresh Luna model 3 FX controller status unavailable")
            for name, axes in GROUPS.items():
                actual = controller.get("groups", {}).get(name, {})
                if actual.get("dof") != len(axes) or actual.get("fault_code") != 0:
                    raise ValueError(f"missing axis status or active FX fault for {name}")
            if command and not controller.get("capabilities", 0) & 8:
                raise ValueError("FX controller lacks a software group stop")
        base_pose = None
        odometry = snapshot.get("odometry")
        if isinstance(odometry, dict):
            pose = [odometry.get(axis) for axis in ("x", "y", "yaw")]
            if all(type(value) in (int, float) and np.isfinite(value) for value in pose):
                base_pose = np.array(pose, dtype=float)
        return cls(stamp, np.array(positions) * FX_TO_MODEL, velocity, controller, base_pose)


class DdsBridge:
    def __init__(self, executable: Path, scope: str, domain: int):
        self.process = subprocess.Popen(
            [str(executable.resolve()), "--scope", scope, "--no-camera",
             "--feedback-timeout-ms", "180", "--snapshot-period-ms", "10"],
            env={**os.environ, "GENTO_DDS_DOMAIN": str(domain)},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )
        self.condition = threading.Condition()
        self.snapshot: dict = {}
        self.responses: dict[int, dict] = {}
        self.pending: set[int] = set()
        self.sequence = 0
        self.failure: str | None = None
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                event = json.loads(line)
                with self.condition:
                    if event.get("event") == "snapshot":
                        self.snapshot = event
                    elif event.get("event") == "result" and event.get("id") in self.pending:
                        self.responses[event["id"]] = event
                    self.condition.notify_all()
        except (OSError, ValueError, AttributeError) as error:
            failure = f"native SDK bridge output failed: {error}"
        else:
            failure = "native SDK bridge output closed"
        with self.condition:
            self.failure = failure
            self.condition.notify_all()

    def feedback(self, *, hardware: bool, command: bool, wait: float = 0.,
                 after: int = 0) -> Feedback:
        end = time.monotonic() + wait
        with self.condition:
            while True:
                if self.failure:
                    raise RuntimeError(self.failure)
                try:
                    feedback = Feedback.parse(self.snapshot, hardware=hardware, command=command)
                    if feedback.stamp <= after:
                        raise ValueError("native MotionState is not advancing")
                    return feedback
                except ValueError:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise
                self.condition.wait(remaining)

    def send(self, commands: list[dict], *, timeout: float = 0.14,
             expires_ns: int | None = None) -> None:
        if not commands:
            return
        if expires_ns is None:
            expires_ns = time.time_ns() + 120_000_000
        with self.condition:
            if self.failure:
                raise RuntimeError(self.failure)
            requests = {}
            for command in commands:
                self.sequence += 1
                requests[self.sequence] = {**command, "id": self.sequence, "expires_ns": expires_ns}
            self.pending.update(requests)
        try:
            self.process.stdin.write("".join(json.dumps(row, allow_nan=False) + "\n"
                                             for row in requests.values()))
            self.process.stdin.flush()
            with self.condition:
                ready = self.condition.wait_for(
                    lambda: self.failure or all(i in self.responses for i in requests), timeout,
                )
                if self.failure:
                    raise RuntimeError(self.failure)
                if not ready:
                    operations = ", ".join(requests[i]["op"] for i in requests
                                           if i not in self.responses)
                    raise TimeoutError(f"native DDS acknowledgement timeout for {operations}")
                for i, request in requests.items():
                    response = self.responses[i]
                    if response.get("ok") is not True:
                        tag = f"{request['op']} {request.get('group', '')}".strip()
                        error, native = response.get("error"), response.get("native_code")
                        raise RuntimeError(f"DDS {tag} rejected with SDK {error}, native {native}")
        finally:
            with self.condition:
                self.pending.difference_update(requests)
                for i in requests:
                    self.responses.pop(i, None)

    def close(self) -> None:
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdout.close()
