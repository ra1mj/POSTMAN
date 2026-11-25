"""Run the Luna 680 policy through the native FX/RDK DDS SDK."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from .control import (
    LEAD_TIME,
    LunaPolicy,
    anchor_target,
    decode,
    default_policy,
    offset_target,
    tracking,
)
from .dds import DdsBridge, Feedback, Limits
from .model import (
    BASE_RATE,
    DT,
    FX_TO_MODEL,
    GROUP_SLICES,
    GROUPS,
    HOME,
    JOINT_RATE,
    JOINTS,
    CollisionCheck,
    LunaModel,
    default_urdf,
)
from .web import Jog, Operator

STOPPED_VELOCITY = 0.006
BASE_ACCEL = np.array([0.5, 0.5, 0.8])


def sdk_sim_model(model: LunaModel, output: Path) -> None:
    spec = model.spec.copy()
    # The SDK simulator applies explicit damping and gravity compensation.
    spec.option.timestep = 0.0005
    for body in spec.bodies:
        body.gravcomp = 0.
    model_qpos = np.zeros(model.model.nq)
    model_qpos[model.qpos] = HOME * FX_TO_MODEL
    for axis, sign in zip(JOINTS, FX_TO_MODEL):
        joint = spec.joint(axis)
        joint.axis *= sign
        joint.range = np.sort(joint.range * sign)
    spec.add_key(name="home", qpos=model_qpos, ctrl=HOME * FX_TO_MODEL)
    spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())


def probe(bridge: DdsBridge, scope: str, domain: int, limits: Limits | None) -> dict:
    frame = bridge.feedback(hardware=True, command=False, wait=4.)
    serial = frame.controller["frame_serial"]
    end = time.monotonic() + 1.6
    while frame.controller["frame_serial"] == serial:
        if time.monotonic() >= end:
            raise ValueError("FX frame serial did not advance during the probe")
        frame = bridge.feedback(hardware=True, command=False, wait=0.25, after=frame.stamp)

    violation = None
    if limits is not None:
        try:
            limits.check_position(frame.q)
        except ValueError as error:
            violation = str(error)
    snapshot = bridge.snapshot
    rdk_state = snapshot.get("rdk", {})
    mode_fields = ("hard_auto", "soft_auto", "joystick", "handle", "parking", "avoid")
    rdk_mode = None
    if rdk_state.get("available"):
        rdk_mode = {key: rdk_state.get(key) for key in mode_fields}
    return {
        "scope": scope, "domain": domain, "joint_axes": len(frame.q),
        "motion_stamp_ns": frame.stamp, "controller": frame.controller,
        "fx_position_rad": (frame.q * FX_TO_MODEL).tolist(),
        "base": frame.velocity.tolist(), "base_stationary": bool(
            np.all(np.abs(frame.velocity) <= STOPPED_VELOCITY)),
        "rdk_mode": rdk_mode,
        "joint_limits_ok": None if limits is None else violation is None,
        "joint_limit_violation": violation,
        "base_velocity_caps": None if limits is None else limits.base.tolist(),
        "command_endpoints": snapshot.get("command_endpoints"),
        "command_deadline_supported": snapshot.get("command_deadline_supported"),
        "group_stop_supported": bool(frame.controller.get("capabilities", 0) & 8),
    }


class Teleop:
    def __init__(self, bridge: DdsBridge, robot: LunaModel, policy: LunaPolicy, limits: Limits,
                 operator: Operator, mode: str, groups: tuple[str, ...], speed: float):
        self.bridge, self.robot, self.policy, self.limits = bridge, robot, policy, limits
        self.operator, self.mode, self.groups, self.speed = operator, mode, groups, speed
        self.hardware = mode != "sim"
        self.command = mode != "shadow"
        self.collision = CollisionCheck(LunaModel(robot.urdf, meshes=True))
        self.prepared: list[str] = []
        self.base_owned = False
        self.feedback: Feedback | None = None
        self.qd = np.zeros(len(JOINTS))
        self.anchor = None
        self.previous_target = None
        self.base_target = np.zeros(3)
        self.reanchor = 0
        self.active = False
        self.blocked = False
        self.status = "Waiting for DDS feedback"
        self.previous_frame: int | None = None
        self.frame_seen = 0.
        self.position_confirmed = False

    def read(self, wait: float = 0.03) -> Feedback:
        old = self.feedback
        feedback = self.bridge.feedback(hardware=self.hardware, command=self.command, wait=wait,
                                        after=old.stamp if old else 0)
        self.limits.check_position(feedback.q)
        now = time.monotonic()
        if self.hardware:
            serial = feedback.controller["frame_serial"]
            if serial != self.previous_frame:
                self.previous_frame, self.frame_seen = serial, now
            if now - self.frame_seen > 1.5:
                raise ValueError("FX frame serial stopped advancing")
        self.feedback = feedback
        if old is not None:
            dt = (feedback.stamp - old.stamp) * 1e-9
            self.qd = 0.55 * self.qd + 0.45 * ((feedback.q - old.q) / dt)
        return feedback

    def engage(self, feedback: Feedback) -> bool:
        if self.command and np.any(np.abs(feedback.velocity) > STOPPED_VELOCITY):
            self.status = "等待底盘停止后再准备关节"
            return False
        if self.hardware and self.command:
            groups = feedback.controller["groups"]
            if any(groups[name]["mode"] not in (0, 1) for name in self.groups):
                raise RuntimeError("FX groups must be in Idle or Position before preparation")
        if self.anchor is None:
            self.anchor = anchor_target(self.robot, feedback.q[None])
        missing = [group for group in self.groups if group not in self.prepared]
        if self.command and missing:
            self.status = "DDS 正在准备 FX Position 模式"
            self.publish(feedback)
            for group in missing:
                if time.monotonic() >= self.operator.jog().expires:
                    return False
                self.prepared.append(group)  # A timeout can still leave this group prepared.
                self.bridge.send([{"op": "prepare", "group": group, "speed_percent": self.speed,
                                   "acceleration_percent": self.speed}], timeout=3.1)
            feedback = self.read(wait=0.25)
        self.previous_target = feedback.q.copy()
        self.base_target[:] = 0.
        return True

    def joint_commands(self, target: np.ndarray) -> list[dict]:
        return [{"op": "joint", "group": group,
                 "positions": (target * FX_TO_MODEL)[GROUP_SLICES[group]].tolist()}
                for group in self.prepared]

    def pause(self, feedback: Feedback | None) -> None:
        was_driving = self.base_owned
        try:
            if self.active and self.prepared:
                if feedback is None:
                    raise ValueError("fresh FX feedback needed to hold position")
                self.bridge.send(self.joint_commands(feedback.q))
        finally:
            if was_driving:
                # SDK stop can clear joint preparation. Hold first and stop even if hold fails.
                self.bridge.send([{"op": "stop"}])
                self.base_owned = False
        if was_driving:
            self.prepared.clear()
            self.position_confirmed = False
        self.active = False
        self.base_target[:] = 0.

    def abort(self) -> None:
        try:
            self.pause(None)
        except (OSError, ValueError, RuntimeError) as error:
            if self.base_owned:
                print(f"DDS base stop unconfirmed: {error}", file=sys.stderr)
        # Priority DDS lane uses depth one, so acknowledge each group stop before sending another.
        for group in self.prepared:
            try:
                self.bridge.send([{"op": "emergency_stop", "group": group}], timeout=0.25)
            except (OSError, ValueError, RuntimeError) as error:
                print(f"DDS {group} stop unconfirmed: {error}", file=sys.stderr)
        self.prepared.clear()
        self.base_owned = self.active = False
        self.position_confirmed = False

    def step(self) -> None:
        feedback = self.read()
        jog = self.operator.jog()
        if time.monotonic() >= jog.expires:
            self.pause(feedback)
            self.blocked = False
            if jog.reanchor != self.reanchor or self.anchor is None:
                self.anchor = anchor_target(self.robot, feedback.q[None])
                self.reanchor = jog.reanchor
            self.status = "DDS 已就绪，按住运行开始遥操"
            self.publish(feedback)
            return
        if self.blocked:
            return
        if not self.active:
            self.active = self.engage(feedback)
            feedback = self.feedback
            jog = self.operator.jog()
            if not self.active or time.monotonic() >= jog.expires:
                self.publish(feedback)
                return
        if self.hardware and self.command:
            groups = feedback.controller["groups"]
            if any(groups[name]["mode"] != 1 for name in self.prepared):
                if self.position_confirmed:
                    raise RuntimeError("FX Position mode was lost during teleoperation")
                # ControllerState is decimated; defer the first movement until the new mode arrives.
                self.status = "等待 DDS 确认 FX Position 模式"
                self.publish(feedback)
                return
            self.position_confirmed = True
        self.move(feedback, jog)

    def move(self, feedback: Feedback, jog: Jog) -> None:
        values = jog.values
        hand = values[:12].reshape(1, 2, 6)
        goal = offset_target(self.anchor, hand[..., :3], hand[..., 3:], values[None, 12:15],
                             values[None, 15:])
        control = tracking(self.robot, feedback.q[None], self.qd[None],
                           feedback.velocity[None], goal)
        twist, angles = decode(control, self.policy(control.observation), feedback.q[None],
                               self.limits.position, self.limits.base)
        q = feedback.q
        target = np.clip(angles[0], self.previous_target - DT * self.limits.rate,
                         self.previous_target + DT * self.limits.rate)
        target = np.clip(target, q - LEAD_TIME * self.limits.rate,
                         q + LEAD_TIME * self.limits.rate)
        for group, section in GROUP_SLICES.items():
            if group not in self.groups:
                target[section] = q[section]
        self.collision.check(q, target)
        speed = self.base_target + np.clip(twist[0] - self.base_target,
                                          -DT * BASE_ACCEL, DT * BASE_ACCEL)
        speed[twist[0] == 0.] = 0.
        remains = jog.expires - time.monotonic()
        if remains <= 0.:
            self.pause(feedback)
            return
        if self.command and self.base_owned and not speed.any():
            self.pause(feedback)
            self.status = "等待底盘停止后再准备关节"
            self.publish(feedback)
            return
        if self.command:
            requests = self.joint_commands(target)
            if speed.any():
                self.base_owned = True
                requests.append({"op": "base", "velocity": speed.tolist()})
            self.bridge.send(requests, expires_ns=time.time_ns() + int(min(remains, 0.12) * 1e9))
            self.base_owned = bool(speed.any())
        self.previous_target, self.base_target = target, speed
        self.status = "策略通过 DDS 遥操" if self.command else "策略影子运行"
        errors = np.linalg.norm(control.position_error[0], axis=1).tolist()
        self.publish(feedback, hand_error_m=errors,
                     fx_target_rad=(target * FX_TO_MODEL).round(4).tolist(),
                     base_target=speed.round(4).tolist())

    def publish(self, feedback: Feedback, **fields) -> None:
        self.operator.status(status=self.status, mode=self.mode,
                             fx_position_rad=(feedback.q * FX_TO_MODEL).round(4).tolist(),
                             base_measured=feedback.velocity.round(4).tolist(), **fields)

    def run(self, stopping) -> None:
        while not stopping():
            before = time.monotonic()
            try:
                self.step()
            except (ValueError, RuntimeError, OSError) as error:
                if self.prepared or self.base_owned:
                    self.abort()
                self.blocked = True
                self.active = False
                self.operator.status(status=str(error), mode=self.mode)
                if self.bridge.failure:
                    raise
            time.sleep(max(0., DT - (time.monotonic() - before)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = ("probe", "shadow", "live", "sim", "view", "make-sdk-model")
    parser.add_argument("mode", choices=modes)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--domain", type=int)
    parser.add_argument("--scope", choices=("luna_observe", "luna_control"))
    parser.add_argument("--urdf", type=Path, default=default_urdf())
    parser.add_argument("--policy", type=Path, default=default_policy())
    parser.add_argument("--limits", type=Path, help="calibration JSON from the active DDS gateway")
    parser.add_argument("--output", type=Path, help="output path for make-sdk-model")
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    parser.add_argument("--speed-percent", type=float, default=5.)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    if not 1. <= args.speed_percent <= 100.:
        parser.error("--speed-percent must be between 1 and 100")
    if args.mode == "make-sdk-model":
        if args.output is None:
            parser.error("make-sdk-model requires --output")
        sdk_sim_model(LunaModel(args.urdf), args.output)
        return
    if args.bridge is None:
        parser.error("--bridge must point to gento_sdk_dds_ui_bridge")
    if args.mode == "live" and (args.scope == "luna_observe" or args.limits is None):
        parser.error("live requires the luna_control scope and its machine-calibrated --limits")
    simulation = args.mode in ("sim", "view")
    scope = "luna_sim" if simulation else args.scope or (
        "luna_control" if args.mode == "live" else "luna_observe"
    )
    domain = args.domain if args.domain is not None else (225 if simulation else 226)
    bridge = DdsBridge(args.bridge, scope, domain)
    runtime = operator = None
    stop = False

    def shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        if args.mode == "view":
            from .mujoco_viewer import show_sim

            show_sim(bridge, args.urdf, lambda: stop)
            return
        limits = Limits.from_gateway(args.limits) if args.limits else None
        if args.mode == "probe":
            print(json.dumps(probe(bridge, scope, domain, limits), indent=2, allow_nan=False))
            return
        model = LunaModel(args.urdf)
        if limits is None:
            limits = Limits(model.bounds, JOINT_RATE, BASE_RATE)
        policy = LunaPolicy(args.policy, model)
        operator = Operator(args.port)
        groups = tuple(name for name in GROUPS if name in args.groups)
        runtime = Teleop(bridge, model, policy, limits, operator, args.mode, groups,
                         args.speed_percent)
        print(f"{args.mode} · DDS {domain}/{scope} · http://127.0.0.1:{args.port}", flush=True)
        runtime.run(lambda: stop)
    finally:
        if runtime is not None and (runtime.prepared or runtime.base_owned):
            try:
                runtime.pause(runtime.read(wait=0.15))
            except (OSError, ValueError, RuntimeError):
                runtime.abort()
        if operator is not None:
            operator.close()
        bridge.close()


if __name__ == "__main__":
    main()
