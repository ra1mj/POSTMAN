#!/usr/bin/env python3
"""Record three Luna whole-body actions using the native DDS MuJoCo node."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from postman.luna.control import LunaPolicy, default_policy, rotation_xyz
from postman.luna.dds import DdsBridge, Limits
from postman.luna.model import (
    BASE_RATE,
    DT,
    GROUP_SLICES,
    HANDS,
    JOINT_RATE,
    LunaModel,
    default_urdf,
)
from postman.luna.mujoco_viewer import LunaView
from postman.luna.teleop import Teleop, sdk_sim_model
from postman.luna.web import Jog

DDS_LOOPBACK = """<CycloneDDS><Domain Id="any"><General><Interfaces>
<NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General>
<Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer Address="127.0.0.1"/>
</Peers></Discovery></Domain></CycloneDDS>"""
SIZE = (1280, 720)
CAPTIONS = (
    ("01  双臂高抬外展", "双手向前、向外、向上抬，腰部随动", 8.5),
    ("02  左臂大幅外展", "左手走弧线，右手定点，头部向左转", 9.0),
    ("03  底盘绕弧，双手定点", "底盘沿弧线移动，腰臂追踪地面定点", 12.0),
)


class ScriptedInput:
    def __init__(self):
        self.command = Jog(np.zeros(17), 0., 0)

    def jog(self):
        return self.command

    def status(self, **_fields):
        pass


def ease(t, begin, duration):
    x = np.clip((t - begin) / duration, 0., 1.)
    return x**2 * (3. - 2. * x)


def to_world(points, base):
    rotation = rotation_xyz(np.array([0., 0., base[2]]))
    return points @ rotation.T + np.r_[base[:2], 0.]


def targets(index, t, base, origin, hands):
    """Position/rotation offsets, body-frame base velocity and viewer targets."""
    values = np.zeros(17)
    positions = values[:12].reshape(2, 6)[:, :3]
    travel = ease(t, 0.5, 4.2)
    if index == 0:
        positions[:] = travel * np.array([[0.13, 0.095, 0.27], [0.13, -0.095, 0.27]])
        return values, to_world(hands + positions, base)
    if index == 1:
        swing = ease(t, 1.25, 4.)
        positions[0] = [0.09 * travel, 0.225 * swing, 0.23 * travel]
        values[[5, 15]] = swing * np.array([0.34, 0.50])
        return values, to_world(hands + positions, base)

    fixed = to_world(hands + np.array([0.105, 0., 0.070]) * ease(t, 0.5, 2.4), origin)
    rotation = rotation_xyz(np.array([0., 0., base[2]]))
    positions[:] = (fixed - np.r_[base[:2], 0.]) @ rotation - hands
    values[[5, 11]] = 0.55 * (origin[2] - base[2])
    angle = -0.73 * ease(t, 3.1, 5.7)
    pivot = fixed[:, :2].mean(axis=0)
    turn = rotation_xyz(np.array([0., 0., angle]))[:2, :2]
    desired = np.r_[pivot + turn @ (origin[:2] - pivot), origin[2] + angle]
    error = desired - base
    values[12:14] = np.clip((np.r_[error[:2], 0.] @ rotation)[:2] * 2.4, -0.14, 0.14)
    values[14] = np.clip(error[2] * 2.6, -0.25, 0.25)
    return values, fixed


def draw_marker(scene, position, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, 0.016), position, np.eye(3).ravel(), rgba)
    geom.emission = 0.65
    scene.ngeom += 1


def annotate(rgb, fonts, title, subtitle, changes, travel, yaw, error):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    label_width = max(draw.textlength(title, font=fonts[0]),
                      draw.textlength(subtitle, font=fonts[1]))
    draw.rounded_rectangle((28, 25, 70 + label_width, 118), radius=12, fill=(13, 21, 33, 212))
    draw.text((49, 34), title, font=fonts[0], fill=(242, 247, 255))
    draw.text((49, 78), subtitle, font=fonts[1], fill=(201, 213, 229))
    draw.rounded_rectangle((28, 641, 1252, 695), radius=12, fill=(13, 21, 33, 215))
    line = ("实测变化    腰部 %.1f°    左臂 %.1f°    右臂 %.1f°    头部 %.1f°    "
            "底盘 %.0f cm / %+.1f°    手部平均误差 %.1f cm")
    draw.text((49, 653), line % (*changes, travel * 100, yaw, error * 100),
              font=fonts[1], fill=(230, 240, 253))
    return np.asarray(image)


def record_segment(index, writer, renderer, scene, robot, policy, bridge, fonts):
    title, caption, duration = CAPTIONS[index]
    limits = Limits(robot.bounds, JOINT_RATE, BASE_RATE)
    operator = ScriptedInput()
    runtime = Teleop(bridge, robot, policy, limits, operator, "sim", tuple(GROUP_SLICES), 16.)
    runtime.feedback = bridge.feedback(hardware=False, command=False, wait=4.)
    runtime.step()
    initial = runtime.feedback
    origin = initial.base_pose.copy()
    hands = runtime.anchor.hands[0]
    camera = mujoco.MjvCamera()
    scene.aim(camera)
    camera.lookat[:2] = origin[:2] + ([0.09, 0.17] if index == 2 else [0.10, 0.04])
    camera.distance = 3.14
    camera.azimuth = (148., 206., 154.)[index]
    camera.elevation = -17.
    left, right = [scene.model.body(name).id for name in HANDS]
    colors = ([0.26, 0.87, 1., 0.88], [1., 0.68, 0.30, 0.88])
    start = time.monotonic()
    peak_joint_change = np.zeros(4)
    last_metrics = None
    try:
        for i in range(round(duration / DT)):
            time.sleep(max(0., start + i * DT - time.monotonic()))
            t = i * DT
            base = runtime.feedback.base_pose
            values, _ = targets(index, t, base, origin, hands)
            # Scripted workspace targets are used only in the isolated MuJoCo scene.
            if not np.isfinite(values).all():
                raise ValueError(f"{title} has a non-finite target at {t:.2f} s")
            expires = time.monotonic() + 0.20 if t < duration - 0.65 else 0.
            operator.command = Jog(values, expires, 0)
            runtime.step()
            frame = runtime.feedback
            scene.update(frame)
            _, world_targets = targets(index, t, frame.base_pose, origin, hands)
            actual = scene.data.xpos[[left, right]]
            error = np.linalg.norm(world_targets - actual, axis=-1).mean()
            angles = np.rad2deg(np.abs(frame.q - initial.q))
            change = np.array([angles[section].max() for section in GROUP_SLICES.values()])
            peak_joint_change = np.maximum(peak_joint_change, change)
            travel = np.linalg.norm(frame.base_pose[:2] - origin[:2])
            yaw = np.rad2deg(frame.base_pose[2] - origin[2])
            last_metrics = {"joint_max_deg": peak_joint_change.round(2).tolist(),
                            "base_travel_cm": round(float(travel * 100), 2),
                            "base_yaw_deg": round(float(yaw), 2),
                            "hand_error_cm": round(float(error * 100), 2)}
            renderer.update_scene(scene.data, camera=camera)
            for point, color in zip(world_targets, colors):
                draw_marker(renderer.scene, point, np.array(color, dtype=np.float32))
            image = annotate(renderer.render(), fonts, title, caption, change, travel, yaw, error)
            writer.send(image)
    except BaseException:
        runtime.abort()
        raise
    else:
        if runtime.active:
            runtime.pause(runtime.read(wait=0.15))
    print(title, last_metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-build", type=Path, default=Path("build/gento"))
    parser.add_argument("--urdf", type=Path, default=default_urdf())
    parser.add_argument("--output", type=Path, default=Path("outputs/luna_680_whole_body.mp4"))
    parser.add_argument("--font", type=Path, default=Path(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    robot = LunaModel(args.urdf)
    policy = LunaPolicy(default_policy(), robot)
    scene = LunaView(args.urdf)
    scene.model.vis.global_.offwidth, scene.model.vis.global_.offheight = SIZE
    fonts = [ImageFont.truetype(str(args.font), size=size, index=2) for size in (33, 22)]
    sdk = args.sdk_build.resolve()
    domain = 174 + os.getpid() % 27
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="postman-luna-record-") as directory:
        tmp = Path(directory)
        mjcf = tmp / "luna.xml"
        sdk_sim_model(robot, mjcf)
        config = tmp / "cyclonedds.xml"
        config.write_text(DDS_LOOPBACK)
        os.environ["CYCLONEDDS_URI"] = config.as_uri()
        env = {**os.environ, "GENTO_DDS_DOMAIN": str(domain)}
        with mujoco.Renderer(scene.model, width=SIZE[0], height=SIZE[1]) as renderer:
            # Allocate graphics buffers before the first operator command.
            renderer.update_scene(scene.data)
            renderer.render()
            writer = imageio_ffmpeg.write_frames(
                str(args.output), SIZE, fps=round(1. / DT), codec="libx264", quality=8,
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
            try:
                for index in range(len(CAPTIONS)):
                    with subprocess.Popen([
                        str(sdk / "gento_sdk_dds_sim_node"), "--model", "luna", "--scope",
                        "luna_sim", "--mjcf", str(mjcf),
                    ], env=env, stdout=subprocess.DEVNULL) as sim:
                        try:
                            with closing(DdsBridge(sdk / "gento_sdk_dds_ui_bridge", "luna_sim",
                                                  domain)) as bridge:
                                record_segment(index, writer, renderer, scene, robot, policy,
                                               bridge, fonts)
                        finally:
                            sim.terminate()
                            sim.wait(timeout=4)
            finally:
                writer.close()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
