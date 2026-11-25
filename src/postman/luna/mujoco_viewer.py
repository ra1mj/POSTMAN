"""Display the Luna simulation by following its native DDS feedback."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

from .dds import DdsBridge, Feedback
from .model import HOME, JOINTS, LunaModel


class LunaView:
    def __init__(self, urdf: Path):
        spec = LunaModel(urdf, meshes=True).spec
        spec.modelname = "Luna V3 680 - DDS simulation"
        spec.compiler.discardvisual = False
        spec.body("base_link").add_freejoint(name="viewer_base")
        for geom in spec.geoms:
            body = geom.parent.name
            if body.startswith(("Base_L", "Arm_L", "flange_L")):
                geom.rgba = [0.27, 0.53, 0.78, 1.]
            elif body.startswith(("Base_R", "Arm_R", "flange_R")):
                geom.rgba = [0.85, 0.54, 0.24, 1.]
            elif body.startswith(("front_", "rear_", "base_")):
                geom.rgba = [0.25, 0.30, 0.36, 1.]
            else:
                geom.rgba = [0.78, 0.81, 0.84, 1.]
        world = spec.worldbody
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0., 0., -0.012],
                       size=[4., 4., 0.05], rgba=[0.20, 0.23, 0.28, 1.],
                       contype=0, conaffinity=0)
        for offset in np.arange(-3., 3.01, 0.25):
            for axis in (0, 1):
                pos, size = [0., 0., -0.010], [3., 3., 0.001]
                pos[axis], size[axis] = float(offset), 0.0016
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                               rgba=[0.34, 0.38, 0.44, 1.], contype=0, conaffinity=0)
        world.add_light(pos=[0., 0., 4.], dir=[0., 0., -1.], diffuse=[0.85, 0.85, 0.85])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.root_qpos = int(self.model.joint("viewer_base").qposadr[0])
        self.qpos = [int(self.model.joint(name).qposadr[0]) for name in JOINTS]
        self.data.qpos[self.qpos] = HOME
        mujoco.mj_forward(self.model, self.data)

    def update(self, frame: Feedback) -> None:
        self.data.qpos[self.qpos] = frame.q
        if frame.base_pose is not None:
            x, y, yaw = frame.base_pose
            self.data.qpos[self.root_qpos:self.root_qpos + 7] = [
                x, y, 0., np.cos(yaw / 2), 0., 0., np.sin(yaw / 2),
            ]
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def aim(camera: mujoco.MjvCamera) -> None:
        camera.lookat[:] = (0., 0., 0.80)
        camera.distance = 2.85
        camera.azimuth = 150.
        camera.elevation = -13.


def show_sim(bridge: DdsBridge, urdf: Path, stopping) -> None:
    scene = LunaView(urdf)
    frame = bridge.feedback(hardware=False, command=False, wait=4.)
    scene.update(frame)
    with viewer.launch_passive(scene.model, scene.data, show_left_ui=False,
                               show_right_ui=False) as window:
        with window.lock():
            scene.aim(window.cam)
        while window.is_running() and not stopping():
            try:
                update = bridge.feedback(hardware=False, command=False, after=frame.stamp)
            except ValueError:
                pass
            else:
                with window.lock():
                    scene.update(update)
                frame = update
            window.sync()
            time.sleep(0.016)
