"""Gento Luna V3 680 model and FX/RDK axis contract."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

GROUPS = {
    "waist": tuple(f"Leg_J{i}_Joint" for i in range(1, 7)),
    "left_arm": tuple(f"Arm_L{i}_Joint" for i in range(1, 8)),
    "right_arm": tuple(f"Arm_R{i}_Joint" for i in range(1, 8)),
    "head": ("Head_J1_joint", "Head_J2_Joint"),
}
JOINTS = tuple(joint for group in GROUPS.values() for joint in group)
GROUP_SLICES = {"waist": slice(0, 6), "left_arm": slice(6, 13),
                "right_arm": slice(13, 20), "head": slice(20, 22)}
# Delivered joint_signs.json for this URDF. Apply only at the SDK boundary.
FX_TO_MODEL = np.array([1.] * 21 + [-1.])
HANDS = ("flange_L", "flange_R")
ACTION_SIZE = 3 + len(JOINTS)
DT = 0.05
BASE_RATE = np.array([0.22, 0.18, 0.35])
JOINT_RATE = np.repeat([0.24, 0.50, 0.50, 0.40], [6, 7, 7, 2])
# Delivered MJCF Home, used as a simulation seed. Live anchors come from FX feedback.
HOME = np.array([
    0., 0.4, -0.6, 0.2, 0., 0.,
    1.57, -1.41, -1.3, -1.57, -0.151, -0.01, 0.157,
    -1.57, -1.41, 1.3, -1.57, 0.151, -0.01, -0.157,
    0., 0.,
])


def default_urdf() -> Path:
    configured = os.environ.get("POSTMAN_LUNA_URDF")
    if configured:
        return Path(configured).expanduser()
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    package = data / "microagi-gento/luna-v3-680/marvin_description"
    return package / "urdf/Gento_Luna_v3_680/Gento_Luna_v3_680.urdf"


def _spec(path: Path, *, meshes: bool) -> tuple[mujoco.MjSpec, np.ndarray]:
    root = ET.parse(path).getroot()
    names = {j.attrib["name"]: j for j in root.findall("joint") if j.get("type") != "fixed"}
    if (root.get("name", "").lower() != "gento_luna_v3_680" or set(names) != set(JOINTS)
            or any(joint.get("type") != "revolute" for joint in names.values())):
        raise ValueError("expected the delivered Gento Luna V3 680 URDF with 22 rotary axes")
    limits = np.array([
        [float(names[j].find("limit").attrib[k]) for k in ("lower", "upper", "effort")]
        for j in JOINTS
    ])
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.update(fusestatic="false", strippath="false", discardvisual="true")
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        for collision in link.findall("collision"):
            if not meshes:
                link.remove(collision)
                continue
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib["filename"]
            if filename.startswith("package://"):
                relative = Path(filename.removeprefix("package://"))
                mesh_path = next((p / relative for p in path.parents if (p / relative).is_file()),
                                 None)
                if mesh_path is None:
                    raise FileNotFoundError(f"missing Luna mesh {filename}")
            else:
                mesh_path = path.parent / filename
            mesh.set("filename", str(mesh_path))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for mesh in spec.meshes:
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        source = Path(mesh.file)
        if source.suffix.lower() != ".stl" or source.stat().st_size <= 10_000_084:
            continue
        # MuJoCo's STL decoder caps the number of faces at 200k. Its collision
        # hull can be built directly from the original vertex cloud.
        triangle = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)),
                             ("attribute", "<u2")])
        with source.open("rb") as stream:
            stream.seek(80)
            count = int.from_bytes(stream.read(4), "little")
            if source.stat().st_size != 84 + count * triangle.itemsize:
                raise ValueError(f"invalid binary STL {source}")
            vertices = np.fromfile(stream, dtype=triangle)["vertices"].reshape(-1, 3)
        mesh.file = ""
        mesh.uservert = np.unique(vertices, axis=0).ravel()
    return spec, limits


class LunaModel:
    def __init__(self, urdf: Path, *, meshes: bool = False):
        self.urdf = Path(urdf).resolve()
        self.fingerprint = hashlib.sha256(self.urdf.read_bytes()).hexdigest()
        spec, limits = _spec(self.urdf, meshes=meshes)
        self.bounds = limits[:, :2].copy()
        spec.option.timestep = DT / 10
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for body in spec.bodies:
            body.gravcomp = 1.0
        for index, name in enumerate(JOINTS):
            joint = spec.joint(name)
            joint.armature = 0.12 if index < 6 else 0.05
            kp = 180.0 if index < 6 else 70.0
            spec.add_actuator(
                name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp, *([0.] * 9)],
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                biasprm=[0., -kp, -0.13 * kp, *([0.] * 7)],
                forcelimited=True, forcerange=[-limits[index, 2], limits[index, 2]],
            )
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name).id for name in JOINTS]
        self.qpos = self.model.jnt_qposadr[joints]
        self.qvel = self.model.jnt_dofadr[joints]
        self.hand_bodies = [self.model.body(name).id for name in HANDS]
        self.chain = np.array([[True] * 6 + [side == 0] * 7 + [side == 1] * 7 + [False] * 2
                               for side in range(2)])

    def kinematics(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch FK and geometric Jacobians, expressed in base_link."""
        n = len(position)
        hands = np.empty((n, 2, 3))
        rotations = np.empty((n, 2, 3, 3))
        anchors = np.empty((n, len(JOINTS), 3))
        axes = np.empty_like(anchors)
        for i, row in enumerate(position):
            self.data.qpos[self.qpos] = row
            mujoco.mj_kinematics(self.model, self.data)
            hands[i] = self.data.xpos[self.hand_bodies]
            rotations[i] = self.data.xmat[self.hand_bodies].reshape(2, 3, 3)
            # Only one DOF per joint so the DOF addresses also index xaxis/xanchor.
            anchors[i] = self.data.xanchor[self.qvel]
            axes[i] = self.data.xaxis[self.qvel]
        jac_pos = np.cross(axes[:, None], hands[:, :, None] - anchors[:, None])
        jac_rot = np.broadcast_to(axes[:, None], jac_pos.shape)
        jacobian = np.concatenate((jac_pos, jac_rot), axis=-1) * self.chain[None, :, :, None]
        return hands, rotations, jacobian.swapaxes(-1, -2).reshape(n, 12, len(JOINTS))


class CollisionCheck:
    """Check the short FX position step against the delivered Luna convex hulls."""

    def __init__(self, model: LunaModel):
        self.robot = model
        # Fixed/mating hulls overlap even in the delivered Home pose.
        overlaps = {frozenset(pair) for pair in (
            ("base_link", "Leg_J1_Link"), ("base_link", "Leg_J2_Link"),
            ("Leg_J3_Link", "Leg_J5_Link"), ("Arm_L5_Link", "Arm_L7_Link"),
            ("Arm_R5_Link", "Arm_R7_Link"),
        )}
        self.body_names = [model.model.body(i).name for i in model.model.geom_bodyid]
        self.overlaps = overlaps

    def penetrations(self, q: np.ndarray) -> dict[frozenset, float]:
        model, data = self.robot.model, self.robot.data
        data.qpos[self.robot.qpos] = q
        mujoco.mj_fwdPosition(model, data)
        depths: dict[frozenset, float] = {}
        for contact in data.contact:
            pair = frozenset(self.body_names[i] for i in contact.geom)
            if pair not in self.overlaps:
                depths[pair] = max(depths.get(pair, 0.), -contact.dist)
        return depths

    def check(self, measured: np.ndarray, target: np.ndarray) -> None:
        current = self.penetrations(measured)
        for q in ((measured + target) / 2, target):
            for pair, depth in self.penetrations(q).items():
                if depth > current.get(pair, 0.) + 0.0005:
                    bodies = " / ".join(sorted(pair))
                    raise ValueError(f"Luna self-collision prediction between {bodies}")
