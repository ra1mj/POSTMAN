"""TIAGo++ model metadata and optional MuJoCo/Menagerie loader."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TiagoPPConfig:
    name: str = "tiago_pp"
    # nu=25 actuators; the floating-base MJCF has nq=32 and nv=31.
    dof: int = 25
    qpos_dim: int = 32
    qvel_dim: int = 31
    joint_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "wheel_front_right_joint",
            "wheel_front_left_joint",
            "wheel_rear_right_joint",
            "wheel_rear_left_joint",
            "torso_lift_joint",
            "head_1_joint",
            "head_2_joint",
            *(f"arm_left_{index}_joint" for index in range(1, 8)),
            "gripper_left_right_finger_joint",
            "gripper_left_left_finger_joint",
            *(f"arm_right_{index}_joint" for index in range(1, 8)),
            "gripper_right_right_finger_joint",
            "gripper_right_left_finger_joint",
        )
    )
    group_slices: Mapping[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "base": (0, 4),
            "torso": (4, 5),
            "head": (5, 7),
            "left_arm": (7, 14),
            "left_gripper": (14, 16),
            "right_arm": (16, 23),
            "right_gripper": (23, 25),
        }
    )
    default_q: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 25)
    control_frequency_hz: float = 100.0
    physics_frequency_hz: float = 500.0

    def __post_init__(self) -> None:
        if self.dof != len(self.joint_names) or self.dof != len(self.default_q):
            raise ValueError("TIAGo++ joint metadata has inconsistent dimensions")
        if self.qpos_dim <= 0 or self.qvel_dim <= 0:
            raise ValueError("TIAGo++ qpos/qvel dimensions must be positive")
        if self.physics_frequency_hz < self.control_frequency_hz:
            raise ValueError("physics frequency must be >= control frequency")


TIAGO_PP = TiagoPPConfig()


@dataclass(frozen=True)
class TiagoPPScene:
    """Configuration-only scene wrapper; the actual XML remains an external asset."""

    mjcf_path: Path | None = None

    def resolve_mjcf(self) -> Path:
        candidates = []
        if self.mjcf_path is not None:
            candidates.append(Path(self.mjcf_path))
        if os.getenv("POSTMAN_TIAGO_PP_MJCF"):
            candidates.append(Path(os.environ["POSTMAN_TIAGO_PP_MJCF"]))
        candidates.extend(
            [
                Path("assets/robots/tiago_pp/scene.xml"),
                Path("mujoco_menagerie/pal_tiago_dual/scene.xml"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        try:
            import mujoco_menagerie as menagerie

            # ``xml()`` downloads the pinned official asset into the user cache
            # and returns the default scene XML path.
            return Path(menagerie.get("pal_tiago_dual").xml()).resolve()
        except (ImportError, OSError, RuntimeError, KeyError) as exc:
            menagerie_error = str(exc)
        else:  # pragma: no cover - return above is unconditional
            menagerie_error = "unknown menagerie error"
        raise FileNotFoundError(
            "TIAGo++ MJCF was not found. Clone MuJoCo Menagerie or set "
            "POSTMAN_TIAGO_PP_MJCF to an XML scene path. "
            f"Menagerie lookup failed: {menagerie_error}"
        )

    def resolve_robot_mjcf(self, entry: str = "tiago_dual_position") -> Path:
        """Resolve the robot-only Menagerie XML (without its floor scene)."""
        if self.mjcf_path is not None:
            return self.mjcf_path.resolve()
        try:
            import mujoco_menagerie as menagerie

            return Path(menagerie.get("pal_tiago_dual").xml(entry)).resolve()
        except (ImportError, OSError, RuntimeError, KeyError) as exc:
            raise FileNotFoundError(
                f"TIAGo++ robot XML entry {entry!r} is unavailable; install "
                "mujoco-menagerie or set mjcf_path"
            ) from exc

    def load_mujoco(self):
        """Load the scene if MuJoCo is installed, with a useful error otherwise."""
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[sim] to load MuJoCo scenes") from exc
        return mujoco.MjModel.from_xml_path(str(self.resolve_mjcf()))
