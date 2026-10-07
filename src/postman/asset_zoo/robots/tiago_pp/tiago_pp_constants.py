"""TIAGo++ asset and actuator configuration for native mjlab tasks."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from postman.robots import TiagoPPScene
from postman.sim.scene_composer import FurniturePlacement, compose_tiago_spec

TIAGO_PP_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinPositionActuatorCfg(
            target_names_expr=(r"(torso|head|arm|gripper).*",),
            stiffness=120.0,
            damping=8.0,
            effort_limit=100.0,
        ),
        BuiltinVelocityActuatorCfg(
            target_names_expr=(r"wheel_.*",),
            damping=2000.0,
            effort_limit=100.0,
        ),
    )
)


def _initial_state() -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )


def get_tiago_pp_robot_cfg() -> EntityCfg:
    """Return a fresh bare TIAGo++ entity config."""
    scene = TiagoPPScene()
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(scene.resolve_robot_mjcf("tiago_dual"))),
        articulation=TIAGO_PP_ARTICULATION,
        init_state=_initial_state(),
    )


def get_tiago_pp_table_robot_cfg() -> EntityCfg:
    """Return TIAGo++ with the checked-in SimpleTable attached."""
    scene = TiagoPPScene()
    repo_root = Path(__file__).resolve().parents[5]
    furniture_root = Path(
        os.environ.get(
            "POSTMAN_FURNITURE_SCENE_ROOT",
            str(repo_root / "third_party/scenes/furniture_sim"),
        )
    )
    table = FurniturePlacement(
        path=furniture_root / "simpleTable.xml",
        position=(1.0, 0.0, 0.0),
        prefix="table_",
    )
    return EntityCfg(
        spec_fn=lambda: compose_tiago_spec(
            [table], robot_scene=scene.resolve_robot_mjcf("tiago_dual")
        ),
        articulation=TIAGO_PP_ARTICULATION,
        init_state=_initial_state(),
    )
