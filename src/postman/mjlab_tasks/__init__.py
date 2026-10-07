"""Native mjlab task registrations for POSTMAN.

This module is imported through the ``mjlab.tasks`` entry-point group after
POSTMAN is installed. It is deliberately separate from the dependency-light
CPU reference environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.config.yam.rl_cfg import yam_lift_cube_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from postman.robots import TiagoPPScene
from postman.sim.scene_composer import FurniturePlacement, compose_tiago_spec


def _robot_entity() -> EntityCfg:
    scene = TiagoPPScene()
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(scene.resolve_robot_mjcf("tiago_dual"))),
        articulation=EntityArticulationInfoCfg(
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
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )


def _furniture_robot_entity() -> EntityCfg:
    root = TiagoPPScene()
    repo_root = Path(__file__).resolve().parents[3]
    furniture_root = Path(
        os.environ.get(
            "POSTMAN_FURNITURE_SCENE_ROOT",
            str(repo_root / "third_party/scenes/furniture_sim"),
        )
    )
    furniture = FurniturePlacement(
        path=furniture_root / "simpleTable.xml",
        position=(1.0, 0.0, 0.0),
        prefix="table_",
    )
    # The workspace path above is only used when the package is installed from
    # this checkout. If the asset is absent, fail with a clear message when the
    # task is selected rather than during package discovery.
    return EntityCfg(
        spec_fn=lambda: compose_tiago_spec(
            [furniture], robot_scene=root.resolve_robot_mjcf("tiago_dual")
        ),
        articulation=EntityArticulationInfoCfg(
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
        ),
    )


def _cfg(*, play: bool = False, furniture: bool = False) -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(terms=actor_terms, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms=dict(actor_terms), enable_corruption=False),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=3.0),
            entities={"robot": _furniture_robot_entity() if furniture else _robot_entity()},
            num_envs=1,
        ),
        observations=observations,
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(r"(torso|head|arm|gripper).*",),
                scale=0.25,
                use_default_offset=True,
            ),
            "wheel_vel": JointVelocityActionCfg(
                entity_name="robot",
                actuator_names=(r"wheel_.*",),
                scale=5.0,
                use_default_offset=False,
            ),
        },
        events={},
        rewards={
            "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
            "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        },
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        sim=SimulationCfg(
            nconmax=256,
            njmax=2000,
            mujoco=MujocoCfg(timestep=0.005, iterations=10),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=3.0,
            elevation=-10.0,
            azimuth=120.0,
        ),
        decimation=4,
        episode_length_s=10.0 if not play else 1e9,
    )
    return cfg


def _rl_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = yam_lift_cube_ppo_runner_cfg()
    cfg.experiment_name = "postman_tiago_pp"
    cfg.num_steps_per_env = 24
    cfg.save_interval = 100
    return cfg


register_mjlab_task(
    task_id="Postman-TiagoPP-ArmReach",
    env_cfg=_cfg(),
    play_env_cfg=_cfg(play=True),
    rl_cfg=_rl_cfg(),
)

register_mjlab_task(
    task_id="Postman-TiagoPP-ArmReach-SimpleTable",
    env_cfg=_cfg(furniture=True),
    play_env_cfg=_cfg(play=True, furniture=True),
    rl_cfg=_rl_cfg(),
)
