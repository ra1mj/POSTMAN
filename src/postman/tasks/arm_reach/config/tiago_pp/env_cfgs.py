"""TIAGo++ arm-reach environment configurations."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from postman.asset_zoo.robots.tiago_pp import (
    get_tiago_pp_robot_cfg,
    get_tiago_pp_table_robot_cfg,
)


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=dict(actor_terms),
            enable_corruption=False,
        ),
    }


def _actions() -> dict[str, object]:
    return {
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
    }


def _make_env_cfg(*, play: bool, table: bool) -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=3.0),
            entities={
                "robot": get_tiago_pp_table_robot_cfg()
                if table
                else get_tiago_pp_robot_cfg()
            },
            num_envs=1,
        ),
        observations=_observations(play),
        actions=_actions(),
        events={},
        rewards={
            "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
            "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
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
        episode_length_s=1e9 if play else 10.0,
    )


def tiago_pp_arm_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Bare TIAGo++ arm reach configuration."""
    return _make_env_cfg(play=play, table=False)


def tiago_pp_arm_reach_table_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """TIAGo++ arm reach configuration with a table background."""
    return _make_env_cfg(play=play, table=True)
