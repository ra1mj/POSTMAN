"""TIAGo++ arm-reach environment configurations."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
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
from postman.tasks.arm_reach.mdp.randomization import randomize_dynamics
from postman.tasks.arm_reach.wbc_runtime import (
    RlWbcActionCfg,
    SyntheticUrrCommandCfg,
    action_smoothness_reward,
    command_tracking_reward,
    joint_limit_penalty,
    no_oscillation_penalty,
    privileged_command_context,
    urr_command,
    urr_command_delta,
)


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "urr_command": ObservationTermCfg(func=urr_command),
        "urr_command_delta": ObservationTermCfg(func=urr_command_delta),
        "actions": ObservationTermCfg(
            func=mdp.last_action, params={"action_name": "whole_body"}
        ),
    }
    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms={
                **actor_terms,
                "privileged_context": ObservationTermCfg(func=privileged_command_context),
            },
            enable_corruption=False,
        ),
    }


def _actions() -> dict[str, object]:
    return {
        "whole_body": RlWbcActionCfg(
            entity_name="robot",
            dt=0.01,
            alpha=0.25,
            max_delay=2,
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
        commands={
            "urr": SyntheticUrrCommandCfg(
                resampling_time_range=(2.0, 2.0),
                scale=1.0,
            ),
        },
        events={
            "randomize_dynamics": EventTermCfg(
                mode="reset",
                func=randomize_dynamics,
            ),
        },
        rewards={
            "task_tracking": RewardTermCfg(func=command_tracking_reward, weight=4.0),
            "action_smoothness": RewardTermCfg(func=action_smoothness_reward, weight=0.05),
            "joint_pos_limits": RewardTermCfg(func=joint_limit_penalty, weight=2.0),
            "no_oscillation": RewardTermCfg(func=no_oscillation_penalty, weight=0.02),
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
        decimation=2,
        episode_length_s=1e9 if play else 10.0,
    )


def tiago_pp_arm_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Bare TIAGo++ arm reach configuration."""
    return _make_env_cfg(play=play, table=False)


def tiago_pp_arm_reach_table_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """TIAGo++ arm reach configuration with a table background."""
    return _make_env_cfg(play=play, table=True)
