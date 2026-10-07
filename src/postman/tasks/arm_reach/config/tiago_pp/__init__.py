"""Register TIAGo++ tasks following mjlab's robot config convention."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
    tiago_pp_arm_reach_env_cfg,
    tiago_pp_arm_reach_table_env_cfg,
)
from .rl_cfg import tiago_pp_ppo_runner_cfg

register_mjlab_task(
    task_id="Postman-TiagoPP-ArmReach",
    env_cfg=tiago_pp_arm_reach_env_cfg(),
    play_env_cfg=tiago_pp_arm_reach_env_cfg(play=True),
    rl_cfg=tiago_pp_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Postman-TiagoPP-ArmReach-SimpleTable",
    env_cfg=tiago_pp_arm_reach_table_env_cfg(),
    play_env_cfg=tiago_pp_arm_reach_table_env_cfg(play=True),
    rl_cfg=tiago_pp_ppo_runner_cfg(),
)

__all__ = [
    "tiago_pp_arm_reach_env_cfg",
    "tiago_pp_arm_reach_table_env_cfg",
    "tiago_pp_ppo_runner_cfg",
]
