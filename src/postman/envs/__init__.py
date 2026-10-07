from .base import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from .registry import TASK_REGISTRY, TaskRegistry, make
from .tiago_pp import (
    TiagoPPBaseArmCoordinationEnv,
    TiagoPPBimanualReachEnv,
    TiagoPPBimanualTransportEnv,
)
from .vectorized import VectorizedManagerEnv

TASK_REGISTRY.register(
    "tiago_pp_bimanual_reach", TiagoPPBimanualReachEnv, TiagoPPBimanualReachEnv.default_cfg()
)
TASK_REGISTRY.register(
    "tiago_pp_base_arm_coordination",
    TiagoPPBaseArmCoordinationEnv,
    TiagoPPBaseArmCoordinationEnv.default_cfg(),
)
TASK_REGISTRY.register(
    "tiago_pp_bimanual_transport",
    TiagoPPBimanualTransportEnv,
    TiagoPPBimanualTransportEnv.default_cfg(),
)

__all__ = [
    "ManagerBasedRlEnv",
    "ManagerBasedRlEnvCfg",
    "TASK_REGISTRY",
    "TaskRegistry",
    "VectorizedManagerEnv",
    "make",
]
