"""Deterministic, dependency-light TIAGo++ research tasks.

The environment is deliberately separated from MuJoCo so unit tests can run
without a GPU. A future mjlab adapter can replace ``_physics_step`` while
retaining the same managers, commands, rewards and WBC interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from postman.envs.base import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from postman.representations import RobotState, WholeBodyCommand
from postman.robots import TIAGO_PP
from postman.wbc import KinematicQpWbc, KinematicQpWbcConfig


@dataclass
class TiagoPPEnvCfg(ManagerBasedRlEnvCfg):
    dof: int = TIAGO_PP.dof
    target_radius: float = 0.08
    command_noise: float = 0.15
    action_dim: int = 24


class TiagoPPBimanualReachEnv(ManagerBasedRlEnv):
    @classmethod
    def default_cfg(cls) -> TiagoPPEnvCfg:
        return TiagoPPEnvCfg()

    def __init__(self, cfg: TiagoPPEnvCfg | None = None, seed: int | None = None):
        self.cfg = cfg or self.default_cfg()
        super().__init__(self.cfg, seed=seed)
        self.q = np.asarray(TIAGO_PP.default_q, dtype=np.float64)
        self.qd = np.zeros(TIAGO_PP.dof, dtype=np.float64)
        self.target = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self.command = WholeBodyCommand.from_vector(self.target)
        self.last_wbc = None
        self.controller = KinematicQpWbc(KinematicQpWbcConfig(dof=TIAGO_PP.dof, dt=self.cfg.dt))
        self._install_terms()

    def _install_terms(self) -> None:
        self.observation_manager.add("q", lambda env: env.q)
        self.observation_manager.add("qd", lambda env: env.qd)
        self.observation_manager.add("command", lambda env: env.target)
        self.reward_manager.add(
            "target_tracking", lambda env: -float(np.linalg.norm(env.q[:24] - env.target))
        )
        self.reward_manager.add(
            "action_smoothness",
            lambda env: -0.01 * float(np.linalg.norm(env.action_manager.last_action)),
        )
        self.termination_manager.add(
            "solver_failure",
            lambda env: (
                env.last_wbc is not None and env.last_wbc.solver_status == "singular_fallback"
            ),
        )
        self.termination_manager.add("time_limit", lambda env: False)
        self.metrics_manager.add(
            "tracking_error", lambda env: float(np.linalg.norm(env.q[:24] - env.target))
        )

    def _reset_state(self, options):
        del options
        self.q = np.asarray(TIAGO_PP.default_q, dtype=np.float64)
        self.qd.fill(0.0)
        self.target = self.rng.normal(0.0, self.cfg.command_noise, self.cfg.action_dim)
        self.command = WholeBodyCommand.from_vector(self.target)
        self.controller.reset()
        self.last_wbc = None

    def _apply_action(self, action: np.ndarray) -> None:
        self.command = WholeBodyCommand.from_vector(self.target + 0.25 * action)
        state = RobotState(self.q, self.qd)
        self.last_wbc = self.controller.compute(state, self.command)

    def _physics_step(self, dt: float) -> None:
        if self.last_wbc is None:
            return
        acceleration = 30.0 * (self.last_wbc.q_des - self.q) - 2.0 * self.qd
        self.qd += dt * acceleration
        self.q += dt * self.qd
        self.qd = np.clip(self.qd, -5.0, 5.0)


class TiagoPPBaseArmCoordinationEnv(TiagoPPBimanualReachEnv):
    """Reach task with a stronger base-motion component in the command."""

    def _reset_state(self, options):
        super()._reset_state(options)
        self.target[:3] *= 1.5
        self.command = WholeBodyCommand.from_vector(self.target)


class TiagoPPBimanualTransportEnv(TiagoPPBimanualReachEnv):
    """Bimanual transport placeholder sharing the v1 manager/WBC pipeline."""

    def _install_terms(self) -> None:
        super()._install_terms()
        self.reward_manager.add(
            "bimanual_sync", lambda env: -0.05 * float(np.linalg.norm(env.q[4:11] - env.q[11:18]))
        )
