"""Small manager-based environment core compatible with mjlab-style terms."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

Term = Callable[["ManagerBasedRlEnv"], Any]


class TermManager:
    def __init__(self, terms: Mapping[str, Term] | None = None):
        self.terms = dict(terms or {})

    def add(self, name: str, term: Term) -> None:
        if name in self.terms:
            raise KeyError(f"term {name!r} already exists")
        self.terms[name] = term

    def compute(self, env: ManagerBasedRlEnv) -> dict[str, Any]:
        return {name: term(env) for name, term in self.terms.items()}


class ObservationManager(TermManager):
    def compute_vector(self, env: ManagerBasedRlEnv) -> np.ndarray:
        values = self.compute(env)
        if not values:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(
            [np.asarray(value, dtype=np.float64).reshape(-1) for value in values.values()]
        )


class ActionManager:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.last_action = np.zeros(action_dim, dtype=np.float64)

    def process_action(self, action: Any) -> np.ndarray:
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.size != self.action_dim:
            raise ValueError(f"expected action dim {self.action_dim}, got {vector.size}")
        self.last_action = np.clip(vector, -1.0, 1.0)
        return self.last_action.copy()


@dataclass
class ManagerBasedRlEnvCfg:
    dt: float = 0.002
    decimation: int = 2
    episode_length_s: float = 20.0
    action_dim: int = 24
    observation_dim: int = 0


class ManagerBasedRlEnv:
    """Minimal Gymnasium-shaped environment with explicit manager lifecycle."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg, seed: int | None = None):
        if cfg.dt <= 0 or cfg.decimation <= 0:
            raise ValueError("dt and decimation must be positive")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.max_steps = max(1, int(cfg.episode_length_s / (cfg.dt * cfg.decimation)))
        self.observation_manager = ObservationManager()
        self.action_manager = ActionManager(cfg.action_dim)
        self.reward_manager = TermManager()
        self.termination_manager = TermManager()
        self.event_manager = TermManager()
        self.command_manager = TermManager()
        self.metrics_manager = TermManager()

    @property
    def action_dim(self) -> int:
        return self.action_manager.action_dim

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._reset_state(options or {})
        self.event_manager.compute(self)
        self.command_manager.compute(self)
        return self._compute_observation(), {"task": self.__class__.__name__}

    def step(self, action: Any):
        processed = self.action_manager.process_action(action)
        self._apply_action(processed)
        for _ in range(self.cfg.decimation):
            self._physics_step(self.cfg.dt)
        self.step_count += 1
        reward_terms = self.reward_manager.compute(self)
        reward = float(sum(float(value) for value in reward_terms.values()))
        terminated_terms = self.termination_manager.compute(self)
        terminated = bool(any(bool(value) for value in terminated_terms.values()))
        truncated = self.step_count >= self.max_steps
        self.command_manager.compute(self)
        observation = self._compute_observation()
        info = {
            "reward_terms": reward_terms,
            "termination_terms": terminated_terms,
            "metrics": self.metrics_manager.compute(self),
        }
        return observation, reward, terminated, truncated, info

    def _compute_observation(self) -> np.ndarray:
        return self.observation_manager.compute_vector(self)

    def _reset_state(self, options: Mapping[str, Any]) -> None:
        del options

    def _apply_action(self, action: np.ndarray) -> None:
        del action

    def _physics_step(self, dt: float) -> None:
        del dt
