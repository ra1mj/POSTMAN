"""Small fallback policy used for smoke tests and dependency validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PolicyConfig:
    observation_dim: int
    action_dim: int
    seed: int = 0


class NumpyLinearPolicy:
    def __init__(self, config: PolicyConfig, weights=None, bias=None):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.weights = (
            np.asarray(weights)
            if weights is not None
            else rng.normal(0, 0.01, (config.action_dim, config.observation_dim))
        )
        self.bias = np.asarray(bias) if bias is not None else np.zeros(config.action_dim)
        if self.weights.shape != (config.action_dim, config.observation_dim):
            raise ValueError("policy weights have the wrong shape")
        if self.bias.shape != (config.action_dim,):
            raise ValueError("policy bias has the wrong shape")

    def act(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.ndim == 1:
            if obs.size != self.config.observation_dim:
                raise ValueError(
                    f"expected observation dim {self.config.observation_dim}, got {obs.size}"
                )
            return np.tanh(self.weights @ obs + self.bias)
        if obs.ndim == 2 and obs.shape[1] == self.config.observation_dim:
            return np.tanh(obs @ self.weights.T + self.bias)
        raise ValueError(
            f"expected observation shape (*, {self.config.observation_dim}), got {obs.shape}"
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            observation_dim=self.config.observation_dim,
            action_dim=self.config.action_dim,
        )

    @classmethod
    def load(cls, path: str | Path) -> NumpyLinearPolicy:
        data = np.load(path)
        config = PolicyConfig(int(data["observation_dim"]), int(data["action_dim"]))
        return cls(config, data["weights"], data["bias"])
