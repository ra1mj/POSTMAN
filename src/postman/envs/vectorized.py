"""CPU reference vectorization used by smoke tests and RSL-RL adapter checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


class VectorizedManagerEnv:
    """Batch independent manager environments behind a VecEnv-like API.

    mjlab/MuJoCo Warp remains the production backend. This class gives the
    same task implementation a deterministic CPU reference path and makes
    action/observation shapes explicit before moving to GPU vectorization.
    """

    def __init__(self, env_factory: Callable[[int], object], num_envs: int, seed: int = 0):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.envs = [env_factory(seed + index) for index in range(num_envs)]
        self.num_envs = num_envs
        self.num_actions = self.envs[0].action_dim
        self.max_episode_length = self.envs[0].max_steps
        self.device = "cpu"
        self._observations = np.zeros((num_envs, 0), dtype=np.float64)

    @property
    def action_dim(self) -> int:
        return self.num_actions

    def reset(self, seed: int | None = None):
        observations = []
        infos = []
        for index, env in enumerate(self.envs):
            obs, info = env.reset(seed=None if seed is None else seed + index)
            observations.append(obs)
            infos.append(info)
        self._observations = np.stack(observations)
        return self._observations.copy(), infos

    def get_observations(self) -> np.ndarray:
        return self._observations.copy()

    def step(self, actions: Sequence[Sequence[float]]):
        batch = np.asarray(actions, dtype=np.float64)
        if batch.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"expected actions {(self.num_envs, self.num_actions)}, got {batch.shape}"
            )
        observations, rewards, terminated, truncated, infos = [], [], [], [], []
        for index, (env, action) in enumerate(zip(self.envs, batch)):
            result = env.step(action)
            obs, reward, done, timeout, info = result
            if done or timeout:
                obs, reset_info = env.reset()
                info = {**info, "reset_info": reset_info}
            observations.append(obs)
            rewards.append(reward)
            terminated.append(done)
            truncated.append(timeout)
            infos.append(info)
        self._observations = np.stack(observations)
        return (
            self._observations.copy(),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            infos,
        )
