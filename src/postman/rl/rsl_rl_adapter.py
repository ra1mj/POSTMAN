"""Optional RSL-RL integration point."""

from __future__ import annotations

import numpy as np


class RslRlAdapter:
    def __init__(self, *args, **kwargs):
        try:
            import rsl_rl  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[rl] to enable RSL-RL training") from exc
        self.args = args
        self.kwargs = kwargs

    def build_runner(self, env, train_cfg):
        """Construct the upstream ``OnPolicyRunner`` around a VecEnv adapter."""
        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[rl] to enable RSL-RL training") from exc
        wrapped = RslRlVecEnvAdapter(env, device=self.kwargs.get("device", "cpu"))
        return OnPolicyRunner(
            env=wrapped,
            train_cfg=train_cfg,
            log_dir=self.kwargs.get("log_dir"),
            device=self.kwargs.get("device", "cpu"),
        )

    def build_mjlab_runner(self, env, train_cfg):
        """Use mjlab's native vector wrapper and runner for GPU environments.

        ``env`` must already be a constructed mjlab ManagerBasedRlEnv. Keeping
        this method separate prevents the lightweight CPU reference environment
        from importing mjlab or Warp.
        """
        try:
            from mjlab.rl.runner import MjlabOnPolicyRunner
            from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[sim,rl] for the mjlab runner") from exc
        wrapped = RslRlVecEnvWrapper(env)
        return MjlabOnPolicyRunner(
            env=wrapped,
            train_cfg=train_cfg,
            log_dir=self.kwargs.get("log_dir"),
            device=self.kwargs.get("device", "cuda:0"),
        )


class RslRlVecEnvAdapter:
    """Single-environment VecEnv facade for the RSL-RL runner.

    mjlab users should prefer its native vectorized wrapper. This adapter is
    useful for validating the POSTMAN manager lifecycle and for small CPU
    experiments; it intentionally keeps the same observation/action contract.
    """

    def __init__(self, env, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[rl] to use the RSL-RL adapter") from exc
        self.torch = torch
        self.env = env
        self.device = device
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.num_actions = env.action_dim
        self.max_episode_length = int(getattr(env, "max_steps", env.max_episode_length))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.cfg = getattr(env, "cfg", {})
        self._obs, _ = env.reset()

    def get_observations(self):
        return self.torch.as_tensor(
            self._obs, dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)

    def step(self, actions):
        action = actions.detach().to("cpu").numpy()
        if self.num_envs == 1:
            action = action.reshape(-1)
        else:
            action = action.reshape(self.num_envs, self.num_actions)
        self._obs, reward, terminated, truncated, extras = self.env.step(action)
        self.episode_length_buf += 1
        done = np.asarray(terminated) | np.asarray(truncated)
        if self.num_envs == 1 and bool(done):
            self._obs, _ = self.env.reset()
            self.episode_length_buf.zero_()
        obs = self.get_observations()
        rewards = self.torch.as_tensor(
            reward, dtype=self.torch.float32, device=self.device
        ).reshape(self.num_envs)
        dones = self.torch.as_tensor(done, dtype=self.torch.bool, device=self.device).reshape(
            self.num_envs
        )
        return obs, rewards, dones, extras

    def get_privileged_observations(self):
        return None
