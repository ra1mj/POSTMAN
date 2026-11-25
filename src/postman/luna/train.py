"""Train and evaluate the Luna 680 residual WBC actor with MuJoCo and RSL-RL PPO."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mujoco.rollout import Rollout
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from .control import (
    HEAD_OFFSET,
    POLICY_VERSION,
    POSITION_OFFSET,
    ROTATION_OFFSET,
    LunaPolicy,
    Target,
    anchor_target,
    decode,
    offset_target,
    tracking,
)
from .model import ACTION_SIZE, BASE_RATE, DT, HOME, JOINTS, LunaModel, default_urdf


class LunaReach(VecEnv):
    """Parallel Luna arm/waist/head physics and a velocity-controlled RDK chassis model."""

    num_actions = ACTION_SIZE
    max_episode_length = 100
    device = "cpu"

    def __init__(self, model: LunaModel, count: int, seed: int, workers: int):
        self.model = model
        self.num_envs = count
        self.cfg = {"model": "Luna V3 680", "urdf_sha256": model.fingerprint, "dt": DT,
                    "seed": seed}
        self.rng = np.random.default_rng(seed)
        self.episode_length_buf = torch.zeros(count, dtype=torch.long)
        self.substeps = round(DT / model.model.opt.timestep)
        self.rollout = Rollout(nthread=workers)
        self.worker_data = [mujoco.MjData(model.model) for _ in range(workers)]
        # FX gain, damping and gravity-compensation mismatch across environments.
        variants = [copy.copy(model.model) for _ in range(8)]
        for variant in variants:
            gain = self.rng.uniform(0.8, 1.2, len(JOINTS))
            variant.actuator_gainprm[:, 0] *= gain
            variant.actuator_biasprm[:, 1] *= gain
            variant.actuator_biasprm[:, 2] *= self.rng.uniform(0.75, 1.25, len(JOINTS))
            variant.body_gravcomp[1:] = self.rng.uniform(0.97, 1.0)
        self.models = [variants[i % len(variants)] for i in range(count)]
        self.physics = np.zeros((count, 1 + model.model.nq + model.model.nv))
        self.sim_output = np.empty((count, self.substeps, self.physics.shape[1]))
        self.anchors = Target(np.zeros((count, 2, 3)), np.zeros((count, 2, 3, 3)),
                              np.zeros((count, 3)), np.zeros((count, 2)))
        self.offsets = self.anchors.hands.copy()
        self.orientations = self.offsets.copy()
        self.base_goal = self.anchors.base.copy()
        self.base_velocity = self.base_goal.copy()
        self.head_goal = self.anchors.head.copy()
        self.previous_action = np.zeros((count, ACTION_SIZE))
        self.delay = np.zeros(count, dtype=bool)
        self.previous_target = np.zeros((count, len(JOINTS)))
        self.previous_base = self.base_velocity.copy()
        self.reset(np.arange(count))
        self.update_tracking()

    @property
    def q(self) -> np.ndarray:
        return self.physics[:, 1 + self.model.qpos]

    @property
    def qd(self) -> np.ndarray:
        return self.physics[:, 1 + self.model.model.nq + self.model.qvel]

    def reset(self, ids: np.ndarray) -> None:
        n = len(ids)
        flex = self.rng.uniform(-0.30, 0.72, (n, 1))
        q = HOME + self.rng.uniform(-0.30, 0.30, (n, len(JOINTS)))
        q[:, :6] = HOME[:6] + self.rng.uniform(-0.10, 0.10, (n, 6))
        q[:, 1:4] += flex * np.array([1., -1.7, 0.7])
        q = np.clip(q, self.model.bounds[:, 0] + 0.03, self.model.bounds[:, 1] - 0.03)
        self.physics[ids] = 0.
        self.physics[ids[:, None], 1 + self.model.qpos] = q
        anchor = anchor_target(self.model, q)
        self.anchors.hands[ids] = anchor.hands
        self.anchors.rotations[ids] = anchor.rotations
        self.anchors.head[ids] = anchor.head
        # Mix single-hand, bimanual and stationary-hand trials.
        hands = self.rng.random((n, 2, 1)) < 0.78
        self.offsets[ids] = self.rng.uniform(-1., 1., (n, 2, 3)) * POSITION_OFFSET * hands
        self.orientations[ids] = self.rng.uniform(-1., 1., (n, 2, 3)) * ROTATION_OFFSET * hands
        moving = self.rng.random((n, 1)) < 0.55
        self.base_goal[ids] = self.rng.uniform(-0.82, 0.82, (n, 3)) * BASE_RATE * moving
        self.head_goal[ids] = self.rng.uniform(-HEAD_OFFSET, HEAD_OFFSET, (n, 2))
        self.base_velocity[ids] = 0.
        self.previous_action[ids] = 0.
        self.previous_target[ids] = q
        self.previous_base[ids] = 0.
        self.delay[ids] = self.rng.random(n) < 0.33
        self.episode_length_buf[ids] = 0

    def update_tracking(self) -> None:
        phase = np.clip(self.episode_length_buf.numpy() / 18., 0., 1.)[:, None]
        blend = phase * phase * (3. - 2. * phase)
        goal = offset_target(self.anchors, self.offsets * blend[:, None],
                             self.orientations * blend[:, None], self.base_goal * blend,
                             self.head_goal * blend)
        self.goal = goal
        self.control = tracking(self.model, self.q, self.qd, self.base_velocity, goal)

    def get_observations(self) -> TensorDict:
        return TensorDict({"state": torch.from_numpy(self.control.observation)},
                          batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        action = actions.cpu().numpy().clip(-1., 1.)
        twist, target = decode(self.control, action, self.q, self.model.bounds, BASE_RATE)
        applied = np.where(self.delay[:, None], self.previous_target, target)
        self.rollout.rollout(self.models, self.worker_data, self.physics,
                             np.repeat(applied[:, None], self.substeps, axis=1),
                             state=self.sim_output)
        self.physics[:] = self.sim_output[:, -1]
        applied_base = np.where(self.delay[:, None], self.previous_base, twist)
        self.base_velocity += 0.36 * (0.93 * applied_base - self.base_velocity)
        self.previous_target = target
        self.previous_base = twist
        self.episode_length_buf += 1
        self.update_tracking()
        distance = np.linalg.norm(self.control.position_error, axis=-1).mean(axis=1)
        angle = np.linalg.norm(self.control.orientation_error, axis=-1).mean(axis=1)
        base_error = np.abs((self.goal.base - self.base_velocity) / BASE_RATE).mean(axis=1)
        head_error = np.abs(self.control.head_error).mean(axis=1)
        clearance = np.minimum(self.q - self.model.bounds[:, 0], self.model.bounds[:, 1] - self.q)
        motion = np.square(action - self.previous_action).mean(axis=1)
        # Smooth position and angle rewards preserve gradient as the pose settles.
        reward = (2.0 * np.exp(-10 * distance) + 0.5 * np.exp(-4 * angle)
                  + 0.35 * np.exp(-4 * base_error) + 0.25 * np.exp(-8 * head_error)
                  - 0.04 * motion - 0.08 * np.square(action).mean(axis=1)
                  - 0.5 * (clearance < 0.02).mean(axis=1))
        self.previous_action = action
        done = self.episode_length_buf >= self.max_episode_length
        self.metrics = {"position_m": distance, "rotation_rad": angle,
                        "head_rad": head_error, "base_normalized": base_error}
        if done.any():
            self.reset(np.flatnonzero(done.numpy()))
            self.update_tracking()
        return self.get_observations(), torch.from_numpy(reward.astype(np.float32)), done, {
            "time_outs": done, "log": {name: value.mean() for name, value in self.metrics.items()},
        }


def evaluate(model, actor, count: int, seed: int, workers: int) -> dict[str, float]:
    env = LunaReach(model, count, seed, workers)
    totals: dict[str, list[float]] = {}
    try:
        for i in range(env.max_episode_length):
            action = actor(env.control.observation) if actor else np.zeros((count, ACTION_SIZE))
            env.step(torch.from_numpy(action))
            if i >= 59:
                for name, scores in env.metrics.items():
                    totals.setdefault(name, []).extend(scores.tolist())
        return {name: float(np.mean(scores)) for name, scores in totals.items()}
    finally:
        env.rollout.close()


def export(actor, env: LunaReach, output: Path) -> None:
    data = {"format": POLICY_VERSION, "urdf_sha256": env.model.fingerprint, "joints": JOINTS}
    layers = [layer for layer in actor.mlp.modules() if isinstance(layer, torch.nn.Linear)]
    for index, layer in enumerate(layers):
        data[f"weight_{index}"] = layer.weight.detach().cpu().numpy()
        data[f"bias_{index}"] = layer.bias.detach().cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **data)
    with torch.inference_mode():
        expected = actor(env.get_observations().to(next(actor.parameters()).device)).cpu().numpy()
    actual = LunaPolicy(output, env.model)(env.control.observation)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=default_urdf())
    parser.add_argument("--output", type=Path, default=Path("outputs/luna_680_policy.npz"))
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=680)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if min(args.num_envs, args.workers, args.iterations) < 1:
        parser.error("environment count, workers and iterations must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model = LunaModel(args.urdf)
    if not args.evaluate_only:
        env = LunaReach(model, args.num_envs, args.seed, args.workers)
        network = {"class_name": "MLPModel", "hidden_dims": (128, 96), "activation": "elu",
                   "obs_normalization": False}
        cfg = {
            "num_steps_per_env": 32, "save_interval": 50, "logger": "tensorboard",
            "obs_groups": {"actor": ["state"], "critic": ["state"]},
            "actor": {**network, "distribution_cfg": {"class_name": "GaussianDistribution",
                      "init_std": 0.30, "std_type": "log"}},
            "critic": network.copy(),
            "algorithm": {"class_name": "PPO", "num_learning_epochs": 4, "num_mini_batches": 8,
                          "learning_rate": 4e-4, "schedule": "adaptive", "desired_kl": 0.012,
                          "gamma": 0.99, "lam": 0.95, "entropy_coef": 0.001,
                          "value_loss_coef": 1., "clip_param": 0.2, "max_grad_norm": 1.,
                          "use_clipped_value_loss": True},
        }
        try:
            runner = OnPolicyRunner(env, cfg, str(args.output.with_suffix("")), args.device)
            runner.learn(args.iterations)
            export(runner.get_inference_policy(args.device), env, args.output)
        finally:
            env.rollout.close()
    policy = LunaPolicy(args.output, model)
    seed = args.seed + 1000
    stats = {"seed": seed, "trials": 128,
             "wbc": evaluate(model, None, 128, seed, args.workers),
             "ppo_wbc": evaluate(model, policy, 128, seed, args.workers)}
    print(json.dumps(stats, indent=2))
    args.output.with_suffix(".eval.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
