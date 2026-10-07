from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from postman.envs import VectorizedManagerEnv, make
from postman.rl.policy import NumpyLinearPolicy, PolicyConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a POSTMAN smoke training loop.")
    parser.add_argument("--task", default="tiago_pp_bimanual_reach")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/postman_policy.npz"))
    parser.add_argument("--backend", choices=("numpy", "rsl_rl", "auto"), default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.backend == "rsl_rl":
        from postman.rl.rsl_rl_adapter import RslRlAdapter

        RslRlAdapter()
        raise NotImplementedError("Configure the RSL-RL runner in postman.rl.rsl_rl_adapter")

    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.num_envs == 1:
        env = make(args.task, seed=args.seed)
    else:
        env = VectorizedManagerEnv(
            lambda seed: make(args.task, seed=seed), args.num_envs, args.seed
        )
    observation, _ = env.reset(seed=args.seed)
    observation_dim = observation.shape[-1]
    policy = NumpyLinearPolicy(PolicyConfig(observation_dim, env.action_dim, args.seed))
    rewards = []
    for _ in range(args.steps):
        action = policy.act(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(np.mean(reward)))
        if args.num_envs == 1 and (terminated or truncated):
            observation, _ = env.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    summary = {
        "task": args.task,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "mean_reward": float(np.mean(rewards)),
        "backend": "numpy_smoke",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
