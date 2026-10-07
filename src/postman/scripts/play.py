from __future__ import annotations

import argparse
import json
from pathlib import Path

from postman.envs import make
from postman.rl.policy import NumpyLinearPolicy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play a saved POSTMAN policy.")
    parser.add_argument("--task", default="tiago_pp_bimanual_reach")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args(argv)
    env = make(args.task, seed=0)
    policy = NumpyLinearPolicy.load(args.checkpoint)
    observation, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(args.steps):
        observation, reward, terminated, truncated, _ = env.step(policy.act(observation))
        total += reward
        if terminated or truncated:
            observation, _ = env.reset()
    print(json.dumps({"task": args.task, "steps": args.steps, "return": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
