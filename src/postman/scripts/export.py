from __future__ import annotations

import argparse
import json
from pathlib import Path

from postman.rl.policy import NumpyLinearPolicy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export a POSTMAN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("npz", "onnx"), default="npz")
    args = parser.parse_args(argv)
    policy = NumpyLinearPolicy.load(args.checkpoint)
    if args.format == "onnx":
        raise RuntimeError("ONNX export requires the optional torch/onnx backend")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "observation_dim": policy.config.observation_dim,
                "action_dim": policy.config.action_dim,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
