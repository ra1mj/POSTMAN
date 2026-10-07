from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from postman.sim.scene_composer import FurniturePlacement, compose_tiago_scene


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compose TIAGo++ with MuJoCo furniture assets.")
    parser.add_argument("--furniture", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)
    placements = [
        FurniturePlacement(path=path, position=(1.0 + 0.7 * index, 0.0, 0.0), prefix=f"f{index}_")
        for index, path in enumerate(args.furniture)
    ]
    model = compose_tiago_scene(placements, output=args.output)
    import mujoco

    data = mujoco.MjData(model)
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    report = {
        "furniture": [str(item.path) for item in placements],
        "output": str(args.output) if args.output else None,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
