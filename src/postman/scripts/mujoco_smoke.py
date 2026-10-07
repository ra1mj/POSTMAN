"""Load and step a real MuJoCo Menagerie scene without opening a viewer."""

from __future__ import annotations

import argparse
import json

import numpy as np

from postman.robots import TiagoPPScene


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a headless MuJoCo scene smoke test.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--asset", default="pal_tiago_dual")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Install postman-rl-wbc[sim] first") from exc

    if args.asset == "pal_tiago_dual":
        xml_path = TiagoPPScene().resolve_mjcf()
    else:
        raise ValueError(f"unsupported asset {args.asset!r}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.qpos0.size:
        data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    for _ in range(args.steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    report = {
        "asset": args.asset,
        "xml": str(xml_path),
        "steps": args.steps,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "finite_qpos": bool(np.isfinite(data.qpos).all()),
        "finite_qvel": bool(np.isfinite(data.qvel).all()),
        "max_abs_qvel": float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["finite_qpos"] and report["finite_qvel"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
