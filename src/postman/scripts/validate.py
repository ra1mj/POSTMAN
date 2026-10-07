from __future__ import annotations

import argparse
import json

from postman.envs import TASK_REGISTRY
from postman.robots import TiagoPPScene
from postman.sim import check_mjlab


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate POSTMAN optional runtime dependencies and assets."
    )
    parser.add_argument("--require-sim", action="store_true")
    args = parser.parse_args(argv)
    status = check_mjlab()
    try:
        asset = str(TiagoPPScene().resolve_mjcf())
    except FileNotFoundError:
        asset = None
    report = {"tasks": TASK_REGISTRY.names(), "mjlab": status.installed, "tiago_pp_mjcf": asset}
    print(json.dumps(report, indent=2))
    if args.require_sim and (not status.installed or asset is None):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
