"""Compose real MuJoCo furniture assets with the TIAGo++ robot via MjSpec."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from postman.robots import TiagoPPScene


@dataclass(frozen=True)
class FurniturePlacement:
    path: Path
    position: tuple[float, float, float] = (1.0, 0.0, 0.0)
    prefix: str = "furniture_"


def compose_tiago_scene(
    placements: Iterable[FurniturePlacement],
    *,
    robot_scene: Path | None = None,
    output: Path | None = None,
):
    """Attach furniture specs to a robot spec and compile a MuJoCo model.

    ``MjSpec.attach`` preserves each asset's local mesh paths and avoids the
    fragile nested-XML include behavior that breaks relative mesh directories
    when a Menagerie scene is included from another file.
    """

    spec = compose_tiago_spec(placements, robot_scene=robot_scene)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # ``MjSpec.to_file`` only accepts compiled models in MuJoCo 3.3; the
        # editable composition is serialized explicitly here. For a portable
        # asset bundle use ``spec.to_zip`` from the caller's pinned MuJoCo.
        output.write_text(spec.to_xml(), encoding="utf-8")
    return spec.compile()


def compose_tiago_spec(
    placements: Iterable[FurniturePlacement],
    *,
    robot_scene: Path | None = None,
):
    """Return an editable MjSpec for a TIAGo++ plus furniture composition."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Install postman-rl-wbc[sim] to compose scenes") from exc

    robot_path = robot_scene or TiagoPPScene().resolve_robot_mjcf()
    spec = mujoco.MjSpec.from_file(str(robot_path))
    for index, placement in enumerate(placements):
        furniture = mujoco.MjSpec.from_file(str(placement.path))
        frame = spec.worldbody.add_frame(
            name=f"{placement.prefix}frame_{index}",
            pos=placement.position,
        )
        spec.attach(furniture, prefix=placement.prefix, frame=frame)
    return spec
