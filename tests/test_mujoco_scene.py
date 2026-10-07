from pathlib import Path

import mujoco

from postman.sim import FurniturePlacement, compose_tiago_scene


def test_tiago_furniture_composition():
    root = Path(__file__).parents[1]
    table = root / "third_party/scenes/furniture_sim/simpleTable.xml"
    model = compose_tiago_scene([FurniturePlacement(table)])
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert model.nu == 25
