import numpy as np
import pytest

pytest.importorskip("mujoco")


def test_robot_holds_on_floor_with_gravity_for_five_seconds():
    from postman.wbc.hand_follow import HandFollowController

    controller = HandFollowController()
    np.testing.assert_allclose(controller.model.opt.gravity, [0, 0, -9.81])
    floor = controller.model.geom("floor").id
    assert controller.data.geom_xpos[floor, 2] == pytest.approx(0.0)
    for _ in range(500):
        metrics = controller.step()
        assert abs(metrics["base_height_m"]) < 0.002
        assert metrics["base_tilt_rad"] < 0.01
        assert controller.data.ncon >= 4
    assert controller.data.time == pytest.approx(5.0)
    assert metrics["base_motion"] < 0.005
    assert max(metrics["left_hand_error"], metrics["right_hand_error"]) < 0.002


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_target_update_tracks_quickly_under_real_physics(side):
    from postman.wbc.hand_follow import HandFollowController

    controller = HandFollowController()
    start = controller.target[side]
    controller.set_active_hands((side,))
    target = start + np.array([0.12, 0.0, 0.08])
    # A single UI event must continue tracking; filtering cannot require a
    # stream of repeated set_target calls to reach the requested position.
    controller.set_target(side, target)
    for i in range(200):
        metrics = controller.step()
        if i == 99:
            assert metrics[f"{side}_hand_error"] < 0.025
        assert abs(metrics["base_height_m"]) < 0.005
        assert metrics["base_tilt_rad"] < 0.03
        assert controller.data.ncon >= 4

    assert np.isfinite(controller.data.qpos).all()
    assert np.isfinite(controller.data.qvel).all()
    assert metrics[f"{side}_hand_error"] < 0.015
    assert metrics["base_motion"] > 0.02
    assert metrics["torso_motion"] > 0.01
    assert controller.data.time == pytest.approx(2.0)
    limits = controller.model.actuator_ctrlrange
    assert np.all(controller.data.ctrl >= limits[:, 0])
    assert np.all(controller.data.ctrl <= limits[:, 1])


def test_reset_clears_servo_and_filtered_target_history():
    from postman.wbc.hand_follow import HandFollowController

    controller = HandFollowController()
    initial = controller.target["left"]
    controller.set_target("left", initial + [.2, 0, .1])
    for _ in range(50):
        controller.step()
    controller.reset()
    np.testing.assert_allclose(controller.target["left"], initial)
    assert controller.data.time == 0
    for _ in range(100):
        controller.step()
    assert controller.metrics()["left_hand_error"] < 0.002


def test_default_floor_fix_does_not_edit_cached_asset():
    import mujoco

    from postman.robots import TiagoPPScene
    from postman.wbc.hand_follow import HandFollowController

    original = mujoco.MjModel.from_xml_path(str(TiagoPPScene().resolve_mjcf()))
    controller = HandFollowController()
    assert controller.model.geom("floor").pos[2] == pytest.approx(0.0)
    assert original.geom("floor").pos[2] == pytest.approx(-0.99)
