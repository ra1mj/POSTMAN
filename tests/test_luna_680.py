"""Luna axis mapping, actor and native-SDK teleoperation regressions."""

import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock
from urllib.request import Request, urlopen

import numpy as np
import pytest

pytest.importorskip("mujoco", minversion="3.8.1")

from postman.luna.control import LunaPolicy, anchor_target, decode, default_policy, tracking
from postman.luna.dds import Feedback, Limits
from postman.luna.model import (
    ACTION_SIZE,
    BASE_RATE,
    FX_TO_MODEL,
    GROUP_SLICES,
    GROUPS,
    HOME,
    JOINT_RATE,
    LunaModel,
    default_urdf,
)
from postman.luna.teleop import Teleop, sdk_sim_model


@pytest.fixture(scope="module")
def luna():
    urdf = default_urdf()
    if not urdf.is_file():
        pytest.skip("set POSTMAN_LUNA_URDF to the delivered Luna V3 680 model")
    return LunaModel(urdf)


def test_axis_contract_and_stationary_target(luna, tmp_path):
    assert luna.model.njnt == luna.model.nu == 22
    assert len(set(luna.qpos)) == 22
    q = HOME.copy()
    anchor = anchor_target(luna, q[None])
    state = tracking(luna, q[None], np.zeros((1, 22)), np.zeros((1, 3)), anchor)
    predicted = LunaPolicy(default_policy(), luna)(state.observation)
    assert predicted.shape == (1, ACTION_SIZE)
    twist, position = decode(state, predicted, q[None], luna.bounds, BASE_RATE)
    np.testing.assert_array_equal(twist, np.zeros((1, 3)))
    np.testing.assert_allclose(position[0], q, atol=1e-10)

    config = {"base_velocity_limits": [0.08, 0.08, 0.15], "joint_limits": {
        name: luna.bounds[section].tolist() for name, section in GROUP_SLICES.items()
    }}
    config["joint_limits"]["head"][1] = [-0.10, 0.30]
    file = tmp_path / "limits.json"
    file.write_text(json.dumps(config))
    limits = Limits.from_gateway(file)
    np.testing.assert_allclose(limits.position[-1], [-0.30, 0.10])
    q[-1] = 0.11
    with pytest.raises(ValueError, match="head axis 2 outside"):
        limits.check_position(q)


def test_sdk_feedback_schema_requires_valid_fx_and_fresh_motion():
    now = time.time_ns()
    hardware = {"model": 3, "stamp_ns": now, "frame_fresh": True, "frame_age_ms": 10,
                "frame_serial": 7, "capabilities": 8, "groups": {
                    name: {"dof": len(axes), "fault_code": 0, "mode": 1}
                    for name, axes in GROUPS.items()
                }}
    fx_pos = np.linspace(-0.20, 0.20, 22)
    snapshot = {"motion_stamp_ns": now, "base": [0.1, 0.2, 0.3], "controller": hardware,
                "command_endpoints": True, "command_deadline_supported": True, "joints": {
                    name: {"position": fx_pos[section].tolist()}
                    for name, section in GROUP_SLICES.items()
                }}
    parsed = Feedback.parse(snapshot, hardware=True, command=True)
    np.testing.assert_allclose(parsed.q, fx_pos * FX_TO_MODEL)
    hardware["groups"]["waist"]["fault_code"] = 53
    with pytest.raises(ValueError, match="active FX fault for waist"):
        Feedback.parse(snapshot, hardware=True, command=True)
    snapshot["motion_stamp_ns"] = now - 900_000_000
    with pytest.raises(ValueError, match="MotionState"):
        Feedback.parse(snapshot, hardware=True, command=True)


def test_deadman_holds_joints_before_base_stop_fences_preparation(luna):
    bridge = Mock()
    limits = Limits(luna.bounds, JOINT_RATE, BASE_RATE)
    runtime = Teleop(bridge, luna, None, limits, None, "sim", ("waist",), 5.)
    frame = Feedback(time.time_ns(), HOME, np.zeros(3), {})
    runtime.prepared = ["waist"]
    runtime.active = runtime.base_owned = runtime.position_confirmed = True
    runtime.pause(frame)
    sent = [call.args[0] for call in bridge.send.call_args_list]
    assert sent == [[{"op": "joint", "group": "waist", "positions": HOME[:6].tolist()}],
                    [{"op": "stop"}]]
    assert not runtime.active and not runtime.prepared and not runtime.position_confirmed

    bridge.reset_mock()
    bridge.send.side_effect = [RuntimeError("joint hold rejected"), None]
    runtime.prepared = ["waist"]
    runtime.active = runtime.base_owned = True
    with pytest.raises(RuntimeError, match="joint hold rejected"):
        runtime.pause(frame)
    assert bridge.send.call_args.args[0] == [{"op": "stop"}]
    assert runtime.prepared == ["waist"]  # The error path can still stop this group.
    assert not runtime.base_owned


def isolated_dds(domain):
    dds = ('<CycloneDDS><Domain Id="any"><General><Interfaces>'
           '<NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast>'
           '</General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
           '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    # Keep the inherited workstation FX settings out of the isolated test.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GENTO_")}
    return {**env, "GENTO_DDS_DOMAIN": str(domain), "CYCLONEDDS_URI": dds,
            "OPENBLAS_NUM_THREADS": "1"}


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def operator_status(url, predicate, timeout):
    end = time.monotonic() + timeout
    state = None
    while time.monotonic() < end:
        try:
            with urlopen(f"{url}/status", timeout=0.4) as response:
                state = json.load(response)
        except OSError:
            time.sleep(0.04)
            continue
        if predicate(state):
            return state
        time.sleep(0.04)
    pytest.fail(f"missing expected SDK feedback, last status: {state}")


def jog_sender(url, values):
    with urlopen(url, timeout=0.4) as response:
        html = response.read().decode()
    token = re.search(r'const token="([^"]+)"', html)[1]
    sequence = int(re.search(r"sequence=(\d+);", html)[1])

    def send(index, enable=True):
        body = {"enable": enable, "values": values.tolist(), "sequence": sequence + index + 1}
        request = Request(f"{url}/jog", json.dumps(body).encode(), {"X-Postman-Session": token})
        with urlopen(request, timeout=0.4) as response:
            assert response.status == 204

    return send


@contextmanager
def running(args, env):
    with subprocess.Popen(args, env=env) as process:
        try:
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise


def test_operator_policy_native_dds_and_deadman(luna, tmp_path):
    build = os.environ.get("POSTMAN_GENTO_TEST_BUILD")
    if not build:
        pytest.skip("set POSTMAN_GENTO_TEST_BUILD to the Gento DDS and MuJoCo SDK build")
    sim = Path(build).resolve() / "gento_sdk_dds_sim_node"
    bridge = sim.with_name("gento_sdk_dds_ui_bridge")
    scene = tmp_path / "luna.xml"
    sdk_sim_model(luna, scene)
    port = free_port()
    domain = str(110 + os.getpid() % 60)
    env = isolated_dds(domain)
    url = f"http://127.0.0.1:{port}"
    sim_args = [str(sim), "--model", "luna", "--scope", "luna_sim", "--mjcf", str(scene)]
    app_args = [sys.executable, "-m", "postman.luna.teleop", "sim", "--domain", domain,
                "--bridge", str(bridge), "--port", str(port), "--urdf", str(luna.urdf)]
    with running(sim_args, env), running(app_args, env):
        initial = operator_status(url, lambda s: "fx_position_rad" in s, 8)
        values = np.zeros(17)
        values[[0, 6, 12, 16]] = [0.42, 0.30, 0.38, 0.40]
        refresh = jog_sender(url, values)
        for index in range(82):
            refresh(index)
            time.sleep(0.04)
        moving = operator_status(url, lambda s: max(s.get("hand_error_m", [1.])) < 0.02, 0.20)
        for section in GROUP_SLICES.values():
            displacement = (np.asarray(moving["fx_position_rad"])[section]
                            - np.asarray(initial["fx_position_rad"])[section])
            assert np.max(np.abs(displacement)) > 0.015
        assert moving["base_measured"][0] > 0.04
        # Stop renewing the lease, simulating a dropped input connection.
        stopped = operator_status(url, lambda s: "hand_error_m" not in s and
                                  np.max(np.abs(s.get("base_measured", [1.]))) < 0.006, 2.5)
        for index in range(82, 90):
            refresh(index)
            time.sleep(0.04)
        held = operator_status(url, lambda s: "hand_error_m" not in s and
                               np.max(np.abs(s.get("base_measured", [1.]))) < 0.006, 0.20)
        np.testing.assert_allclose(held["fx_position_rad"], stopped["fx_position_rad"], atol=0.008)


def test_live_policy_through_native_hardware_gateway(luna, tmp_path):
    sdk_build = os.environ.get("POSTMAN_GENTO_TEST_BUILD")
    if not sdk_build:
        pytest.skip("set POSTMAN_GENTO_TEST_BUILD to the Gento SDK build")
    build = Path(sdk_build).resolve()
    gateway = build / "gento_sdk_dds_hardware_node"
    stub = build / "libgento_sdk_fx_test_stub.so"
    if not gateway.is_file() or not stub.is_file():
        pytest.skip("enable the native Gento hardware node and build gento_sdk_fx_test_stub")

    cache = (build / "CMakeCache.txt").read_text()
    cyclone = Path(re.search(r"^CycloneDDS_DIR:PATH=(.+)$", cache, re.M)[1]).parent.parent
    compiler = re.search(r"^CMAKE_CXX_COMPILER:FILEPATH=(.+)$", cache, re.M)[1]
    pkg_path = os.pathsep.join(x for x in (str(cyclone / "pkgconfig"),
                                         os.environ.get("PKG_CONFIG_PATH")) if x)
    flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", "CycloneDDS"],
                                    env={**os.environ, "PKG_CONFIG_PATH": pkg_path}, text=True)
    fixture = tmp_path / "rdk_loopback"
    fixture_src = Path(__file__).parent / "fixtures" / "luna_rdk_loopback.cpp"
    subprocess.run([compiler, "-std=c++20", f"-I{build}", str(fixture_src),
                    str(build / "libgento_sdk_apex_dds.a"), *shlex.split(flags),
                    f"-Wl,-rpath,{cyclone}", "-o", str(fixture)], check=True, timeout=20)

    domain = str(50 + os.getpid() % 40)
    base_domain = str(int(domain) + 45)
    env = isolated_dds(domain)
    trace = tmp_path / "fx_trace.txt"
    env.update(GENTO_FX_STUB_MODEL="luna", GENTO_FX_STUB_FOLLOW_POSITIONS="1",
               GENTO_FX_STUB_TRACE_PATH=str(trace))
    limits = tmp_path / "limits.json"
    bounds = np.sort(luna.bounds * FX_TO_MODEL[:, None], axis=1)
    limits.write_text(json.dumps({"base_velocity_limits": [0.12, 0.0, 0.0], "joint_limits": {
        name: bounds[section].tolist() for name, section in GROUP_SLICES.items()
    }}))
    bridge = build / "gento_sdk_dds_ui_bridge"
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    gateway_args = [str(gateway), "--controller-ip", "127.0.0.1", "--fx-library", str(stub),
                    "--scope", "luna_control", "--expect-model", "luna", "--limits", str(limits),
                    "--base-domain", base_domain]
    app_args = [sys.executable, "-m", "postman.luna.teleop", "live", "--domain", domain,
                "--bridge", str(bridge), "--port", str(port), "--urdf", str(luna.urdf),
                "--limits", str(limits), "--groups", "head", "--speed-percent", "1"]
    probe_args = [sys.executable, "-m", "postman.luna.teleop", "probe", "--domain", domain,
                  "--scope", "luna_control", "--bridge", str(bridge), "--limits", str(limits)]
    with running([str(fixture), base_domain], env), running(gateway_args, env), \
            running(app_args, env):
        report = json.loads(subprocess.check_output(probe_args, env=env, timeout=8))
        assert report["joint_axes"] == 22 and report["joint_limits_ok"] is True
        assert report["base_stationary"] and report["command_endpoints"]
        assert report["command_deadline_supported"] and report["group_stop_supported"]
        assert report["base_velocity_caps"] == [0.12, 0.0, 0.0]
        assert "prepare" not in trace.read_text() and "position" not in trace.read_text()
        initial = operator_status(url, lambda s: "fx_position_rad" in s, 8)
        values = np.zeros(17)
        values[[12, 16]] = [0.38, 0.1]
        refresh = jog_sender(url, values)
        for index in range(80):
            refresh(index)
            time.sleep(0.04)
        moving = operator_status(url, lambda s: s.get("base_measured", [0.])[0] > 0.04, 0.20)
        assert moving["mode"] == "live"
        assert moving["fx_position_rad"][-1] < initial["fx_position_rad"][-1] - 0.005
        np.testing.assert_allclose(moving["fx_position_rad"][:20], initial["fx_position_rad"][:20])
        events = {line.split()[0] for line in trace.read_text().splitlines()}
        assert {"prepare", "position"} <= events
        stopped = operator_status(url, lambda s: "base_target" not in s and
                                  np.max(np.abs(s.get("base_measured", [1.]))) < 0.006, 2.5)
        assert stopped["fx_position_rad"][-1] < initial["fx_position_rad"][-1] - 0.005
