"""Browser TIAGo++ hand targets and live whole-body QP reference control.

The complete MuJoCo robot is rendered through mjviser. Drag the two XYZ gizmos
in the same browser scene; the simulator runs with the controller's configured
scene and gravity. This reference demo does not load a trained RL policy.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from postman.wbc.hand_follow import HandFollowController


class RuntimeTiming:
    """Measured simulation and browser publishing rates; independent of model time resets."""

    def __init__(self):
        self.started = self.sampled_at = time.monotonic()
        self.steps = self.renders = 0
        self.simulated_s = self.control_s = self.render_s = 0.0
        self._previous = (0, 0, 0.0)
        self.rates = {"control_hz": 0.0, "visual_hz": 0.0, "realtime_factor": 0.0}

    def record_step(self, elapsed_s, simulation_s):
        self.steps += 1
        self.control_s += elapsed_s
        self.simulated_s += simulation_s

    def record_render(self, elapsed_s):
        self.renders += 1
        self.render_s += elapsed_s

    def status(self, now):
        elapsed = now - self.sampled_at
        if elapsed >= 0.5:
            steps, renders, simulated = self._previous
            self.rates = {
                "control_hz": (self.steps - steps) / elapsed,
                "visual_hz": (self.renders - renders) / elapsed,
                "realtime_factor": (self.simulated_s - simulated) / elapsed,
            }
            self.sampled_at = now
            self._previous = (self.steps, self.renders, self.simulated_s)
        return {
            **self.rates,
            "control_step_ms": 1000.0 * self.control_s / max(self.steps, 1),
            "visual_update_ms": 1000.0 * self.render_s / max(self.renders, 1),
        }

    def summary(self):
        elapsed = max(time.monotonic() - self.started, 1e-9)
        return {
            "elapsed_s": elapsed,
            "simulated_s": self.simulated_s,
            "realtime_factor": self.simulated_s / elapsed,
            "control_hz": self.steps / elapsed,
            "visual_hz": self.renders / elapsed,
            "control_step_ms": 1000.0 * self.control_s / max(self.steps, 1),
            "visual_update_ms": 1000.0 * self.render_s / max(self.renders, 1),
        }


class BrowserHandFollowViewer:
    """Browser meshes, hand gizmos and a thread-safe UI-to-control handoff."""

    def __init__(self, controller, *, host="127.0.0.1", port=8080, hand="both"):
        try:
            import viser
            from mjviser import ViserMujocoScene
        except ImportError as exc:  # pragma: no cover - optional simulation extra
            raise RuntimeError("Install postman-rl-wbc[sim] for the browser viewer") from exc

        self.lock = threading.Lock()
        self.targets = controller.target
        self.hand = hand
        self.paused = False
        self.reset_requested = False
        self.server = viser.ViserServer(
            host=host, port=port, label="POSTMAN · Whole-body hand follow"
        )
        try:
            self.server.gui.configure_theme(
                control_layout="collapsible", control_width="medium",
                show_logo=False, show_share_button=False,
            )
            self.server.scene.set_up_direction("+z")
            self.scene = ViserMujocoScene(self.server, controller.model, num_envs=1)
            # Targets use world coordinates. Recentring the robot alone would
            # visually separate its meshes from their world-space hand gizmos.
            self.scene.camera_tracking_enabled = False
            self.floor_ids = set(np.flatnonzero(controller.model.geom_type == 0).tolist())
            if self.floor_ids:
                floor_position = controller.data.geom_xpos[min(self.floor_ids)].copy()
                self.server.scene.add_label(
                    "/physics/floor_label", "Collision floor",
                    position=floor_position + [-0.6, -0.65, 0.01],
                    font_screen_scale=0.65,
                )
            self.handles = {}
            self.error_lines = {}
            self._make_controls()
            for side, position in self.targets.items():
                handle = self.server.scene.add_transform_controls(
                    f"/targets/{side}", position=position, scale=0.16,
                    active_axes=(True, True, True), disable_rotations=True,
                    depth_test=False,
                )
                self.handles[side] = handle
                self.server.scene.add_label(
                    f"/targets/{side}/label", f"{side.title()} hand target",
                    position=(0.0, 0.0, 0.09), font_screen_scale=0.75,
                )
                self.error_lines[side] = self.server.scene.add_line_segments(
                    f"/errors/{side}", points=np.stack([position, position])[None],
                    colors=(240, 170, 50) if side == "left" else (60, 170, 230),
                    thickness=0.004,
                )

                @handle.on_update
                async def _on_target_update(event, side=side):
                    position = np.asarray(event.target.position, dtype=np.float64)
                    if position.shape == (3,) and np.isfinite(position).all():
                        with self.lock:
                            if self.hand in (side, "both"):
                                self.targets[side] = position.copy()

            @self.server.on_client_connect
            def _set_camera(client):
                client.camera.position = (2.4, -2.5, 1.8)
                client.camera.look_at = (0.2, 0.0, 0.85)
                client.camera.up_direction = (0.0, 0.0, 1.0)

            self._update_active_handles()
            self.update(controller, controller.metrics(), steps=0, update_status=True)
        except BaseException:
            self.server.stop()
            raise
        self.url = f"http://{host}:{self.server.get_port()}"
        print(f"Open {self.url} — full robot and both hand targets (QP reference).", flush=True)

    def _make_controls(self):
        self.server.gui.add_markdown(
            "**Whole-body hand follow · QP reference**\n\n"
            "Drag the X/Y/Z arrows at either hand. The base, torso and arms "
            "coordinate through the reference controller. No trained RL policy is loaded."
        )
        self.hand_selector = self.server.gui.add_dropdown(
            "Active hand", options=("both", "left", "right"), initial_value=self.hand,
        )
        pause = self.server.gui.add_checkbox("Pause simulation", initial_value=False)
        reset = self.server.gui.add_button("Reset robot and targets")
        self.status = self.server.gui.add_markdown("Loading robot…")

        @self.hand_selector.on_update
        def _choose_hand(event):
            with self.lock:
                self.hand = event.target.value
            self._update_active_handles()

        @pause.on_update
        def _pause(event):
            with self.lock:
                self.paused = bool(event.target.value)

        @reset.on_click
        def _reset(_event):
            # Physics state is changed only by the simulation thread.
            with self.lock:
                self.reset_requested = True

    def _update_active_handles(self):
        with self.lock:
            hand = self.hand
        for side, handle in self.handles.items():
            active = hand in (side, "both")
            handle.disable_axes = not active
            handle.disable_sliders = not active
            handle.opacity = 1.0 if active else 0.3

    def snapshot(self):
        with self.lock:
            result = (
                {side: target.copy() for side, target in self.targets.items()},
                self.hand, self.paused, self.reset_requested,
            )
            self.reset_requested = False
        return result

    def reset_targets(self, controller):
        self.set_targets(controller.target)

    def set_targets(self, targets):
        with self.lock:
            self.targets = {side: value.copy() for side, value in targets.items()}
            targets = {side: value.copy() for side, value in self.targets.items()}
        with self.server.atomic():
            for side, target in targets.items():
                self.handles[side].position = target

    def update(self, controller, metrics, *, steps, update_status=False, timing=None):
        targets, hand, paused, _ = self.snapshot_for_display()
        errors = {}
        with self.server.atomic():
            self.scene.update_from_mjdata(controller.data)
            for side, target in targets.items():
                body_id = controller.wbc.body_ids[f"{side}_wrist"]
                measured = controller.data.xpos[body_id].copy()
                errors[side] = float(np.linalg.norm(target - measured))
                self.error_lines[side].points = np.stack([target, measured])[None]
            if update_status:
                left = errors["left"] * 100.0
                right = errors["right"] * 100.0
                base = float(metrics.get("base_motion", 0.0)) * 100.0
                torso = float(metrics.get("torso_motion", 0.0)) * 100.0
                gravity = float(np.linalg.norm(controller.model.opt.gravity))
                floor_height = (
                    f"{controller.data.geom_xpos[min(self.floor_ids), 2]:.3f} m"
                    if self.floor_ids else "no plane in scene"
                )
                contacts = sum(
                    int(contact.geom1) in self.floor_ids
                    or int(contact.geom2) in self.floor_ids
                    for contact in controller.data.contact
                )
                rate = timing or {}
                self.status.content = (
                    f"**{'Paused' if paused else 'Running'}** · {steps} updates · {hand}\n\n"
                    "| Tracking | Error / motion |\n| :-- | --: |\n"
                    f"| Left hand error | {left:.2f} cm |\n"
                    f"| Right hand error | {right:.2f} cm |\n"
                    f"| Base displacement | {base:.2f} cm |\n"
                    f"| Torso displacement | {torso:.2f} cm |\n\n"
                    f"Speed **{rate.get('realtime_factor', 0):.2f}×** real time · "
                    f"control {rate.get('control_hz', 0):.0f}/100 Hz · "
                    f"visuals {rate.get('visual_hz', 0):.0f}/30 Hz\n\n"
                    f"Control {rate.get('control_step_ms', 0):.2f} ms/update · "
                    f"render {rate.get('visual_update_ms', 0):.2f} ms/update\n\n"
                    f"Gravity {gravity:.2f} m/s² · floor z {floor_height} · "
                    f"floor contacts {contacts}"
                )

    def snapshot_for_display(self):
        """Read UI state without consuming a pending reset request."""
        with self.lock:
            return (
                {side: target.copy() for side, target in self.targets.items()},
                self.hand, self.paused, self.reset_requested,
            )

    def close(self):
        self.server.stop()


def _select_hands(controller, hand):
    """Select support-motion contributors when the controller exposes the API."""
    if hasattr(controller, "set_active_hands"):
        controller.set_active_hands(("left", "right") if hand == "both" else (hand,))
    elif hasattr(controller, "enable_hand"):
        for side in controller.sides:
            controller.enable_hand(side, hand in (side, "both"))


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    if args.viewer == "none" and args.steps == 0:
        raise ValueError("headless runs require --steps greater than zero")
    offset = np.asarray(getattr(args, "offset", (0.0, 0.0, 0.0)), dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("offset must contain three finite values")
    controller = HandFollowController(xml_path=args.xml)
    _select_hands(controller, args.hand)
    targets = controller.target
    for side in controller.sides:
        if args.hand in (side, "both"):
            targets[side] += offset
    browser = None
    if args.viewer == "viser":
        browser = BrowserHandFollowViewer(
            controller, host=args.host, port=args.port, hand=args.hand,
        )
        browser.set_targets(targets)
    steps = 0
    timing = RuntimeTiming()
    next_tick = next_render = timing.started
    last_status = 0.0
    metrics = controller.metrics()
    active_hand = args.hand
    summary = None
    try:
        while args.steps == 0 or steps < args.steps:
            tick_started = time.monotonic()
            paused = False
            if browser is not None:
                targets, selected_hand, paused, reset_requested = browser.snapshot()
                if reset_requested:
                    controller.reset()
                    browser.reset_targets(controller)
                    targets = controller.target
                    metrics = controller.metrics()
                if selected_hand != active_hand or reset_requested:
                    active_hand = selected_hand
                    _select_hands(controller, active_hand)
            if not paused:
                for side, position in targets.items():
                    if active_hand in (side, "both"):
                        controller.set_target(side, position)
                simulation_start = float(controller.data.time)
                control_started = time.monotonic()
                metrics = controller.step()
                timing.record_step(
                    time.monotonic() - control_started,
                    max(0.0, float(controller.data.time) - simulation_start),
                )
                steps += 1
            now = time.monotonic()
            if browser is not None:
                rates = timing.status(now)
                if now >= next_render:
                    update_status = now - last_status >= 0.2
                    render_started = time.monotonic()
                    browser.update(
                        controller, metrics, steps=steps,
                        update_status=update_status, timing=rates,
                    )
                    timing.record_render(time.monotonic() - render_started)
                    render_period = 1.0 / 30.0
                    next_render += (int((now - next_render) / render_period) + 1) * render_period
                    if update_status:
                        last_status = now
            # Deadline pacing includes rendering. Small scheduling jitter is
            # recovered on the next tick; long stalls do not build a backlog.
            now = time.monotonic()
            dt = controller.config.control_dt
            next_tick = tick_started + dt if paused else next_tick + dt
            if next_tick < now - dt:
                next_tick = now
            time.sleep(max(0.0, next_tick - now))
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown cost is not part of the simulation's measured rate.
        summary = timing.summary()
        if browser is not None:
            browser.close()
    result = {
        "xml": str(args.xml) if args.xml else "default TIAGo++ floor scene",
        "controller": "task_space_qp_reference",
        "viewer": args.viewer,
        "hand": active_hand,
        "steps": steps,
        "offset_m": offset.tolist(),
        "gravity_m_s2": float(np.linalg.norm(controller.model.opt.gravity)),
        **summary,
    }
    result.update(controller.metrics())
    selected = controller.sides if active_hand == "both" else (active_hand,)
    for side in controller.sides:
        measured = controller.data.xpos[controller.wbc.body_ids[f"{side}_wrist"]]
        result[f"{side}_requested_error_m"] = float(np.linalg.norm(targets[side] - measured))
    result["target_error_m"] = max(result[f"{side}_requested_error_m"] for side in selected)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, help="TIAGo++ MuJoCo scene override")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
    parser.add_argument("--host", default="127.0.0.1", help="browser server bind address")
    parser.add_argument("--port", type=int, default=8080, help="browser server port")
    parser.add_argument(
        "--steps", type=int, default=0, help="control steps; 0 keeps browser running"
    )
    parser.add_argument(
        "--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
        help="initial world-axis hand target displacement in metres; applies to --hand",
    )
    args = parser.parse_args(argv)
    if args.steps < 0 or (args.viewer == "none" and args.steps == 0):
        parser.error("use positive --steps for headless runs; browser runs may use 0")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not np.isfinite(args.offset).all():
        parser.error("--offset requires three finite values")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
