# TIAGo++ browser hand-follow demo

The browser shows the complete MuJoCo robot, floor and a draggable XYZ target
at each hand. Moving a target asks the task-space QP reference controller to
coordinate the arms, torso and wheeled base. This demonstration uses the
reference controller; it does not load a trained RL policy.

## Open the robot

From the repository root, run:

```bash
.venv/bin/python scripts/hand_follow_demo.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). Both hand targets are active
by default. Drag a colored axis arrow for one-axis motion, or a square handle
for motion in a plane. The target axes and live robot share the same browser
scene; no native MuJoCo window or `mjpython` launcher is required on macOS.

Use **Active hand** to select `left`, `right`, or `both`. The inactive target
is dimmed. **Pause simulation** lets you reposition targets before resuming.
**Reset robot and targets** restores the initial scene and re-anchors both
hand targets. The panel reports each hand's tracking error and the base and
torso displacement; colored lines connect the requested targets to the hands.
It also reports the measured control rate, visual update rate, real-time
factor, controller/render cost, gravity, floor height and floor contacts. These
values make a slow browser refresh distinguishable from a slow physics/control
loop.

Start with small target movements. The robot's joint limits and its base's
nonholonomic motion can leave some requested positions unreachable, which
appears as remaining tracking error. The default scene has normal gravity and
a collision floor; the demo starts from a neutral arm pose and keeps the
floating base on that floor while the controller follows the target.

Choose another port or initial hand with:

```bash
.venv/bin/python scripts/hand_follow_demo.py --port 8081 --hand left
```

The server binds to `127.0.0.1` by default. Use `--host` to select a different
bind address. A custom MuJoCo scene can be selected with `--xml /path/to/scene.xml`.
Stop the process with Ctrl-C; a final JSON summary is printed.

## Run without a browser

```bash
.venv/bin/python scripts/hand_follow_demo.py --viewer none --steps 100
```

This loads the same controller, advances 100 control updates at the real 100
Hz wall-clock pace, and prints tracking metrics. It opens no browser or native
window. Give the selected hand a reproducible target offset to measure
whole-body response:

```bash
.venv/bin/python scripts/hand_follow_demo.py \
  --viewer none --hand left --offset 0.08 0.00 0.03 --steps 100
```

The three offset values are world-axis metres. A finite browser smoke run is
also available with `--viewer viser --steps 20 --port 8081`; the server shuts
down when the requested updates finish.

## Contact and response regression

The default Menagerie scene puts its floor 0.99 m below the wheel bottoms.
The demo corrects that plane to z=0 before compiling its own model; it never
edits the cached asset. Gravity remains -9.81 m/s². The free base moves through
wheel-ground contact, not direct base-pose assignment. Custom XML overrides
retain their own floor geometry.

The reference integrates joint position targets with bounded lead and adds
gravity/bias compensation through the existing position actuator commands.
Actuator control/force limits still apply. Wheel velocity gain is reduced from
2000 to 20 to avoid chassis pitch feedback; the wheel speed limit increases
from 1 to 4 rad/s and target filtering runs once per control tick.

The physical regression checks both hands after a +12 cm forward/+8 cm up
target change: error below 2.5 cm after 1 s and 1.5 cm after 2 s, with floor
contacts throughout. A separate 5 s stationary test checks drift, height and
tilt. Measured left-hand error after 1 s is 1.10 cm, with 5.47 cm base motion
and 3.55 cm lift motion. This is one tested target, not a guarantee for all
reachable poses or a trained RL result.
