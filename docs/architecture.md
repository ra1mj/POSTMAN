# POSTMAN architecture

POSTMAN follows the manager-based shape used by mjlab while keeping the core
representation and WBC code runnable without a simulator installation.

1. A task manager samples a short-horizon `WholeBodyCommand` in UMMR-1.0.
2. The native RL-WBC policy observes proprioception, current/next command
   waypoints and the previous action; the privileged critic also sees the QP
   conservative actuator reference and randomized dynamics context.
3. The policy emits a bounded 25-dimensional actuator target: four skid-steer
   wheel velocities followed by 21 joint positions.
4. A device-resident action filter performs physical scaling, rate limiting,
   smoothing, delay and invalid-action fallback. The native reference currently
   holds arm posture and commands base/lift motion; it does not run the full QP.
5. MuJoCo/mjlab applies the mixed velocity/position targets and exposes task,
   training diagnostics. Tracking quality and absence of oscillation still
   require evaluation of a trained checkpoint.

The separate browser demo runs the explicit `TiagoTaskSpaceWbc` on real MuJoCo
physics with gravity and floor contact. It integrates bounded joint setpoints
and offsets position-servo commands for gravity load; base pose is never
teleported during stepping. This is a QP reference demonstration, not an RL policy.

The dependency-light CPU path still accepts explicit `LinearTask` stacks and
uses `KinematicQpWbc` for regression tests. The native path trains the learned
whole-body controller at 100 Hz while keeping the UMMR command interface stable.

`DynamicWbcBackend` is intentionally an adapter boundary. The first backend is
the dependency-light `KinematicQpWbc`; TSID/ARC-OPT integrations can be added
without changing environments, policies, or datasets.
