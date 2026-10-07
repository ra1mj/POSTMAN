# POSTMAN architecture

POSTMAN follows the manager-based shape used by mjlab while keeping the core
representation and WBC code runnable without a simulator installation.

1. A task manager samples a `WholeBodyCommand` in UMMR-1.0.
2. The policy observes proprioception, command state, and optional privileged
   teacher inputs.
3. The policy emits a bounded 24-dimensional residual.
4. The WBC converts the residual into task objectives and solves for joint and
   base commands.
5. MuJoCo/mjlab applies impedance or torque actuation and exposes diagnostics.

`DynamicWbcBackend` is intentionally an adapter boundary. The first backend is
the dependency-light `KinematicQpWbc`; TSID/ARC-OPT integrations can be added
without changing environments, policies, or datasets.
