# mjlab integration path

POSTMAN keeps its UMMR and WBC layers independent from mjlab, then attaches
them through a native mjlab `ManagerBasedRlEnv` task:

1. Build a `SceneCfg` from the pinned TIAGo++ MJCF asset.
2. Register an action term that converts the policy tensor into
   `WholeBodyAction` and calls the WBC backend.
3. Register actor and privileged critic observation groups.
4. Register reward, termination, command, event and metrics terms.
5. Wrap the resulting environment with mjlab's `RslRlVecEnvWrapper`.
6. Use `RslRlAdapter.build_mjlab_runner` to construct
   `MjlabOnPolicyRunner` and export JIT/ONNX policies.

The installed POSTMAN package also exposes the `mjlab.tasks` entry point. This
registers `Postman-TiagoPP-ArmReach` and
`Postman-TiagoPP-ArmReach-SimpleTable` directly in mjlab's task registry. The
first task drives the 21 position-controlled torso/head/arm/gripper joints;
the four wheel velocity actuators remain available for the next mobile-base
action term.

The CPU `VectorizedManagerEnv` mirrors the same lifecycle for tests. It does
not pretend to be a replacement for MuJoCo Warp; it catches shape, reset,
action clipping and reward-term regressions before a GPU environment is built.
