# mjlab integration path

POSTMAN follows the same split used by mjlab's Unitree robot tasks:

```text
src/postman/asset_zoo/robots/tiago_pp/
  tiago_pp_constants.py       # EntityCfg and actuator groups
src/postman/tasks/arm_reach/config/tiago_pp/
  env_cfgs.py                 # ManagerBasedRlEnvCfg variants
  rl_cfg.py                   # RslRlOnPolicyRunnerCfg
  __init__.py                 # register_mjlab_task calls
```

The UMMR/WBC layers stay independent from mjlab, then attach through a native
mjlab `ManagerBasedRlEnv` task:

1. Build a `SceneCfg` from the pinned TIAGo++ MJCF asset.
2. Register an action term that converts the policy tensor into
   `WholeBodyAction` and calls the WBC backend.
3. Register actor and privileged critic observation groups.
4. Register reward, termination, command, event and metrics terms.
5. Wrap the resulting environment with mjlab's `RslRlVecEnvWrapper`.
6. Use `RslRlAdapter.build_mjlab_runner` to construct
   `MjlabOnPolicyRunner` and export JIT/ONNX policies.

The installed POSTMAN package exposes the `mjlab.tasks` entry point through
`src/postman/tasks`. This registers `Postman-TiagoPP-ArmReach` and
`Postman-TiagoPP-ArmReach-SimpleTable` directly in mjlab's task registry. The
first task drives 21 position-controlled torso/head/arm/gripper joints and
four wheel velocity actuators as a separate action term.

The CPU `VectorizedManagerEnv` mirrors the same lifecycle for tests. It does
not pretend to be a replacement for MuJoCo Warp; it catches shape, reset,
action clipping and reward-term regressions before a GPU environment is built.
