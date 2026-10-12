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
2. Register the synthetic UMMR command term and a unified RL-WBC action term.
   The policy emits 25 normalized targets in canonical order: four wheel
   velocities followed by 21 joint positions. The action term applies
   normalization, smoothing, rate limits and bounded action delay.
3. Register actor and privileged critic observation groups. Actor observations
   contain proprioception, current/next UMMR targets and the previous action;
   critic observations additionally contain conservative reference targets and randomized
   dynamics context.
4. Register reward, termination, command, event and metrics terms. Tracking
   weights prioritize wrists over elbows, torso and footprint, and stability
   terms penalize limits, action rate and oscillation.
5. Wrap the resulting environment with mjlab's `RslRlVecEnvWrapper`.
6. Use `RslRlAdapter.build_mjlab_runner` to construct
   `MjlabOnPolicyRunner` and export JIT/ONNX policies.

The installed POSTMAN package exposes the `mjlab.tasks` entry point through
`src/postman/tasks`. This registers `Postman-TiagoPP-ArmReach` and
`Postman-TiagoPP-ArmReach-SimpleTable` directly in mjlab's task registry. Both
tasks drive all 25 actuators through one RL-WBC term at 100 Hz. Their current
fallback is a base/lift/arm-hold reference. The full `TiagoTaskSpaceWbc` is used
by the separate browser hand demo, not yet by the native teacher. Native PPO
training is experimental; a successful smoke iteration is not evidence of
converged whole-body tracking.

The CPU `VectorizedManagerEnv` mirrors the same lifecycle for tests. It does
not pretend to be a replacement for MuJoCo Warp; it catches shape, reset,
action clipping and reward-term regressions before a GPU environment is built.
