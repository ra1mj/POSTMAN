# POSTMAN

POSTMAN is a manager-based MuJoCo/mjlab research framework for reinforcement
learning and whole-body control of wheeled mobile manipulators.

The v1 reference platform is TIAGo++ from MuJoCo Menagerie. The repository
keeps the model external so its upstream license and revision can be preserved;
set `POSTMAN_TIAGO_PP_MJCF` to the Menagerie `scene.xml` path when using the
real simulator. The dependency-light environment and WBC tests run without a
MuJoCo installation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
# Full simulator/control stack:
pip install -e '.[all]'
```

The tested CPU simulation stack on Apple Silicon is `mujoco==3.11.0`,
`mjlab==1.6.0`, `mujoco-menagerie==2026.9.0`, `torch==2.14.1`, and
`rsl-rl-lib==5.4.2`. mjlab/Warp runs with its CPU device on this machine;
CUDA is unavailable on Apple Silicon, so GPU throughput claims require a
Linux/NVIDIA host.

## Smoke run

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m postman.scripts.train \
  --task tiago_pp_bimanual_reach --steps 1000 \
  --output outputs/postman_policy.npz
PYTHONPATH=src python -m postman.scripts.play \
  --task tiago_pp_bimanual_reach \
  --checkpoint outputs/postman_policy.npz
PYTHONPATH=src python -m postman.scripts.train \
  --task tiago_pp_bimanual_reach --num-envs 32 --steps 1000
PYTHONPATH=src python -m postman.scripts.validate
PYTHONPATH=src python -m postman.scripts.mujoco_smoke --steps 500
PYTHONPATH=src python -m postman.scripts.compose_scene \
  --furniture third_party/scenes/furniture_sim/simpleTable.xml \
  --furniture third_party/scenes/furniture_sim/hingecabinet.xml

# Native mjlab task registration (after `pip install -e .`):
list-envs --keyword Postman
train Postman-TiagoPP-ArmReach \
  --gpu-ids '[]' --env.scene.num-envs 1 \
  --agent.max-iterations 1 --agent.num-steps-per-env 24 \
  --agent.logger tensorboard --agent.upload-model False
```

The `numpy` policy is a dependency-light smoke backend. Install the optional
`rl` and `sim` extras before connecting RSL-RL and mjlab. The public interfaces
are intentionally independent of those optional packages.

The native mjlab entry point registers `Postman-TiagoPP-ArmReach` and
`Postman-TiagoPP-ArmReach-SimpleTable`. The latter composes the real TIAGo++
Menagerie robot XML with the checked-in Apache-2.0 table asset through
`MjSpec.attach`.

For GPU training with the current mjlab stack, construct a native mjlab
`ManagerBasedRlEnv` and pass it to `RslRlAdapter.build_mjlab_runner`; mjlab's
own `RslRlVecEnvWrapper` and `MjlabOnPolicyRunner` are used in that path. The
CPU reference path uses `VectorizedManagerEnv` and is intended for interface
and reward/WBC regression tests.

## Control contract

The policy emits a bounded UMMR-1.0 residual. `KinematicQpWbc` converts it to
joint targets and impedance torques under per-joint step limits. Robot-specific
Cartesian Jacobians can be supplied as `LinearTask` objects; this is the seam
for Pinocchio, TSID and ARC-OPT backends.

See [docs/architecture.md](docs/architecture.md) and the dependency audit in
[licenses/dependency-audit.md](licenses/dependency-audit.md).

The available scene candidates and their intended tasks are recorded in
[`configs/scene_catalog.json`](configs/scene_catalog.json). The checked-in
Apache-2.0 furniture subset is under
[`third_party/scenes/furniture_sim`](third_party/scenes/furniture_sim).
