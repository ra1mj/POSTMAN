# POSTMAN

POSTMAN is a manager-based MuJoCo/mjlab research framework for reinforcement
learning and whole-body control of wheeled mobile manipulators. The reference
robot is TIAGo++, with a floating base, four wheel actuators, a lift, a head,
two 7-DoF arms, and two grippers.

The repository has two execution paths:

- A dependency-light CPU reference path for UMMR, WBC, task registration,
  vectorized rollouts, and regression tests.
- A native mjlab path for MuJoCo Warp scenes, RSL-RL PPO, Viser visualization,
  and task registration through the `mjlab.tasks` entry-point group.

## Install

Use Python 3.12 on the tested Apple Silicon setup or Python 3.10+ on a Linux
workstation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
```

The tested environment is:

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| MuJoCo | 3.11.0 |
| mjlab | 1.6.0 |
| MuJoCo Menagerie | 2026.9.0 |
| MuJoCo Warp | 3.11.0 |
| PyTorch | 2.14.1 |
| RSL-RL | 5.4.2 |

On Apple Silicon, mjlab/Warp uses its CPU device. CUDA training requires a
Linux/NVIDIA machine.

## Get the TIAGo++ model

POSTMAN resolves the pinned Menagerie package automatically when
`mujoco-menagerie` is installed. The first command that needs the model places
it in the user cache:

```bash
postman-validate
postman-mujoco-smoke --steps 500
```

To use a local XML explicitly, set:

```bash
export POSTMAN_TIAGO_PP_MJCF=/absolute/path/to/scene_position.xml
```

The model metadata and provenance are documented in
[`assets/robots/tiago_pp/README.md`](assets/robots/tiago_pp/README.md).

## Run the CPU reference path

```bash
python -m pytest -q
ruff check src tests scripts

python -m postman.scripts.train \
  --task tiago_pp_bimanual_reach \
  --num-envs 32 \
  --steps 1000 \
  --output outputs/postman_policy.npz

python -m postman.scripts.play \
  --task tiago_pp_bimanual_reach \
  --checkpoint outputs/postman_policy.npz \
  --steps 500
```

This path uses the NumPy smoke policy and is intended to validate UMMR shapes,
WBC constraints, reset behavior, reward terms, and vectorized environment
interfaces. It is not a replacement for MuJoCo Warp training.

## Run native mjlab tasks

After installing the package with `pip install -e .`, POSTMAN tasks are visible
to mjlab:

```bash
list-envs --keyword Postman
```

Available tasks:

| Task | Purpose |
| --- | --- |
| `Postman-TiagoPP-ArmReach` | TIAGo++ scene with 21 position-controlled torso/head/arm/gripper joints and 4 wheel velocity actions |
| `Postman-TiagoPP-ArmReach-SimpleTable` | Same robot with the Apache-2.0 SimpleTable scene attached |

Run a short CPU PPO iteration:

```bash
train Postman-TiagoPP-ArmReach \
  --gpu-ids '[]' \
  --env.scene.num-envs 1 \
  --agent.max-iterations 1 \
  --agent.num-steps-per-env 24 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --log-root /tmp/postman_mjlab_train
```

On Linux/NVIDIA, omit `--gpu-ids '[]'` or select the desired GPU. The current
native task is the baseline scene/action integration. The custom WBC action
term remains the next control-layer integration point; the standalone
`KinematicQpWbc` is already tested through the CPU reference path.

## Open the visualization

Start a Viser browser viewer:

```bash
play Postman-TiagoPP-ArmReach-SimpleTable \
  --agent zero \
  --num-envs 1 \
  --device cpu \
  --viewer viser \
  --no-terminations True \
  --log-root /tmp/postman_tiago_view
```

Open [http://localhost:8080](http://localhost:8080). Stop the process with
`Ctrl-C`.

The scene contains the TIAGo++ model, a ground plane, and a table. The native
task exposes 21 arm/torso/head position actions and four wheel velocity
actions, so it is ready for the next mobile-base policy experiment.

## Compose furniture scenes

The checked-in Apache-2.0 furniture subset includes a simple table, hinged
cabinet, sliding cabinet, oven, and microwave. See
[`configs/scene_catalog.json`](configs/scene_catalog.json) and
[`third_party/scenes/furniture_sim/SOURCE.md`](third_party/scenes/furniture_sim/SOURCE.md).

Validate a composed TIAGo++ scene:

```bash
python -m postman.scripts.compose_scene \
  --furniture third_party/scenes/furniture_sim/simpleTable.xml \
  --furniture third_party/scenes/furniture_sim/hingecabinet.xml \
  --steps 250
```

The composer uses MuJoCo `MjSpec.attach`, which preserves Menagerie mesh paths
and avoids broken nested XML relative paths.

## Control architecture

The control path is:

```text
task/reference → RL policy → UMMR residual → WBC/QP → impedance/actuator commands
```

The public control types are in
[`src/postman/representations/ummr.py`](src/postman/representations/ummr.py).
The QP WBC and Pinocchio task seam are in
[`src/postman/wbc`](src/postman/wbc). The mjlab manager and RSL-RL integration
notes are in [`docs/mjlab_integration.md`](docs/mjlab_integration.md).

## Project layout

```text
src/postman/representations/   UMMR-1.0 state/action/command types
src/postman/wbc/               QP WBC, impedance control, task builders
src/postman/envs/              CPU manager env, registry, vectorized runner
src/postman/mjlab_tasks/       native mjlab task registration
src/postman/robots/             TIAGo++ metadata and asset resolution
src/postman/sim/                MuJoCo adapter, scene composer, mjlab bridge
src/postman/rl/                 smoke policy, RSL-RL adapter, distillation
third_party/scenes/             licensed furniture assets
tests/                          unit, scene, vectorization, and WBC tests
```

## Validation status

The current checkout has been validated with:

- `pytest`: 9 tests passed
- `ruff check`: clean
- MuJoCo TIAGo++ smoke: 500 stable steps, `nq=32`, `nv=31`, `nu=25`
- TIAGo++ + SimpleTable + HingeCabinet composition: compiled and stepped
- mjlab/RSL-RL CPU PPO: one 24-step training iteration completed
- Viser visualization: TIAGo++ with table background at `localhost:8080`

See [`licenses/dependency-audit.md`](licenses/dependency-audit.md) for the
tested environment and license policy.
