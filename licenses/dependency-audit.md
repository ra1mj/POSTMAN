# Dependency audit

| Component | Role | License policy |
| --- | --- | --- |
| mjlab / MuJoCo | GPU simulation and manager API | Apache-2.0 / upstream notices |
| RSL-RL | PPO and distillation | BSD-3-Clause |
| Pinocchio / TSID | dynamics and inverse dynamics | BSD |
| OSQP | QP backend | Apache-2.0 |
| TIAGo++ Menagerie model | canonical robot asset | Preserve per-model Apache-2.0 notice |

## Tested environment (2026-10-05)

- Python 3.12.14 on Apple M4 macOS arm64
- MuJoCo 3.11.0
- mjlab 1.6.0
- MuJoCo Warp 3.11.0 / Warp 1.17.0 CPU device
- PyTorch 2.14.1 with MPS available; CUDA unavailable
- RSL-RL 5.4.2
- MuJoCo Menagerie 2026.9.0

This file is a planning record; a release build must regenerate it from the
locked environment and asset revisions.
