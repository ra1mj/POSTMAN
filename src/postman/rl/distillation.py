"""Privileged teacher to deployable student utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistillationBatch:
    student_observation: np.ndarray
    teacher_observation: np.ndarray
    teacher_action: np.ndarray

    def __post_init__(self) -> None:
        if self.student_observation.shape[0] != self.teacher_action.shape[0]:
            raise ValueError("student and teacher batch sizes must match")
        if self.teacher_observation.shape[0] != self.teacher_action.shape[0]:
            raise ValueError("teacher observation and action batch sizes must match")


def mse_distillation_loss(predicted_action, teacher_action) -> float:
    predicted = np.asarray(predicted_action, dtype=np.float64)
    teacher = np.asarray(teacher_action, dtype=np.float64)
    if predicted.shape != teacher.shape:
        raise ValueError("predicted and teacher actions must have the same shape")
    return float(np.mean((predicted - teacher) ** 2))
