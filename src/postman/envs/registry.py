"""Legged-gym-style task registration without importing a simulator at import time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_cls: type[Any]
    cfg: Any


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskSpec] = {}

    def register(self, name: str, env_cls: type[Any], cfg: Any) -> None:
        if name in self._tasks:
            raise KeyError(f"task {name!r} is already registered")
        self._tasks[name] = TaskSpec(name, env_cls, cfg)

    def get(self, name: str) -> TaskSpec:
        try:
            return self._tasks[name]
        except KeyError as exc:
            raise KeyError(f"unknown task {name!r}; available={sorted(self._tasks)}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))

    def make(self, name: str, **kwargs: Any):
        spec = self.get(name)
        cfg = kwargs.pop("cfg", spec.cfg)
        return spec.env_cls(cfg=cfg, **kwargs)


TASK_REGISTRY = TaskRegistry()


def make(name: str, **kwargs: Any):
    return TASK_REGISTRY.make(name, **kwargs)
