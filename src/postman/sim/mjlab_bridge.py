"""Optional bridge for running POSTMAN terms inside an mjlab environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MjlabStatus:
    installed: bool
    version: str | None = None
    message: str = ""


def check_mjlab() -> MjlabStatus:
    try:
        import mjlab
    except ImportError:
        return MjlabStatus(False, message="mjlab is not installed; use pip install -e '.[sim]'")
    return MjlabStatus(True, getattr(mjlab, "__version__", None), "mjlab import succeeded")


class MjlabEnvBridge:
    """Adapter around a constructed mjlab ``ManagerBasedRlEnv``.

    POSTMAN does not monkey-patch mjlab internals. The bridge only forwards the
    Gymnasium-shaped lifecycle, allowing POSTMAN task terms to be registered by
    a robot plugin once the exact mjlab release is pinned.
    """

    def __init__(self, env: Any):
        if not hasattr(env, "reset") or not hasattr(env, "step"):
            raise TypeError("env must expose reset() and step()")
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    @property
    def raw(self) -> Any:
        return self.env
