"""Versioned UMMR trajectory storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from postman.representations import UMMR_SCHEMA_VERSION


@dataclass
class UMMRTrajectory:
    timestamps: np.ndarray
    commands: np.ndarray
    states: np.ndarray
    schema_version: str = UMMR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64).reshape(-1)
        self.commands = np.asarray(self.commands, dtype=np.float64)
        self.states = np.asarray(self.states, dtype=np.float64)
        if self.commands.ndim != 2 or self.states.ndim != 2:
            raise ValueError("commands and states must be rank-2 arrays")
        if not (self.timestamps.size == self.commands.shape[0] == self.states.shape[0]):
            raise ValueError("trajectory arrays must have the same number of samples")

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            timestamps=self.timestamps,
            commands=self.commands,
            states=self.states,
            schema_version=self.schema_version,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> UMMRTrajectory:
        data = np.load(path, allow_pickle=False)
        version = str(data["schema_version"])
        if version != UMMR_SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema {version!r}")
        return cls(data["timestamps"], data["commands"], data["states"], version)

    def save_hdf5(self, path: str | Path) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[data] for HDF5 support") from exc
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = self.schema_version
            handle.create_dataset("timestamps", data=self.timestamps)
            handle.create_dataset("commands", data=self.commands)
            handle.create_dataset("states", data=self.states)

    @classmethod
    def load_hdf5(cls, path: str | Path) -> UMMRTrajectory:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Install postman-rl-wbc[data] for HDF5 support") from exc
        with h5py.File(path, "r") as handle:
            version = handle.attrs.get("schema_version", "")
            if isinstance(version, bytes):
                version = version.decode()
            if version != UMMR_SCHEMA_VERSION:
                raise ValueError(f"unsupported trajectory schema {version!r}")
            return cls(handle["timestamps"][:], handle["commands"][:], handle["states"][:], version)
