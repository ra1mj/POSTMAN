"""Episode domain randomization for the native TIAGo++ environment.

The event is deliberately kept as one batched operation.  Defaults are always
read from the compiled MuJoCo model, so repeated resets never multiply an
already randomized value.  Only the ``robot`` entity is selected; furniture
and other scene bodies are left untouched.
"""

from __future__ import annotations

from typing import Any

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

_MASS_RANGE = 0.10
_JOINT_RANGE = 0.20
_ACTUATOR_RANGE = 0.15


def _device(env) -> torch.device:
    return torch.device(getattr(env, "device", getattr(env.sim, "device", "cpu")))


def _tensor(value: Any, env) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=_device(env))


def _ids(value: Any, env) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=_device(env)).flatten()


def _default_field(env, field: str) -> torch.Tensor:
    """Get an unmodified, single-world field from the compiled model."""
    sim = env.sim
    model = getattr(sim, "mj_model", None)
    if model is None:
        model = getattr(sim, "_mj_model", None)
    if model is not None and hasattr(model, field):
        return _tensor(getattr(model, field), env)
    getter = getattr(sim, "get_default_field", None)
    if getter is not None:
        return _tensor(getter(field), env)
    raise AttributeError("simulation has no compiled mj_model")


def _robot_indices(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    asset = env.scene["robot"]
    indexing = asset.indexing
    body_ids = _ids(indexing.body_ids, env)
    # joint_v_adr is already expanded to scalar DOF addresses. Include the
    # floating-base addresses when the robot is mobile.
    dof_ids = _ids(getattr(indexing, "joint_v_adr", torch.empty(0)), env)
    free_ids = _ids(getattr(indexing, "free_joint_v_adr", torch.empty(0)), env)
    if free_ids.numel():
        dof_ids = torch.cat((free_ids, dof_ids))
    if dof_ids.numel() == 0:
        dof_ids = _ids(getattr(indexing, "joint_ids", torch.empty(0)), env)
    ctrl_ids = _ids(getattr(indexing, "ctrl_ids", torch.empty(0)), env)
    return body_ids, dof_ids, ctrl_ids


def _all_envs(env, env_ids) -> torch.Tensor:
    device = _device(env)
    if env_ids is None:
        return torch.arange(int(env.num_envs), device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()


def _ensure_factors(env) -> dict[str, torch.Tensor]:
    """Create full-size factor buffers once, initialized to one."""
    state = getattr(env, "_postman_dr", None)
    if state is not None:
        return state
    body_ids, dof_ids, ctrl_ids = _robot_indices(env)
    nbody = int(_default_field(env, "body_mass").shape[0])
    ndof = int(_default_field(env, "dof_damping").shape[0])
    nctrl = int(_default_field(env, "actuator_gainprm").shape[0])
    device = _device(env)
    n = int(env.num_envs)
    state = {
        "mass": torch.ones(n, nbody, device=device),
        "inertia": torch.ones(n, nbody, device=device),
        "friction": torch.ones(n, ndof, device=device),
        "damping": torch.ones(n, ndof, device=device),
        "gain": torch.ones(n, nctrl, device=device),
        "bias": torch.ones(n, nctrl, device=device),
        # Delay is owned by the action filter; retaining a fixed feature keeps
        # the critic input shape stable before and after the first reset.
        "delay": torch.ones(n, 1, device=device),
        "body_ids": body_ids,
        "dof_ids": dof_ids,
        "ctrl_ids": ctrl_ids,
    }
    # Verbose aliases make the buffer self-describing for observation builders.
    state["mass_factor"] = state["mass"]
    state["inertia_factor"] = state["inertia"]
    state["friction_factor"] = state["friction"]
    state["damping_factor"] = state["damping"]
    state["gain_factor"] = state["gain"]
    state["bias_factor"] = state["bias"]
    state["delay_factor"] = state["delay"]
    setattr(env, "_postman_dr", state)
    return state


def _assign_world_field(
    field: torch.Tensor, env_ids: torch.Tensor, ids: torch.Tensor, values: torch.Tensor
) -> None:
    """Batched assignment to a per-world model field."""
    if ids.numel() == 0:
        return
    if field.ndim == values.ndim - 1:  # unexpanded single-world fallback
        field[ids] = values[0]
        return
    field[env_ids[:, None], ids] = values


@requires_model_fields(
    "body_mass",
    "body_inertia",
    "dof_frictionloss",
    "dof_damping",
    "actuator_gainprm",
    "actuator_biasprm",
    recompute=RecomputeLevel.set_const,
)
def randomize_dynamics(env, env_ids: torch.Tensor | None = None) -> None:
    """Randomize robot mass, joints and actuators on episode reset.

    Body mass and inertia share one factor per body (constant density change).
    Friction and damping are independent factors per DOF.  Gain and bias use
    one shared factor per actuator, preserving the actuator's internal
    proportional relationship. All draws are batched over ``env_ids``.
    """
    state = _ensure_factors(env)
    ids = _all_envs(env, env_ids)
    if ids.numel() == 0:
        return
    body_ids = state["body_ids"]
    dof_ids = state["dof_ids"]
    ctrl_ids = state["ctrl_ids"]
    device = _device(env)
    count = ids.numel()

    def draw(width: int, amount: float) -> torch.Tensor:
        return 1.0 + amount * (2.0 * torch.rand(count, width, device=device) - 1.0)

    mass = draw(body_ids.numel(), _MASS_RANGE)
    friction = draw(dof_ids.numel(), _JOINT_RANGE)
    damping = draw(dof_ids.numel(), _JOINT_RANGE)
    actuator = draw(ctrl_ids.numel(), _ACTUATOR_RANGE)

    state["mass"][ids[:, None], body_ids] = mass
    state["inertia"][ids[:, None], body_ids] = mass
    state["friction"][ids[:, None], dof_ids] = friction
    state["damping"][ids[:, None], dof_ids] = damping
    state["gain"][ids[:, None], ctrl_ids] = actuator
    state["bias"][ids[:, None], ctrl_ids] = actuator

    model = env.sim.model
    default_mass = _default_field(env, "body_mass")
    default_inertia = _default_field(env, "body_inertia")
    default_friction = _default_field(env, "dof_frictionloss")
    default_damping = _default_field(env, "dof_damping")
    default_gain = _default_field(env, "actuator_gainprm")
    default_bias = _default_field(env, "actuator_biasprm")

    _assign_world_field(model.body_mass, ids, body_ids, default_mass[body_ids] * mass)
    _assign_world_field(
        model.body_inertia, ids, body_ids, default_inertia[body_ids] * mass[..., None]
    )
    _assign_world_field(
        model.dof_frictionloss, ids, dof_ids, default_friction[dof_ids] * friction
    )
    _assign_world_field(model.dof_damping, ids, dof_ids, default_damping[dof_ids] * damping)
    _assign_world_field(
        model.actuator_gainprm,
        ids,
        ctrl_ids,
        default_gain[ctrl_ids] * actuator[..., None],
    )
    _assign_world_field(
        model.actuator_biasprm,
        ids,
        ctrl_ids,
        default_bias[ctrl_ids] * actuator[..., None],
    )


def dynamics_observation(env) -> torch.Tensor:
    """Return fixed-size per-world DR factors for critic observations.

    Before the first reset this returns all ones. Delay is represented by a
    single feature and can be updated by the action term when it is available.
    """
    state = _ensure_factors(env)
    # Expose only robot factors to the critic; full-width buffers remain on the
    # environment for exact model writes and diagnostics.
    mass = state["mass"][:, state["body_ids"]]
    friction = state["friction"][:, state["dof_ids"]]
    damping = state["damping"][:, state["dof_ids"]]
    gain = state["gain"][:, state["ctrl_ids"]]
    return torch.cat(
        (mass, friction, damping, gain, state["delay"]),
        dim=-1,
    )

