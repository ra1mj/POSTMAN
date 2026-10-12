"""Device-resident, batched box-constrained least squares for WBC baselines.

The decision variable is a canonical joint velocity, including the four wheel
velocities. All tensors stay on the caller's device. This is a fixed-iteration
projected solver, not an exact active-set QP solver or the learned RL WBC.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchedLeastSquaresConfig:
    regularization: float = 1e-3
    iterations: int = 48

    def __post_init__(self):
        if self.regularization <= 0 or self.iterations <= 0:
            raise ValueError("regularization and iterations must be positive")


class BatchedBoundedLeastSquares:
    """Minimize weighted ``||J velocity - target||²`` under box bounds.

    Inputs are ``J[N,M,D]``, ``target[N,M]``, and broadcastable ``lower/upper``
    bounds. Optional row weights have shape ``[M]`` or ``[N,M]``. Callers must
    provide finite tensors and feasible bounds; shape checks do not synchronize
    GPU data. There are no environment loops, NumPy conversions, or host copies.
    """

    def __init__(self, config: BatchedLeastSquaresConfig | None = None):
        self.config = config or BatchedLeastSquaresConfig()

    def solve(self, jacobian, target, lower, upper, weights=None):
        import torch

        target = torch.as_tensor(target, dtype=jacobian.dtype, device=jacobian.device)
        if target.ndim == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if jacobian.ndim != 3 or target.shape != jacobian.shape[:2]:
            raise ValueError("expected J[N,M,D] and target[N,M]")
        if not jacobian.is_floating_point() or not target.is_floating_point():
            raise ValueError("jacobian and target must be floating point tensors")
        dof = jacobian.shape[-1]
        lower = torch.as_tensor(lower, dtype=jacobian.dtype, device=jacobian.device)
        upper = torch.as_tensor(upper, dtype=jacobian.dtype, device=jacobian.device)
        try:
            lower = torch.broadcast_to(lower, (jacobian.shape[0], dof))
            upper = torch.broadcast_to(upper, (jacobian.shape[0], dof))
        except RuntimeError as exc:
            raise ValueError("lower and upper must broadcast to [N,D]") from exc
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=jacobian.dtype, device=jacobian.device)
            try:
                weights = torch.broadcast_to(weights, target.shape)
            except RuntimeError as exc:
                raise ValueError("weights must broadcast to [N,M]") from exc
        if bool(torch.any(lower > upper)):
            raise ValueError("lower bounds must not exceed upper bounds")
        weighted_j = jacobian if weights is None else jacobian * weights[..., None]
        transpose = jacobian.transpose(-1, -2)
        hessian = transpose @ weighted_j
        eye = torch.eye(jacobian.shape[-1], dtype=jacobian.dtype, device=jacobian.device)
        hessian = hessian + self.config.regularization * eye
        weighted_target = target if weights is None else target * weights
        rhs = (transpose @ weighted_target.unsqueeze(-1)).squeeze(-1)
        velocity = torch.linalg.solve(hessian, rhs.unsqueeze(-1)).squeeze(-1)
        velocity = torch.maximum(lower, torch.minimum(upper, velocity))
        step = hessian.abs().sum(dim=-1).amax(dim=-1, keepdim=True).reciprocal()
        for _ in range(self.config.iterations):
            gradient = (hessian @ velocity.unsqueeze(-1)).squeeze(-1) - rhs
            velocity = torch.maximum(lower, torch.minimum(upper, velocity - step * gradient))
        return velocity


def mixed_actuator_targets(q, velocity, dt: float, lower_q=None, upper_q=None):
    """Return canonical TIAGo targets: four wheel velocities, 21 positions.

    ``q`` and ``velocity`` may be torch tensors or NumPy arrays; the operation
    preserves type, shape, dtype, and device. Wheels are never integrated into
    position targets. Position bounds apply only to the other 21 joints.
    """

    if dt <= 0 or q.shape != velocity.shape or q.shape[-1] != 25:
        raise ValueError("expected matching (...,25) q/velocity and positive dt")
    targets = q + dt * velocity
    if hasattr(targets, "clone"):
        import torch

        targets = targets.clone()
        if lower_q is not None:
            lower_q = torch.as_tensor(lower_q, dtype=targets.dtype, device=targets.device)
            if lower_q.shape[-1] == 21:
                targets[..., 4:] = torch.maximum(targets[..., 4:], lower_q)
            else:
                targets = torch.maximum(targets, lower_q)
        if upper_q is not None:
            upper_q = torch.as_tensor(upper_q, dtype=targets.dtype, device=targets.device)
            if upper_q.shape[-1] == 21:
                targets[..., 4:] = torch.minimum(targets[..., 4:], upper_q)
            else:
                targets = torch.minimum(targets, upper_q)
    else:
        import numpy as np

        targets = targets.copy()
        if lower_q is not None:
            lower_q = np.asarray(lower_q)
            if lower_q.shape[-1] == 21:
                targets[..., 4:] = np.maximum(targets[..., 4:], lower_q)
            else:
                targets = np.maximum(targets, lower_q)
        if upper_q is not None:
            upper_q = np.asarray(upper_q)
            if upper_q.shape[-1] == 21:
                targets[..., 4:] = np.minimum(targets[..., 4:], upper_q)
            else:
                targets = np.minimum(targets, upper_q)
    targets[..., :4] = velocity[..., :4]
    return targets
