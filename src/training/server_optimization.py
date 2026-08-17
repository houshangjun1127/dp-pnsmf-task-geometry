"""Server-side post-processing for released federated updates."""

from __future__ import annotations

import torch


def server_ema_delta(
    current: torch.Tensor,
    proposed: torch.Tensor,
    velocity: torch.Tensor,
    *,
    momentum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an exponential moving average to a released model delta.

    This operation consumes only the current public model and an already
    released private update, so it is deterministic DP post-processing.
    ``momentum=0`` exactly recovers the unmodified proposed model.
    """

    if current.shape != proposed.shape or current.shape != velocity.shape:
        raise ValueError("current, proposed, and velocity must have equal shapes.")
    if any(tensor.dtype != current.dtype for tensor in (proposed, velocity)):
        raise TypeError("current, proposed, and velocity must share a dtype.")
    if any(tensor.device != current.device for tensor in (proposed, velocity)):
        raise ValueError("current, proposed, and velocity must share a device.")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1).")
    released_delta = proposed - current
    next_velocity = momentum * velocity + (1.0 - momentum) * released_delta
    return current + next_velocity, next_velocity
