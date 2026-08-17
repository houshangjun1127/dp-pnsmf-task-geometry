"""Uniform client-update clipping and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClippingDiagnostics:
    """Per-client diagnostics retained locally for mechanism analysis."""

    pre_clip_norm: torch.Tensor
    post_clip_norm: torch.Tensor
    clipping_factor: torch.Tensor
    clipped: torch.Tensor
    cosine_similarity: torch.Tensor


def clip_client_updates(
    updates: torch.Tensor,
    max_norm: float,
    *,
    numerical_epsilon: float = 1e-12,
) -> tuple[torch.Tensor, ClippingDiagnostics]:
    """Clip a batch of flattened client updates to one shared L2 bound.

    Args:
        updates: Tensor shaped ``[num_clients, num_parameters]``.
        max_norm: Shared final contribution bound for every client.
        numerical_epsilon: Denominator guard for zero updates.

    Returns:
        The clipped updates and per-client diagnostics.
    """

    if updates.ndim != 2:
        raise ValueError("updates must have shape [num_clients, num_parameters].")
    if not torch.is_floating_point(updates):
        raise TypeError("updates must use a floating-point dtype.")
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")
    if numerical_epsilon <= 0.0:
        raise ValueError("numerical_epsilon must be positive.")
    if not torch.isfinite(updates).all():
        raise ValueError("updates contain NaN or infinite values.")

    pre_norm = torch.linalg.vector_norm(updates, ord=2, dim=1)
    factor = torch.clamp(max_norm / pre_norm.clamp_min(numerical_epsilon), max=1.0)
    clipped_updates = updates * factor.unsqueeze(1)
    post_norm = torch.linalg.vector_norm(clipped_updates, ord=2, dim=1)

    dot = torch.sum(updates * clipped_updates, dim=1)
    denominator = pre_norm * post_norm
    cosine = torch.where(
        denominator > numerical_epsilon,
        dot / denominator.clamp_min(numerical_epsilon),
        torch.ones_like(denominator),
    )

    diagnostics = ClippingDiagnostics(
        pre_clip_norm=pre_norm,
        post_clip_norm=post_norm,
        clipping_factor=factor,
        clipped=pre_norm > max_norm,
        cosine_similarity=cosine,
    )
    return clipped_updates, diagnostics


def public_model_scaled_clip_norm(
    item_factors: torch.Tensor,
    *,
    scale_ratio: float,
    previous_clip_norm: float | None = None,
    max_growth_factor: float = 1.05,
    min_clip_norm: float = 0.0,
    max_clip_norm: float = float("inf"),
) -> float:
    """Return a clipping bound derived only from public model state.

    The multiplicative growth cap prevents noisy model norms from immediately
    creating a larger bound and correspondingly larger Gaussian noise.
    """

    if item_factors.ndim != 2 or not item_factors.is_floating_point():
        raise ValueError("item_factors must be a floating-point matrix.")
    if scale_ratio <= 0.0 or max_growth_factor < 1.0:
        raise ValueError("scale_ratio must be positive and max_growth_factor at least one.")
    if min_clip_norm < 0.0 or max_clip_norm <= 0.0 or min_clip_norm > max_clip_norm:
        raise ValueError("Invalid clipping-bound limits.")
    if previous_clip_norm is not None and previous_clip_norm <= 0.0:
        raise ValueError("previous_clip_norm must be positive when supplied.")
    raw = scale_ratio * float(torch.linalg.vector_norm(item_factors))
    if previous_clip_norm is not None:
        raw = min(raw, previous_clip_norm * max_growth_factor)
    return min(max(raw, min_clip_norm), max_clip_norm)
