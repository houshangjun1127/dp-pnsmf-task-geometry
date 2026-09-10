"""Numerical primitives for the controlled DPALS-style baseline.

The loss follows the implicit-feedback extension described by Chien et al.
(ICML 2021): observed binary squared error plus a full-matrix quadratic
penalty that prevents the all-ones solution.
"""

from __future__ import annotations

import torch


def dpals_implicit_loss_indexed(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    *,
    implicit_penalty: float,
    regularization: float,
) -> torch.Tensor:
    """Evaluate the sparse implicit DPALS objective without a dense matrix."""

    if user_factors.ndim != 2 or item_factors.ndim != 2:
        raise ValueError("Factor matrices must be rank two.")
    if user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("Factor dimensions must match.")
    if train_users.ndim != 1 or train_items.ndim != 1:
        raise ValueError("Interaction indices must be vectors.")
    if train_users.shape != train_items.shape:
        raise ValueError("Interaction indices must be aligned.")
    if train_users.dtype != torch.long or train_items.dtype != torch.long:
        raise TypeError("Interaction indices must be LongTensors.")
    if implicit_penalty < 0.0 or regularization < 0.0:
        raise ValueError("Penalties must be non-negative.")
    interaction_users = user_factors.index_select(0, train_users)
    interaction_items = item_factors.index_select(0, train_items)
    predictions = (interaction_users * interaction_items).sum(dim=1)
    observed_loss = (1.0 - predictions).square().sum()
    user_moment = user_factors.T @ user_factors
    item_moment = item_factors.T @ item_factors
    full_prediction_energy = torch.sum(user_moment * item_moment)
    return (
        observed_loss
        + implicit_penalty * full_prediction_energy
        + regularization
        * (user_factors.square().sum() + item_factors.square().sum())
    )


def random_orthonormal_item_factors(
    num_items: int,
    dimension: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a data-independent random item matrix with orthonormal columns."""

    if num_items <= 0 or dimension <= 0 or dimension > num_items:
        raise ValueError("Require 0 < dimension <= num_items.")
    raw = torch.randn(
        (num_items, dimension),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    orthonormal, signs = torch.linalg.qr(raw, mode="reduced")
    diagonal = torch.diagonal(signs)
    direction = torch.where(diagonal < 0, -1.0, 1.0).to(dtype=dtype)
    return orthonormal * direction.unsqueeze(0)

