"""Controlled DPALS-style sufficient-statistic ALS updates.

This retains the essential private item-update construction from Chien et al.
(ICML 2021) and the paper's implicit-feedback shared Gram term.  It omits the
paper's private popularity filtering, adaptive sampling, and centering, so the
result must be described as a controlled DPALS-style baseline rather than an
exact reproduction of the original benchmark pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.privacy import isometric_vector_to_symmetric


@dataclass(frozen=True)
class DPALSUserDiagnostics:
    pre_clip_norm: torch.Tensor
    clipping_factor: torch.Tensor


@dataclass(frozen=True)
class DPALSItemDiagnostics:
    capped_interactions: int
    item_gram_noise_norm: float
    item_vector_noise_norm: float
    shared_gram_noise_norm: float
    systems_with_nonpositive_eigenvalue: int
    retained_rank_min: int
    retained_rank_mean: float
    retained_rank_max: int


def _validate_indices(
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    *,
    num_users: int,
    num_items: int,
) -> None:
    if train_users.ndim != 1 or train_items.ndim != 1:
        raise ValueError("Interaction indices must be one-dimensional.")
    if train_users.shape != train_items.shape:
        raise ValueError("Interaction indices must be aligned.")
    if train_users.dtype != torch.long or train_items.dtype != torch.long:
        raise TypeError("Interaction indices must be LongTensors.")
    if len(train_users):
        if int(train_users.min()) < 0 or int(train_users.max()) >= num_users:
            raise IndexError("train_users contains an out-of-range index.")
        if int(train_items.min()) < 0 or int(train_items.max()) >= num_items:
            raise IndexError("train_items contains an out-of-range index.")


def dpals_user_update(
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    *,
    num_users: int,
    implicit_penalty: float,
    regularization: float,
    clip_radius: float | None,
) -> tuple[torch.Tensor, DPALSUserDiagnostics]:
    """Solve local implicit ALS systems and optionally clip each user row."""

    if item_factors.ndim != 2 or not item_factors.is_floating_point():
        raise ValueError("item_factors must be a floating-point matrix.")
    if implicit_penalty < 0.0 or regularization <= 0.0:
        raise ValueError("implicit_penalty must be non-negative and regularization positive.")
    if clip_radius is not None and clip_radius <= 0.0:
        raise ValueError("clip_radius must be positive when supplied.")
    _validate_indices(
        train_users,
        train_items,
        num_users=num_users,
        num_items=item_factors.shape[0],
    )
    dimension = item_factors.shape[1]
    identity = torch.eye(
        dimension, dtype=item_factors.dtype, device=item_factors.device
    )
    interaction_items = item_factors.index_select(0, train_items)
    rhs = torch.zeros(
        (num_users, dimension),
        dtype=item_factors.dtype,
        device=item_factors.device,
    )
    rhs.index_add_(0, train_users, interaction_items)
    observed_moments = torch.zeros(
        (num_users, dimension, dimension),
        dtype=item_factors.dtype,
        device=item_factors.device,
    )
    observed_moments.index_add_(
        0,
        train_users,
        interaction_items.unsqueeze(2) * interaction_items.unsqueeze(1),
    )
    systems = (
        observed_moments
        + implicit_penalty * (item_factors.T @ item_factors).unsqueeze(0)
        + regularization * identity.unsqueeze(0)
    )
    users = torch.linalg.solve(systems, rhs.unsqueeze(-1)).squeeze(-1)
    pre_clip_norm = users.norm(dim=1)
    if clip_radius is None:
        clipping_factor = torch.ones_like(pre_clip_norm)
    else:
        clipping_factor = torch.clamp(
            clip_radius / pre_clip_norm.clamp_min(1e-30), max=1.0
        )
        users = users * clipping_factor.unsqueeze(1)
    return users, DPALSUserDiagnostics(
        pre_clip_norm=pre_clip_norm,
        clipping_factor=clipping_factor,
    )


def _symmetric_gaussian(
    batch_shape: tuple[int, ...],
    dimension: int,
    *,
    standard_deviation: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    packed_dimension = dimension * (dimension + 1) // 2
    packed = torch.randn(
        (*batch_shape, packed_dimension),
        dtype=dtype,
        device=device,
        generator=generator,
    ) * standard_deviation
    return isometric_vector_to_symmetric(packed, dimension)


def _psd_pseudoinverse_solve(
    systems: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eigenvalue_rcond: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(systems)
    maximum = eigenvalues.clamp_min(0.0).amax(dim=1, keepdim=True)
    threshold = maximum * eigenvalue_rcond
    retained = (eigenvalues > threshold) & (eigenvalues > 0.0)
    inverse = torch.where(retained, eigenvalues.reciprocal(), 0.0)
    rotated_rhs = torch.einsum("bdi,bd->bi", eigenvectors, rhs)
    solution = torch.einsum(
        "bdi,bi->bd", eigenvectors, inverse * rotated_rhs
    )
    return solution, eigenvalues, retained.sum(dim=1)


def _whiten_item_factors(
    item_factors: torch.Tensor, *, eigenvalue_rcond: float
) -> torch.Tensor:
    gram = item_factors.T @ item_factors
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues.clamp_min(0.0).max()
    retained = (eigenvalues > maximum * eigenvalue_rcond) & (eigenvalues > 0.0)
    inverse_sqrt = torch.where(retained, eigenvalues.rsqrt(), 0.0)
    transform = (eigenvectors * inverse_sqrt.unsqueeze(0)) @ eigenvectors.T
    return item_factors @ transform


def dpals_private_item_update(
    user_factors: torch.Tensor,
    capped_train_users: torch.Tensor,
    capped_train_items: torch.Tensor,
    *,
    num_items: int,
    implicit_penalty: float,
    regularization: float,
    user_clip_radius: float,
    entry_clip: float,
    noise_multiplier: float,
    eigenvalue_rcond: float = 1e-8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, DPALSItemDiagnostics]:
    """Release noisy per-item sufficient statistics and solve ALS systems."""

    if user_factors.ndim != 2 or not user_factors.is_floating_point():
        raise ValueError("user_factors must be a floating-point matrix.")
    if implicit_penalty < 0.0 or regularization <= 0.0:
        raise ValueError("implicit_penalty must be non-negative and regularization positive.")
    if user_clip_radius <= 0.0 or entry_clip <= 0.0:
        raise ValueError("Clipping parameters must be positive.")
    if noise_multiplier < 0.0 or eigenvalue_rcond <= 0.0:
        raise ValueError("noise_multiplier must be non-negative and rcond positive.")
    if float(user_factors.norm(dim=1).max()) > user_clip_radius * (1.0 + 1e-8):
        raise ValueError("user_factors exceed the declared clipping radius.")
    _validate_indices(
        capped_train_users,
        capped_train_items,
        num_users=user_factors.shape[0],
        num_items=num_items,
    )
    dimension = user_factors.shape[1]
    dtype = user_factors.dtype
    device = user_factors.device
    interaction_users = user_factors.index_select(0, capped_train_users)
    rhs = torch.zeros((num_items, dimension), dtype=dtype, device=device)
    rhs.index_add_(0, capped_train_items, entry_clip * interaction_users)
    observed_moments = torch.zeros(
        (num_items, dimension, dimension), dtype=dtype, device=device
    )
    observed_moments.index_add_(
        0,
        capped_train_items,
        interaction_users.unsqueeze(2) * interaction_users.unsqueeze(1),
    )

    gram_noise = _symmetric_gaussian(
        (num_items,),
        dimension,
        standard_deviation=user_clip_radius**2 * noise_multiplier,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    vector_noise = torch.randn(
        rhs.shape, dtype=dtype, device=device, generator=generator
    ) * (user_clip_radius * entry_clip * noise_multiplier)
    shared_noise = _symmetric_gaussian(
        (),
        dimension,
        standard_deviation=user_clip_radius**2 * noise_multiplier,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    identity = torch.eye(dimension, dtype=dtype, device=device)
    shared_gram = user_factors.T @ user_factors + shared_noise
    systems = (
        observed_moments
        + gram_noise
        + implicit_penalty * shared_gram.unsqueeze(0)
        + regularization * identity.unsqueeze(0)
    )
    noisy_rhs = rhs + vector_noise
    items, eigenvalues, retained_rank = _psd_pseudoinverse_solve(
        systems, noisy_rhs, eigenvalue_rcond=eigenvalue_rcond
    )
    items = _whiten_item_factors(items, eigenvalue_rcond=eigenvalue_rcond)
    diagnostics = DPALSItemDiagnostics(
        capped_interactions=int(len(capped_train_users)),
        item_gram_noise_norm=float(torch.linalg.vector_norm(gram_noise)),
        item_vector_noise_norm=float(torch.linalg.vector_norm(vector_noise)),
        shared_gram_noise_norm=float(torch.linalg.vector_norm(shared_noise)),
        systems_with_nonpositive_eigenvalue=int(
            (eigenvalues[:, 0] <= 0.0).sum().item()
        ),
        retained_rank_min=int(retained_rank.min().item()),
        retained_rank_mean=float(retained_rank.double().mean().item()),
        retained_rank_max=int(retained_rank.max().item()),
    )
    return items, diagnostics

