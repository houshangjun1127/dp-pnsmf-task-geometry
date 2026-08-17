"""User-level clipping primitives for P-NSMF sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PNSMFContributionDiagnostics:
    selected_users: torch.Tensor
    observed_rows: int
    pre_clip_norm: torch.Tensor
    clipping_factor: torch.Tensor
    clipped: torch.Tensor


def _contribution_scaling_factors(
    norms: torch.Tensor,
    max_norm: float,
    contribution_rule: str,
    *,
    numerical_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return factors for ordinary clipping or DP-NormFedAvg normalization."""

    if contribution_rule == "clip":
        return torch.clamp(
            max_norm / norms.clamp_min(numerical_epsilon), max=1.0
        )
    if contribution_rule == "normalize":
        return torch.where(
            norms > numerical_epsilon,
            max_norm / norms.clamp_min(numerical_epsilon),
            torch.zeros_like(norms),
        )
    raise ValueError("contribution_rule must be clip or normalize.")


def symmetric_to_isometric_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Half-vectorize symmetric matrices while preserving Frobenius norms."""

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must end in equal square dimensions.")
    if not matrix.is_floating_point():
        raise TypeError("matrix must use a floating-point dtype.")
    if not torch.allclose(matrix, matrix.transpose(-1, -2), atol=1e-10, rtol=1e-8):
        raise ValueError("matrix must be symmetric.")
    dimension = matrix.shape[-1]
    rows, columns = torch.triu_indices(dimension, dimension, device=matrix.device)
    values = matrix[..., rows, columns]
    scales = torch.where(
        rows == columns,
        torch.ones_like(rows, dtype=matrix.dtype),
        torch.full_like(rows, 2.0**0.5, dtype=matrix.dtype),
    )
    return values * scales


def isometric_vector_to_symmetric(vector: torch.Tensor, dimension: int) -> torch.Tensor:
    """Invert :func:`symmetric_to_isometric_vector`."""

    if vector.ndim < 1 or not vector.is_floating_point():
        raise ValueError("vector must be a floating-point tensor.")
    if dimension <= 0 or vector.shape[-1] != dimension * (dimension + 1) // 2:
        raise ValueError("vector length does not match the symmetric dimension.")
    rows, columns = torch.triu_indices(dimension, dimension, device=vector.device)
    scales = torch.where(
        rows == columns,
        torch.ones_like(rows, dtype=vector.dtype),
        torch.full_like(rows, 2.0**0.5, dtype=vector.dtype),
    )
    values = vector / scales
    matrix = torch.zeros(
        (*vector.shape[:-1], dimension, dimension),
        dtype=vector.dtype,
        device=vector.device,
    )
    matrix[..., rows, columns] = values
    matrix[..., columns, rows] = values
    return matrix


def aggregate_clipped_pnsmf_statistics(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    selected_users: torch.Tensor,
    *,
    omega: float,
    max_norm: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    contribution_rule: str = "clip",
) -> tuple[torch.Tensor, torch.Tensor, PNSMFContributionDiagnostics]:
    """Bound and sum users' sparse item rows and second moments."""

    if user_factors.ndim != 2 or item_factors.ndim != 2:
        raise ValueError("Factor matrices must be rank two.")
    if user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("Factor dimensions do not match.")
    if train_users.shape != train_items.shape or train_users.ndim != 1:
        raise ValueError("Interaction indices must be aligned vectors.")
    if train_users.dtype != torch.long or train_items.dtype != torch.long:
        raise TypeError("Interaction indices must be LongTensors.")
    if selected_users.ndim != 1 or selected_users.dtype != torch.long:
        raise TypeError("selected_users must be a one-dimensional LongTensor.")
    if len(torch.unique(selected_users)) != len(selected_users):
        raise ValueError("selected_users must not contain duplicates.")
    if max_norm <= 0.0 or omega <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Norm bound and loss weights must be positive.")
    device = user_factors.device
    tensors = (item_factors, train_users, train_items, selected_users)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("All tensors must be on the same device.")

    selected_factors = user_factors.index_select(0, selected_users)
    lookup = torch.full(
        (user_factors.shape[0],), -1, dtype=torch.long, device=device
    )
    lookup[selected_users] = torch.arange(len(selected_users), device=device)
    local_interaction_users = lookup.index_select(0, train_users)
    included = local_interaction_users >= 0
    local_users = local_interaction_users[included]
    included_items = train_items[included]
    interaction_users = selected_factors.index_select(0, local_users)
    interaction_items = item_factors.index_select(0, included_items)
    predictions = (interaction_users * interaction_items).sum(dim=1)
    coefficients = (omega - alpha * beta) * predictions - omega
    sparse_rows = coefficients.unsqueeze(1) * interaction_users

    sparse_norm_squared = torch.zeros(
        len(selected_users), dtype=user_factors.dtype, device=device
    )
    sparse_norm_squared.index_add_(0, local_users, sparse_rows.square().sum(dim=1))
    moments = alpha * selected_factors.unsqueeze(2) * selected_factors.unsqueeze(1)
    pre_clip_norm = torch.sqrt(
        sparse_norm_squared + moments.square().sum(dim=(1, 2))
    )
    clipping_factor = _contribution_scaling_factors(
        pre_clip_norm, max_norm, contribution_rule
    )

    item_sum = torch.zeros_like(item_factors)
    item_sum.index_add_(
        0, included_items, sparse_rows * clipping_factor.index_select(0, local_users).unsqueeze(1)
    )
    moment_sum = (moments * clipping_factor[:, None, None]).sum(dim=0)
    diagnostics = PNSMFContributionDiagnostics(
        selected_users=selected_users,
        observed_rows=int(included.sum().item()),
        pre_clip_norm=pre_clip_norm,
        clipping_factor=clipping_factor,
        clipped=pre_clip_norm > max_norm,
    )
    return item_sum, moment_sum, diagnostics


def add_gaussian_noise_to_pnsmf_statistics(
    item_sum: torch.Tensor,
    moment_sum: torch.Tensor,
    *,
    max_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add isotropic Gaussian noise in the joint item-row/symmetric space."""

    if item_sum.ndim != 2 or moment_sum.ndim != 2:
        raise ValueError("Statistics must be rank-two tensors.")
    if moment_sum.shape[0] != moment_sum.shape[1]:
        raise ValueError("moment_sum must be square.")
    if item_sum.shape[1] != moment_sum.shape[0]:
        raise ValueError("Statistic dimensions do not match.")
    if max_norm <= 0.0 or noise_multiplier < 0.0:
        raise ValueError("max_norm must be positive and noise_multiplier non-negative.")
    standard_deviation = max_norm * noise_multiplier
    item_noise = torch.randn(
        item_sum.shape, dtype=item_sum.dtype, device=item_sum.device, generator=generator
    ) * standard_deviation
    packed_moment = symmetric_to_isometric_vector(moment_sum)
    moment_noise = torch.randn(
        packed_moment.shape,
        dtype=packed_moment.dtype,
        device=packed_moment.device,
        generator=generator,
    ) * standard_deviation
    return (
        item_sum + item_noise,
        moment_sum
        + isometric_vector_to_symmetric(moment_noise, moment_sum.shape[0]),
    )


def aggregate_clipped_pnsmf_item_gradients(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    selected_users: torch.Tensor,
    *,
    omega: float,
    max_norm: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    contribution_rule: str = "clip",
) -> tuple[torch.Tensor, PNSMFContributionDiagnostics]:
    """Bound and sum each user's complete dense item-gradient contribution.

    For one user, missing-item rows equal ``alpha * beta * <u,v_i> * u``
    and observed-item rows equal ``omega * (<u,v_i> - 1) * u``.  The
    implementation exploits the P-NSMF cache algebra and never materializes a
    ``[clients, items, dimension]`` tensor.
    """

    if user_factors.ndim != 2 or item_factors.ndim != 2:
        raise ValueError("Factor matrices must be rank two.")
    if user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("Factor dimensions do not match.")
    if train_users.shape != train_items.shape or train_users.ndim != 1:
        raise ValueError("Interaction indices must be aligned vectors.")
    if train_users.dtype != torch.long or train_items.dtype != torch.long:
        raise TypeError("Interaction indices must be LongTensors.")
    if selected_users.ndim != 1 or selected_users.dtype != torch.long:
        raise TypeError("selected_users must be a one-dimensional LongTensor.")
    if len(torch.unique(selected_users)) != len(selected_users):
        raise ValueError("selected_users must not contain duplicates.")
    if max_norm <= 0.0 or omega <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Norm bound and loss weights must be positive.")
    device = user_factors.device
    if any(
        tensor.device != device
        for tensor in (item_factors, train_users, train_items, selected_users)
    ):
        raise ValueError("All tensors must be on the same device.")

    selected_factors = user_factors.index_select(0, selected_users)
    lookup = torch.full(
        (user_factors.shape[0],), -1, dtype=torch.long, device=device
    )
    lookup[selected_users] = torch.arange(len(selected_users), device=device)
    local_interaction_users = lookup.index_select(0, train_users)
    included = local_interaction_users >= 0
    local_users = local_interaction_users[included]
    included_items = train_items[included]
    interaction_users = selected_factors.index_select(0, local_users)
    interaction_items = item_factors.index_select(0, included_items)
    predictions = (interaction_users * interaction_items).sum(dim=1)
    sparse_coefficients = (omega - alpha * beta) * predictions - omega
    sparse_rows = sparse_coefficients.unsqueeze(1) * interaction_users

    user_norm_squared = selected_factors.square().sum(dim=1)
    item_moment = item_factors.T @ item_factors
    baseline_scalar_squared_sum = (alpha * beta) ** 2 * torch.einsum(
        "bd,df,bf->b", selected_factors, item_moment, selected_factors
    )
    direct_norm_squared = user_norm_squared * baseline_scalar_squared_sum
    observed_baseline = alpha * beta * predictions
    observed_full = omega * (predictions - 1.0)
    observed_correction = (
        (observed_full.square() - observed_baseline.square())
        * interaction_users.square().sum(dim=1)
    )
    direct_norm_squared.index_add_(0, local_users, observed_correction)
    pre_clip_norm = torch.sqrt(direct_norm_squared.clamp_min(0.0))
    clipping_factor = _contribution_scaling_factors(
        pre_clip_norm, max_norm, contribution_rule
    )

    sparse_sum = torch.zeros_like(item_factors)
    sparse_sum.index_add_(
        0,
        included_items,
        sparse_rows * clipping_factor.index_select(0, local_users).unsqueeze(1),
    )
    moments = alpha * selected_factors.unsqueeze(2) * selected_factors.unsqueeze(1)
    moment_sum = (moments * clipping_factor[:, None, None]).sum(dim=0)
    gradient_sum = sparse_sum + beta * (item_factors @ moment_sum)
    diagnostics = PNSMFContributionDiagnostics(
        selected_users=selected_users,
        observed_rows=int(included.sum().item()),
        pre_clip_norm=pre_clip_norm,
        clipping_factor=clipping_factor,
        clipped=pre_clip_norm > max_norm,
    )
    return gradient_sum, diagnostics


def add_gaussian_noise_to_item_gradient_sum(
    gradient_sum: torch.Tensor,
    *,
    max_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to the minimal dense item-gradient query."""

    if gradient_sum.ndim != 2 or not gradient_sum.is_floating_point():
        raise ValueError("gradient_sum must be a floating-point matrix.")
    if max_norm <= 0.0 or noise_multiplier < 0.0:
        raise ValueError("max_norm must be positive and noise_multiplier non-negative.")
    return gradient_sum + torch.randn(
        gradient_sum.shape,
        dtype=gradient_sum.dtype,
        device=gradient_sum.device,
        generator=generator,
    ) * (max_norm * noise_multiplier)
