"""Numerical core for privacy-aware non-sampling matrix factorization.

The functions in this module implement the optimization equations from
Hu et al. (2022) without secure aggregation or differential privacy.  Keeping
the numerical core separate makes it possible to test algebraic equivalence
before protocol simulation and privacy mechanisms are added.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _validate_factors(
    user_factors: torch.Tensor, item_factors: torch.Tensor
) -> None:
    if user_factors.ndim != 2 or item_factors.ndim != 2:
        raise ValueError("User and item factors must be rank-two tensors.")
    if user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("User and item factors must share an embedding dimension.")
    if not user_factors.is_floating_point() or not item_factors.is_floating_point():
        raise ValueError("Factor tensors must use a floating-point dtype.")
    if user_factors.device != item_factors.device:
        raise ValueError("User and item factors must be on the same device.")


def item_second_moment(
    item_factors: torch.Tensor, *, beta: float = 1.0
) -> torch.Tensor:
    """Return the P-NSMF cache S^V = sum_i beta V_i^T V_i."""

    if item_factors.ndim != 2:
        raise ValueError("Item factors must be a rank-two tensor.")
    return beta * item_factors.T @ item_factors


def user_second_moment(
    user_factors: torch.Tensor, *, alpha: float = 1.0
) -> torch.Tensor:
    """Return the P-NSMF cache S^U = sum_u alpha U_u^T U_u."""

    if user_factors.ndim != 2:
        raise ValueError("User factors must be a rank-two tensor.")
    return alpha * user_factors.T @ user_factors


def weighted_nsmf_loss(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    observed: torch.Tensor,
    *,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    regularization: float = 0.0,
) -> torch.Tensor:
    """Evaluate Eq. (2) exactly on a dense Boolean observation matrix.

    This dense implementation is intended for numerical tests and small
    diagnostics.  Production training uses the cached sparse equations.
    """

    _validate_factors(user_factors, item_factors)
    expected_shape = (user_factors.shape[0], item_factors.shape[0])
    if observed.shape != expected_shape or observed.dtype != torch.bool:
        raise ValueError(
            f"Observed must be a Boolean tensor with shape {expected_shape}."
        )
    prediction = user_factors @ item_factors.T
    positive_loss = omega * (1.0 - prediction[observed]).square().sum()
    missing_loss = alpha * beta * prediction[~observed].square().sum()
    num_users, num_items = observed.shape
    penalty = regularization * (
        num_items * user_factors.square().sum()
        + num_users * item_factors.square().sum()
    )
    return positive_loss + missing_loss + penalty


def user_bgd_gradient(
    user_factor: torch.Tensor,
    item_factors: torch.Tensor,
    positive_items: torch.Tensor,
    *,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    regularization: float = 0.0,
) -> torch.Tensor:
    """Return the normalized user gradient used by Algorithm 4.

    The paper writes the full gradient in Eq. (8) and then multiplies it by
    gamma / m in Eq. (9).  This function returns that full gradient divided by
    m, which is equivalent to the public reference implementation.
    """

    if user_factor.ndim != 1 or item_factors.ndim != 2:
        raise ValueError("Expected one user vector and a rank-two item matrix.")
    if user_factor.shape[0] != item_factors.shape[1]:
        raise ValueError("Embedding dimensions do not match.")
    if positive_items.ndim != 1 or positive_items.dtype != torch.long:
        raise ValueError("Positive item indices must be a one-dimensional LongTensor.")
    selected = item_factors.index_select(0, positive_items)
    predictions = selected @ user_factor
    coefficients = (omega - alpha * beta) * predictions - omega
    observed_term = coefficients @ selected
    cache_term = alpha * (user_factor @ item_second_moment(item_factors, beta=beta))
    return (
        2.0 * (observed_term + cache_term) / item_factors.shape[0]
        + 2.0 * regularization * user_factor
    )


def client_item_statistics(
    user_factor: torch.Tensor,
    item_factors: torch.Tensor,
    positive_items: torch.Tensor,
    *,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse a(u, i) rows and the per-user S^U contribution."""

    if positive_items.ndim != 1 or positive_items.dtype != torch.long:
        raise ValueError("Positive item indices must be a one-dimensional LongTensor.")
    selected = item_factors.index_select(0, positive_items)
    predictions = selected @ user_factor
    coefficients = (omega - alpha * beta) * predictions - omega
    sparse_rows = coefficients.unsqueeze(1) * user_factor.unsqueeze(0)
    second_moment = alpha * torch.outer(user_factor, user_factor)
    return sparse_rows, second_moment


def aggregate_client_item_statistics(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_by_user: Sequence[np.ndarray],
    *,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate all clients' sparse P-NSMF sufficient statistics."""

    _validate_factors(user_factors, item_factors)
    if len(train_by_user) != user_factors.shape[0]:
        raise ValueError("train_by_user must contain one item array per user.")
    item_rows = torch.zeros_like(item_factors)
    summed_user_second_moment = torch.zeros(
        (user_factors.shape[1], user_factors.shape[1]),
        dtype=user_factors.dtype,
        device=user_factors.device,
    )
    for user_index, positive_array in enumerate(train_by_user):
        positive_items = torch.as_tensor(
            positive_array, dtype=torch.long, device=user_factors.device
        )
        rows, second_moment = client_item_statistics(
            user_factors[user_index],
            item_factors,
            positive_items,
            omega=omega,
            alpha=alpha,
            beta=beta,
        )
        item_rows.index_add_(0, positive_items, rows)
        summed_user_second_moment.add_(second_moment)
    return item_rows, summed_user_second_moment


def item_bgd_gradient(
    item_factors: torch.Tensor,
    aggregated_sparse_rows: torch.Tensor,
    summed_user_second_moment: torch.Tensor,
    *,
    num_users: int,
    beta: float = 1.0,
    regularization: float = 0.0,
) -> torch.Tensor:
    """Return normalized item gradients from aggregated client statistics."""

    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if aggregated_sparse_rows.shape != item_factors.shape:
        raise ValueError("Sparse item statistics must match the item-factor shape.")
    expected_moment_shape = (item_factors.shape[1], item_factors.shape[1])
    if summed_user_second_moment.shape != expected_moment_shape:
        raise ValueError(
            f"Summed user second moment must have shape {expected_moment_shape}."
        )
    data_gradient = aggregated_sparse_rows + beta * (
        item_factors @ summed_user_second_moment
    )
    return (
        2.0 * data_gradient / num_users
        + 2.0 * regularization * item_factors
    )


def item_subproblem_hessian_trace(
    user_factors: torch.Tensor,
    train_users: torch.Tensor,
    *,
    num_items: int,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    regularization: float = 0.0,
) -> torch.Tensor:
    """Return the trace of the normalized fixed-user item Hessian.

    The normalized objective is ``weighted_nsmf_loss / num_users``. User
    factors are fixed, so the item subproblem is quadratic and its Hessian is
    independent of the item-factor values.
    """

    if user_factors.ndim != 2 or not user_factors.is_floating_point():
        raise ValueError("user_factors must be a floating-point rank-two tensor.")
    if train_users.ndim != 1 or train_users.dtype != torch.long:
        raise TypeError("train_users must be a one-dimensional LongTensor.")
    if train_users.device != user_factors.device:
        raise ValueError("train_users and user_factors must share a device.")
    if num_items <= 0:
        raise ValueError("num_items must be positive.")
    if omega < 0.0 or alpha < 0.0 or beta < 0.0 or regularization < 0.0:
        raise ValueError("weights and regularization must be non-negative.")
    if omega < alpha * beta:
        raise ValueError("omega must be at least alpha * beta for a PSD decomposition.")
    num_users, embedding_dim = user_factors.shape
    if num_users <= 0:
        raise ValueError("At least one user is required.")
    if len(train_users) and (
        int(train_users.min()) < 0 or int(train_users.max()) >= num_users
    ):
        raise IndexError("train_users contains an out-of-range user index.")

    user_squared_norms = user_factors.square().sum(dim=1)
    all_user_term = num_items * alpha * beta * user_squared_norms.sum()
    observed_term = (omega - alpha * beta) * user_squared_norms.index_select(
        0, train_users
    ).sum()
    return (
        2.0 * (all_user_term + observed_term) / num_users
        + 2.0 * regularization * num_items * embedding_dim
    )


def pnsmf_bgd_round(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_by_user: Sequence[np.ndarray],
    *,
    learning_rate: float,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    regularization: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one non-secure P-NSMF(BGD) round and return new factor tensors."""

    _validate_factors(user_factors, item_factors)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if len(train_by_user) != user_factors.shape[0]:
        raise ValueError("train_by_user must contain one item array per user.")

    updated_users = user_factors.clone()
    for user_index, positive_array in enumerate(train_by_user):
        positive_items = torch.as_tensor(
            positive_array, dtype=torch.long, device=user_factors.device
        )
        gradient = user_bgd_gradient(
            user_factors[user_index],
            item_factors,
            positive_items,
            omega=omega,
            alpha=alpha,
            beta=beta,
            regularization=regularization,
        )
        updated_users[user_index] = user_factors[user_index] - learning_rate * gradient

    sparse_rows, summed_user_second_moment = aggregate_client_item_statistics(
        updated_users,
        item_factors,
        train_by_user,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )
    item_gradient = item_bgd_gradient(
        item_factors,
        sparse_rows,
        summed_user_second_moment,
        num_users=user_factors.shape[0],
        beta=beta,
        regularization=regularization,
    )
    updated_items = item_factors - learning_rate * item_gradient
    return updated_users, updated_items


def pnsmf_bgd_round_indexed(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    *,
    learning_rate: float,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    regularization: float = 0.0,
    user_factor_radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized P-NSMF(BGD) round for full-scale experiments.

    ``train_users`` and ``train_items`` contain aligned observed-interaction
    indices.  The result is algebraically identical to :func:`pnsmf_bgd_round`
    but avoids one Python call per client.
    """

    _validate_factors(user_factors, item_factors)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if user_factor_radius is not None and user_factor_radius <= 0.0:
        raise ValueError("user_factor_radius must be positive when supplied.")
    if (
        train_users.ndim != 1
        or train_items.ndim != 1
        or train_users.dtype != torch.long
        or train_items.dtype != torch.long
    ):
        raise ValueError("Interaction indices must be one-dimensional LongTensors.")
    if train_users.shape != train_items.shape:
        raise ValueError("User and item interaction indices must have equal shapes.")
    if train_users.device != user_factors.device or train_items.device != user_factors.device:
        raise ValueError("Interaction indices and factors must be on the same device.")

    interaction_users = user_factors.index_select(0, train_users)
    interaction_items = item_factors.index_select(0, train_items)
    predictions = (interaction_users * interaction_items).sum(dim=1)
    coefficients = (omega - alpha * beta) * predictions - omega

    sparse_user_rows = torch.zeros_like(user_factors)
    sparse_user_rows.index_add_(
        0, train_users, coefficients.unsqueeze(1) * interaction_items
    )
    user_cache_rows = alpha * (
        user_factors @ item_second_moment(item_factors, beta=beta)
    )
    user_gradient = (
        2.0 * (sparse_user_rows + user_cache_rows) / item_factors.shape[0]
        + 2.0 * regularization * user_factors
    )
    updated_users = user_factors - learning_rate * user_gradient
    if user_factor_radius is not None:
        row_norms = updated_users.norm(dim=1, keepdim=True)
        scales = torch.clamp(
            user_factor_radius / row_norms.clamp_min(1e-30),
            max=1.0,
        )
        updated_users = updated_users * scales

    updated_interaction_users = updated_users.index_select(0, train_users)
    updated_predictions = (updated_interaction_users * interaction_items).sum(dim=1)
    updated_coefficients = (
        (omega - alpha * beta) * updated_predictions - omega
    )
    sparse_item_rows = torch.zeros_like(item_factors)
    sparse_item_rows.index_add_(
        0,
        train_items,
        updated_coefficients.unsqueeze(1) * updated_interaction_users,
    )
    summed_user_second_moment = user_second_moment(
        updated_users, alpha=alpha
    )
    item_gradient = item_bgd_gradient(
        item_factors,
        sparse_item_rows,
        summed_user_second_moment,
        num_users=user_factors.shape[0],
        beta=beta,
        regularization=regularization,
    )
    updated_items = item_factors - learning_rate * item_gradient
    return updated_users, updated_items


def _batched_normal_equation_solve(
    matrices: torch.Tensor,
    right_hand_sides: torch.Tensor,
    *,
    regularization: float,
) -> torch.Tensor:
    """Solve symmetric ALS normal equations, matching ``pinv`` when singular."""

    if regularization > 0.0:
        return torch.linalg.solve(
            matrices, right_hand_sides.unsqueeze(-1)
        ).squeeze(-1)
    return (
        torch.linalg.pinv(matrices) @ right_hand_sides.unsqueeze(-1)
    ).squeeze(-1)


def pnsmf_als_round_indexed(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    *,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    regularization: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one vectorized WMF/P-NSMF ALS round.

    The update is algebraically equivalent to the public WMF(ALS) and
    P-NSMF(ALS) implementations.  User factors are updated first; the item
    update therefore uses the newly updated user factors.
    """

    _validate_factors(user_factors, item_factors)
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative.")
    if (
        train_users.ndim != 1
        or train_items.ndim != 1
        or train_users.dtype != torch.long
        or train_items.dtype != torch.long
    ):
        raise ValueError("Interaction indices must be one-dimensional LongTensors.")
    if train_users.shape != train_items.shape:
        raise ValueError("User and item interaction indices must have equal shapes.")
    if train_users.device != user_factors.device or train_items.device != user_factors.device:
        raise ValueError("Interaction indices and factors must be on the same device.")
    if omega <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("omega, alpha, and beta must be positive.")

    num_users, dimension = user_factors.shape
    num_items = item_factors.shape[0]
    dtype = user_factors.dtype
    device = user_factors.device
    identity = torch.eye(dimension, dtype=dtype, device=device)
    observed_adjustment = omega - alpha * beta

    interaction_items = item_factors.index_select(0, train_items)
    user_rhs = torch.zeros_like(user_factors)
    user_rhs.index_add_(0, train_users, omega * interaction_items)
    user_observed_moments = torch.zeros(
        (num_users, dimension, dimension), dtype=dtype, device=device
    )
    user_observed_moments.index_add_(
        0,
        train_users,
        interaction_items.unsqueeze(2) * interaction_items.unsqueeze(1),
    )
    user_systems = (
        alpha * item_second_moment(item_factors, beta=beta).unsqueeze(0)
        + observed_adjustment * user_observed_moments
        + num_items * regularization * identity.unsqueeze(0)
    )
    updated_users = _batched_normal_equation_solve(
        user_systems, user_rhs, regularization=regularization
    )

    interaction_users = updated_users.index_select(0, train_users)
    item_rhs = torch.zeros_like(item_factors)
    item_rhs.index_add_(0, train_items, omega * interaction_users)
    item_observed_moments = torch.zeros(
        (num_items, dimension, dimension), dtype=dtype, device=device
    )
    item_observed_moments.index_add_(
        0,
        train_items,
        interaction_users.unsqueeze(2) * interaction_users.unsqueeze(1),
    )
    item_systems = (
        beta * user_second_moment(updated_users, alpha=alpha).unsqueeze(0)
        + observed_adjustment * item_observed_moments
        + num_users * regularization * identity.unsqueeze(0)
    )
    updated_items = _batched_normal_equation_solve(
        item_systems, item_rhs, regularization=regularization
    )
    return updated_users, updated_items
