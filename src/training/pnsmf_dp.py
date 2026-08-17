"""Uniform user-level DP training step for the P-NSMF(BGD) core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models import item_second_moment
from src.privacy import (
    PNSMFContributionDiagnostics,
    add_gaussian_noise_to_item_gradient_sum,
    add_gaussian_noise_to_pnsmf_statistics,
    aggregate_clipped_pnsmf_item_gradients,
    aggregate_clipped_pnsmf_statistics,
)


def sample_poisson_clients(
    num_users: int,
    sampling_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently include each user with the declared Bernoulli probability."""
    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if not 0.0 < sampling_probability <= 1.0:
        raise ValueError("sampling_probability must lie in (0, 1].")
    return np.flatnonzero(rng.random(num_users) < sampling_probability).astype(np.int64)


@dataclass(frozen=True)
class DPPNSMFRoundDiagnostics:
    selected_clients: int
    expected_clients: float
    contribution: PNSMFContributionDiagnostics
    aggregate_signal_norm: float
    gaussian_noise_norm: float
    signal_to_noise_ratio: float
    released_item_signal_norm: float
    released_item_noise_norm: float
    released_item_signal_to_noise_ratio: float


def run_uniform_dp_pnsmf_bgd_round(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    selected_users: torch.Tensor,
    *,
    learning_rate: float,
    item_learning_rate: float | None = None,
    sampling_probability: float,
    clip_norm: float,
    noise_multiplier: float,
    omega: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    regularization: float = 0.0,
    contribution_space: str = "joint_sufficient_statistics",
    contribution_rule: str = "clip",
    user_factor_radius: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DPPNSMFRoundDiagnostics]:
    """Run one Poisson-sampled, jointly clipped P-NSMF(BGD) round.

    The noise is added to the secure-aggregation-equivalent sum.  Division by
    the expected sample count is deterministic post-processing.
    """

    if user_factors.ndim != 2 or item_factors.ndim != 2:
        raise ValueError("Factor matrices must be rank two.")
    if user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("Factor dimensions do not match.")
    if learning_rate <= 0.0 or clip_norm <= 0.0:
        raise ValueError("learning_rate and clip_norm must be positive.")
    if user_factor_radius is not None and user_factor_radius <= 0.0:
        raise ValueError("user_factor_radius must be positive when supplied.")
    if item_learning_rate is None:
        item_learning_rate = learning_rate
    if item_learning_rate <= 0.0:
        raise ValueError("item_learning_rate must be positive.")
    if not 0.0 < sampling_probability <= 1.0:
        raise ValueError("sampling_probability must be in (0, 1].")
    if noise_multiplier < 0.0 or regularization < 0.0:
        raise ValueError("noise_multiplier and regularization must be non-negative.")
    if contribution_space not in {
        "joint_sufficient_statistics",
        "direct_item_gradient",
    }:
        raise ValueError("Unknown contribution_space.")
    if contribution_rule not in {"clip", "normalize"}:
        raise ValueError("contribution_rule must be clip or normalize.")
    if selected_users.ndim != 1 or selected_users.dtype != torch.long:
        raise TypeError("selected_users must be a one-dimensional LongTensor.")
    if len(torch.unique(selected_users)) != len(selected_users):
        raise ValueError("selected_users must not contain duplicates.")
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
    coefficients = (omega - alpha * beta) * predictions - omega
    sparse_user_rows = torch.zeros_like(selected_factors)
    sparse_user_rows.index_add_(
        0, local_users, coefficients.unsqueeze(1) * interaction_items
    )
    user_cache_rows = alpha * (
        selected_factors @ item_second_moment(item_factors, beta=beta)
    )
    user_gradient = (
        2.0 * (sparse_user_rows + user_cache_rows) / item_factors.shape[0]
        + 2.0 * regularization * selected_factors
    )
    updated_selected = selected_factors - learning_rate * user_gradient
    if user_factor_radius is not None:
        row_norms = updated_selected.norm(dim=1, keepdim=True)
        scales = torch.clamp(
            user_factor_radius / row_norms.clamp_min(1e-30),
            max=1.0,
        )
        updated_selected = updated_selected * scales
    updated_users = user_factors.clone()
    updated_users.index_copy_(0, selected_users, updated_selected)

    expected_clients = sampling_probability * user_factors.shape[0]
    if contribution_space == "joint_sufficient_statistics":
        item_sum, moment_sum, contribution_diagnostics = aggregate_clipped_pnsmf_statistics(
            updated_users,
            item_factors,
            train_users,
            train_items,
            selected_users,
            omega=omega,
            max_norm=clip_norm,
            alpha=alpha,
            beta=beta,
            contribution_rule=contribution_rule,
        )
        noisy_item_sum, noisy_moment_sum = add_gaussian_noise_to_pnsmf_statistics(
            item_sum,
            moment_sum,
            max_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            generator=generator,
        )
        signal_query = torch.cat(
            (item_sum.flatten(), moment_sum.flatten())
        )
        noise_query = torch.cat(
            (
                (noisy_item_sum - item_sum).flatten(),
                (noisy_moment_sum - moment_sum).flatten(),
            )
        )
        item_data_gradient = 2.0 * (
            noisy_item_sum + beta * (item_factors @ noisy_moment_sum)
        ) / expected_clients
        released_signal = item_sum + beta * (item_factors @ moment_sum)
        released_noise = (
            noisy_item_sum + beta * (item_factors @ noisy_moment_sum)
        ) - released_signal
    else:
        gradient_sum, contribution_diagnostics = aggregate_clipped_pnsmf_item_gradients(
            updated_users,
            item_factors,
            train_users,
            train_items,
            selected_users,
            omega=omega,
            max_norm=clip_norm,
            alpha=alpha,
            beta=beta,
            contribution_rule=contribution_rule,
        )
        noisy_gradient_sum = add_gaussian_noise_to_item_gradient_sum(
            gradient_sum,
            max_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            generator=generator,
        )
        signal_query = gradient_sum.flatten()
        noise_query = (noisy_gradient_sum - gradient_sum).flatten()
        item_data_gradient = 2.0 * noisy_gradient_sum / expected_clients
        released_signal = gradient_sum
        released_noise = noisy_gradient_sum - gradient_sum
    item_gradient = item_data_gradient + 2.0 * regularization * item_factors
    updated_items = item_factors - item_learning_rate * item_gradient

    signal_norm = float(torch.linalg.vector_norm(signal_query))
    noise_norm = float(torch.linalg.vector_norm(noise_query))
    released_signal_norm = float(torch.linalg.vector_norm(released_signal))
    released_noise_norm = float(torch.linalg.vector_norm(released_noise))
    diagnostics = DPPNSMFRoundDiagnostics(
        selected_clients=len(selected_users),
        expected_clients=expected_clients,
        contribution=contribution_diagnostics,
        aggregate_signal_norm=signal_norm,
        gaussian_noise_norm=noise_norm,
        signal_to_noise_ratio=(
            signal_norm / noise_norm if noise_norm > 0.0 else float("inf")
        ),
        released_item_signal_norm=released_signal_norm,
        released_item_noise_norm=released_noise_norm,
        released_item_signal_to_noise_ratio=(
            released_signal_norm / released_noise_norm
            if released_noise_norm > 0.0
            else float("inf")
        ),
    )
    return updated_users, updated_items, diagnostics
