import math

import numpy as np
import torch

from src.models import (
    aggregate_client_item_statistics,
    item_bgd_gradient,
    weighted_nsmf_loss,
)


def _problem() -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[np.ndarray, ...],
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(20260802)
    users = torch.randn((4, 3), dtype=torch.float64, generator=generator) * 0.2
    items = torch.randn((5, 3), dtype=torch.float64, generator=generator) * 0.2
    train_by_user = (
        np.array([0, 2], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0, 3, 4], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
    )
    observed = torch.zeros((len(train_by_user), len(items)), dtype=torch.bool)
    for user_index, item_indices in enumerate(train_by_user):
        observed[user_index, item_indices] = True
    return users, items, train_by_user, observed


def test_data_loss_and_predictions_are_gauge_invariant() -> None:
    users, items, _, observed = _problem()
    scale = 0.37
    scaled_users = scale * users
    scaled_items = items / scale

    assert torch.allclose(
        scaled_users @ scaled_items.T,
        users @ items.T,
        atol=1e-13,
        rtol=1e-12,
    )
    original_loss = weighted_nsmf_loss(
        users, items, observed, omega=4.0, alpha=0.2, beta=3.0
    )
    scaled_loss = weighted_nsmf_loss(
        scaled_users,
        scaled_items,
        observed,
        omega=4.0,
        alpha=0.2,
        beta=3.0,
    )
    assert torch.allclose(scaled_loss, original_loss, atol=1e-12, rtol=1e-12)


def test_direct_item_gradient_and_noisy_update_cancel_under_gauge() -> None:
    users, items, train_by_user, _ = _problem()
    scale = 0.41
    learning_rate = 0.07
    regularization = 0.0

    sparse, moment = aggregate_client_item_statistics(
        users, items, train_by_user, omega=4.0, alpha=0.2, beta=3.0
    )
    gradient = item_bgd_gradient(
        items,
        sparse,
        moment,
        num_users=len(users),
        beta=3.0,
        regularization=regularization,
    )

    scaled_sparse, scaled_moment = aggregate_client_item_statistics(
        scale * users,
        items / scale,
        train_by_user,
        omega=4.0,
        alpha=0.2,
        beta=3.0,
    )
    scaled_gradient = item_bgd_gradient(
        items / scale,
        scaled_sparse,
        scaled_moment,
        num_users=len(users),
        beta=3.0,
        regularization=regularization,
    )
    assert torch.allclose(
        scaled_gradient, scale * gradient, atol=1e-12, rtol=1e-11
    )

    generator = torch.Generator().manual_seed(20260803)
    noise = torch.randn(items.shape, dtype=torch.float64, generator=generator)
    original_updated = items - learning_rate * (gradient + noise)
    scaled_updated = items / scale - (learning_rate / scale**2) * (
        scaled_gradient + scale * noise
    )
    mapped_updated = scale * scaled_updated
    assert torch.allclose(
        mapped_updated, original_updated, atol=1e-12, rtol=1e-11
    )


def test_joint_statistics_scale_by_different_powers_but_reconstruct_signal() -> None:
    users, items, train_by_user, _ = _problem()
    scale = 0.29
    sparse, moment = aggregate_client_item_statistics(
        users, items, train_by_user, omega=4.0, alpha=0.2, beta=3.0
    )
    scaled_sparse, scaled_moment = aggregate_client_item_statistics(
        scale * users,
        items / scale,
        train_by_user,
        omega=4.0,
        alpha=0.2,
        beta=3.0,
    )

    assert torch.allclose(
        scaled_sparse, scale * sparse, atol=1e-12, rtol=1e-11
    )
    assert torch.allclose(
        scaled_moment, scale**2 * moment, atol=1e-12, rtol=1e-11
    )
    reconstructed = sparse + 3.0 * (items @ moment)
    scaled_reconstructed = scaled_sparse + 3.0 * (
        (items / scale) @ scaled_moment
    )
    assert torch.allclose(
        scaled_reconstructed,
        scale * reconstructed,
        atol=1e-12,
        rtol=1e-11,
    )


def test_l2_regularizer_has_closed_form_balanced_scale() -> None:
    users, items, _, _ = _problem()
    num_users = users.shape[0]
    num_items = items.shape[0]
    user_energy = float(users.square().sum())
    item_energy = float(items.square().sum())
    optimum = (
        num_users * item_energy / (num_items * user_energy)
    ) ** 0.25

    def penalty(scale: float) -> float:
        return (
            num_items * scale**2 * user_energy
            + num_users * item_energy / scale**2
        )

    assert math.isclose(
        num_items * optimum**2 * user_energy,
        num_users * item_energy / optimum**2,
        rel_tol=1e-12,
    )
    for multiplier in (0.5, 0.8, 1.2, 2.0):
        assert penalty(optimum) < penalty(optimum * multiplier)


def test_two_block_noise_proxy_has_finite_balanced_scale() -> None:
    sparse_radius = 2.3
    moment_radius = 0.7
    sparse_dimension = 60.0
    moment_reconstruction_energy = 4.2
    optimum = (
        sparse_radius**2
        * moment_reconstruction_energy
        / (moment_radius**2 * sparse_dimension)
    ) ** 0.25

    def proxy(scale: float) -> float:
        return (
            sparse_radius**2 + scale**2 * moment_radius**2
        ) * (
            sparse_dimension
            + moment_reconstruction_energy / scale**2
        )

    for multiplier in (0.5, 0.8, 1.2, 2.0):
        assert proxy(optimum) < proxy(optimum * multiplier)
