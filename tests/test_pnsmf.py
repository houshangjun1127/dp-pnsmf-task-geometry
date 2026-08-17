import math

import numpy as np
import torch

from src.models import (
    aggregate_client_item_statistics,
    client_item_statistics,
    item_bgd_gradient,
    item_subproblem_hessian_trace,
    pnsmf_bgd_round,
    pnsmf_bgd_round_indexed,
    pnsmf_als_round_indexed,
    user_bgd_gradient,
    weighted_nsmf_loss,
)
from src.privacy import expected_quadratic_loss_inflation


def _problem():
    torch.manual_seed(11)
    users = torch.randn(3, 2, dtype=torch.float64) * 0.1
    items = torch.randn(4, 2, dtype=torch.float64) * 0.1
    train_by_user = (
        np.array([0, 2], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
    )
    observed = torch.zeros((3, 4), dtype=torch.bool)
    for user_index, item_indices in enumerate(train_by_user):
        observed[user_index, item_indices] = True
    return users, items, train_by_user, observed


def test_item_hessian_trace_matches_dense_autograd() -> None:
    users, items, _, observed = _problem()
    train_users = observed.nonzero(as_tuple=False)[:, 0]
    regularization = 0.01

    def normalized_item_objective(flat_items: torch.Tensor) -> torch.Tensor:
        return weighted_nsmf_loss(
            users,
            flat_items.reshape_as(items),
            observed,
            omega=4.0,
            regularization=regularization,
        ) / users.shape[0]

    dense_hessian = torch.autograd.functional.hessian(
        normalized_item_objective, items.flatten()
    )
    analytic_trace = item_subproblem_hessian_trace(
        users,
        train_users,
        num_items=items.shape[0],
        omega=4.0,
        regularization=regularization,
    )
    assert torch.allclose(analytic_trace, dense_hessian.diag().sum(), atol=1e-12)


def test_quadratic_noise_loss_inflation_matches_monte_carlo() -> None:
    users, items, _, observed = _problem()
    train_users = observed.nonzero(as_tuple=False)[:, 0]
    regularization = 0.01
    update_noise_variance = 0.0025
    hessian_trace = float(
        item_subproblem_hessian_trace(
            users,
            train_users,
            num_items=items.shape[0],
            omega=4.0,
            regularization=regularization,
        )
    )
    expected = expected_quadratic_loss_inflation(
        hessian_trace=hessian_trace,
        update_noise_variance=update_noise_variance,
    )

    half_trials = 20_000
    generator = torch.Generator().manual_seed(20260804)
    half_noise = torch.randn(
        (half_trials, *items.shape), dtype=torch.float64, generator=generator
    ) * update_noise_variance**0.5
    noise = torch.cat((half_noise, -half_noise), dim=0)
    perturbed_items = items.unsqueeze(0) + noise
    predictions = torch.einsum("ud,tmd->tum", users, perturbed_items)
    positive = 4.0 * (1.0 - predictions[:, observed]).square().sum(dim=1)
    missing = predictions[:, ~observed].square().sum(dim=1)
    penalty = regularization * (
        items.shape[0] * users.square().sum()
        + users.shape[0] * perturbed_items.square().sum(dim=(1, 2))
    )
    perturbed_loss = (positive + missing + penalty) / users.shape[0]
    baseline = float(
        weighted_nsmf_loss(
            users,
            items,
            observed,
            omega=4.0,
            regularization=regularization,
        )
        / users.shape[0]
    )
    empirical = float(perturbed_loss.mean()) - baseline
    assert math.isclose(empirical, expected, rel_tol=0.025)


def test_spherical_sensitivity_gaussian_cost_lower_bound() -> None:
    """Anisotropic covariance cannot beat the isotropic full-ball bound."""

    generator = torch.Generator().manual_seed(20260802)
    dimension = 9
    hessian_factor = torch.randn(
        (dimension, dimension), dtype=torch.float64, generator=generator
    )
    hessian = hessian_factor.T @ hessian_factor
    isotropic_variance = 0.04

    covariance_factor = torch.randn(
        (dimension, 4), dtype=torch.float64, generator=generator
    )
    covariance = (
        isotropic_variance * torch.eye(dimension, dtype=torch.float64)
        + covariance_factor @ covariance_factor.T
    )

    anisotropic_cost = 0.5 * torch.trace(hessian @ covariance)
    isotropic_lower_bound = (
        0.5 * isotropic_variance * torch.trace(hessian)
    )
    assert anisotropic_cost >= isotropic_lower_bound


def test_cached_user_gradient_matches_dense_objective_autograd() -> None:
    users, items, train_by_user, observed = _problem()
    users = users.requires_grad_(True)
    items = items.requires_grad_(True)
    loss = weighted_nsmf_loss(
        users,
        items,
        observed,
        omega=4.0,
        regularization=0.01,
    )
    dense_user_gradient, dense_item_gradient = torch.autograd.grad(
        loss, (users, items)
    )

    for user_index, positive_array in enumerate(train_by_user):
        positive_items = torch.as_tensor(positive_array, dtype=torch.long)
        cached = user_bgd_gradient(
            users.detach()[user_index],
            items.detach(),
            positive_items,
            omega=4.0,
            regularization=0.01,
        )
        assert torch.allclose(
            cached,
            dense_user_gradient[user_index] / items.shape[0],
            atol=1e-12,
            rtol=1e-10,
        )

    sparse_rows, summed_moment = aggregate_client_item_statistics(
        users.detach(),
        items.detach(),
        train_by_user,
        omega=4.0,
    )
    cached_item_gradient = item_bgd_gradient(
        items.detach(),
        sparse_rows,
        summed_moment,
        num_users=users.shape[0],
        regularization=0.01,
    )
    assert torch.allclose(
        cached_item_gradient,
        dense_item_gradient / users.shape[0],
        atol=1e-12,
        rtol=1e-10,
    )


def test_aggregated_client_statistics_equal_direct_item_gradient() -> None:
    users, items, train_by_user, observed = _problem()
    users = users.requires_grad_(True)
    items = items.requires_grad_(True)
    loss = weighted_nsmf_loss(users, items, observed, omega=3.0)
    (dense_item_gradient,) = torch.autograd.grad(loss, (items,))

    sparse_rows, summed_moment = aggregate_client_item_statistics(
        users.detach(),
        items.detach(),
        train_by_user,
        omega=3.0,
    )
    federated_gradient = item_bgd_gradient(
        items.detach(),
        sparse_rows,
        summed_moment,
        num_users=users.shape[0],
    )
    assert torch.allclose(
        federated_gradient,
        dense_item_gradient / users.shape[0],
        atol=1e-12,
        rtol=1e-10,
    )


def test_concealing_zero_rows_do_not_change_client_statistics() -> None:
    users, items, train_by_user, _ = _problem()
    real_items = torch.as_tensor(train_by_user[0], dtype=torch.long)
    real_rows, second_moment = client_item_statistics(
        users[0], items, real_items, omega=4.0
    )

    upload_items = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    concealed_rows = torch.zeros(
        (len(upload_items), items.shape[1]), dtype=items.dtype
    )
    real_positions = torch.tensor([0, 2], dtype=torch.long)
    concealed_rows[real_positions] = real_rows

    direct_aggregate = torch.zeros_like(items)
    direct_aggregate.index_add_(0, real_items, real_rows)
    concealed_aggregate = torch.zeros_like(items)
    concealed_aggregate.index_add_(0, upload_items, concealed_rows)

    assert torch.equal(direct_aggregate, concealed_aggregate)
    assert torch.equal(second_moment, torch.outer(users[0], users[0]))


def test_one_round_preserves_inputs_and_returns_finite_updates() -> None:
    users, items, train_by_user, _ = _problem()
    original_users = users.clone()
    original_items = items.clone()
    updated_users, updated_items = pnsmf_bgd_round(
        users,
        items,
        train_by_user,
        learning_rate=0.1,
        omega=4.0,
        regularization=0.01,
    )
    assert torch.equal(users, original_users)
    assert torch.equal(items, original_items)
    assert torch.isfinite(updated_users).all()
    assert torch.isfinite(updated_items).all()
    assert not torch.equal(updated_users, users)
    assert not torch.equal(updated_items, items)


def test_vectorized_round_matches_client_loop_round() -> None:
    users, items, train_by_user, _ = _problem()
    train_users = torch.tensor(
        [
            user_index
            for user_index, item_indices in enumerate(train_by_user)
            for _ in item_indices
        ],
        dtype=torch.long,
    )
    train_items = torch.tensor(
        [item for item_indices in train_by_user for item in item_indices],
        dtype=torch.long,
    )
    loop_users, loop_items = pnsmf_bgd_round(
        users,
        items,
        train_by_user,
        learning_rate=0.1,
        omega=4.0,
        regularization=0.01,
    )
    vector_users, vector_items = pnsmf_bgd_round_indexed(
        users,
        items,
        train_users,
        train_items,
        learning_rate=0.1,
        omega=4.0,
        regularization=0.01,
    )
    assert torch.allclose(vector_users, loop_users, atol=1e-12, rtol=1e-10)
    assert torch.allclose(vector_items, loop_items, atol=1e-12, rtol=1e-10)


def test_vectorized_round_projects_user_factors_before_item_update() -> None:
    users, items, train_by_user, _ = _problem()
    train_users = torch.tensor(
        [u for u, item_indices in enumerate(train_by_user) for _ in item_indices],
        dtype=torch.long,
    )
    train_items = torch.tensor(
        [item for item_indices in train_by_user for item in item_indices],
        dtype=torch.long,
    )
    radius = 0.05
    projected_users, projected_items = pnsmf_bgd_round_indexed(
        users,
        items,
        train_users,
        train_items,
        learning_rate=0.1,
        omega=4.0,
        regularization=0.01,
        user_factor_radius=radius,
    )

    assert torch.all(projected_users.norm(dim=1) <= radius * (1.0 + 1e-12))
    assert torch.isfinite(projected_items).all()


def test_als_round_matches_direct_normal_equations() -> None:
    users, items, train_by_user, _ = _problem()
    train_users = torch.tensor(
        [u for u, item_indices in enumerate(train_by_user) for _ in item_indices],
        dtype=torch.long,
    )
    train_items = torch.tensor(
        [item for item_indices in train_by_user for item in item_indices],
        dtype=torch.long,
    )
    updated_users, updated_items = pnsmf_als_round_indexed(
        users,
        items,
        train_users,
        train_items,
        omega=4.0,
        regularization=0.01,
    )

    identity = torch.eye(items.shape[1], dtype=torch.float64)
    expected_users = []
    item_moment = items.T @ items
    for positive_array in train_by_user:
        selected = items[torch.as_tensor(positive_array)]
        system = item_moment + 3.0 * selected.T @ selected + items.shape[0] * 0.01 * identity
        rhs = 4.0 * selected.sum(dim=0)
        expected_users.append(torch.linalg.solve(system, rhs))
    expected_users = torch.stack(expected_users)

    expected_items = []
    user_moment = expected_users.T @ expected_users
    for item_index in range(items.shape[0]):
        positive_users = torch.nonzero(train_items == item_index).flatten()
        selected = expected_users.index_select(
            0, train_users.index_select(0, positive_users)
        )
        system = user_moment + 3.0 * selected.T @ selected + users.shape[0] * 0.01 * identity
        rhs = 4.0 * selected.sum(dim=0)
        expected_items.append(torch.linalg.solve(system, rhs))
    expected_items = torch.stack(expected_items)

    assert torch.allclose(updated_users, expected_users, atol=1e-12, rtol=1e-10)
    assert torch.allclose(updated_items, expected_items, atol=1e-12, rtol=1e-10)


def test_als_round_does_not_increase_weighted_objective() -> None:
    users, items, train_by_user, observed = _problem()
    train_users = torch.tensor(
        [u for u, item_indices in enumerate(train_by_user) for _ in item_indices],
        dtype=torch.long,
    )
    train_items = torch.tensor(
        [item for item_indices in train_by_user for item in item_indices],
        dtype=torch.long,
    )
    before = weighted_nsmf_loss(
        users, items, observed, omega=4.0, regularization=0.01
    )
    updated_users, updated_items = pnsmf_als_round_indexed(
        users,
        items,
        train_users,
        train_items,
        omega=4.0,
        regularization=0.01,
    )
    after = weighted_nsmf_loss(
        updated_users, updated_items, observed, omega=4.0, regularization=0.01
    )
    assert after <= before + 1e-12
