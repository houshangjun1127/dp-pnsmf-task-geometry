from __future__ import annotations

import torch

from src.models import dpals_implicit_loss_indexed, random_orthonormal_item_factors
from src.training import dpals_private_item_update, dpals_user_update


def _indices() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0, 0, 1, 2, 2], dtype=torch.long),
        torch.tensor([0, 2, 1, 0, 3], dtype=torch.long),
    )


def test_sparse_dpals_loss_matches_dense_definition() -> None:
    generator = torch.Generator().manual_seed(9)
    users = torch.randn((3, 2), dtype=torch.float64, generator=generator)
    items = torch.randn((4, 2), dtype=torch.float64, generator=generator)
    train_users, train_items = _indices()
    observed = torch.zeros((3, 4), dtype=torch.bool)
    observed[train_users, train_items] = True
    prediction = users @ items.T
    expected = (
        (1.0 - prediction[observed]).square().sum()
        + 0.1 * prediction.square().sum()
        + 0.01 * (users.square().sum() + items.square().sum())
    )
    actual = dpals_implicit_loss_indexed(
        users,
        items,
        train_users,
        train_items,
        implicit_penalty=0.1,
        regularization=0.01,
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-10)


def test_dpals_user_update_matches_normal_equation_and_clips() -> None:
    train_users, train_items = _indices()
    items = random_orthonormal_item_factors(
        4, 2, generator=torch.Generator().manual_seed(10)
    )
    unbounded, _ = dpals_user_update(
        items,
        train_users,
        train_items,
        num_users=3,
        implicit_penalty=0.1,
        regularization=0.01,
        clip_radius=None,
    )
    selected = items.index_select(0, torch.tensor([0, 2]))
    system = (
        selected.T @ selected
        + 0.1 * (items.T @ items)
        + 0.01 * torch.eye(2, dtype=torch.float64)
    )
    expected_first = torch.linalg.solve(system, selected.sum(dim=0))
    assert torch.allclose(unbounded[0], expected_first, atol=1e-12, rtol=1e-10)

    clipped, diagnostics = dpals_user_update(
        items,
        train_users,
        train_items,
        num_users=3,
        implicit_penalty=0.1,
        regularization=0.01,
        clip_radius=0.25,
    )
    assert torch.all(clipped.norm(dim=1) <= 0.25 * (1.0 + 1e-12))
    assert torch.any(diagnostics.clipping_factor < 1.0)


def test_no_noise_dpals_item_update_is_finite_and_whitened() -> None:
    train_users, train_items = _indices()
    items = random_orthonormal_item_factors(
        4, 2, generator=torch.Generator().manual_seed(11)
    )
    users, _ = dpals_user_update(
        items,
        train_users,
        train_items,
        num_users=3,
        implicit_penalty=0.1,
        regularization=0.1,
        clip_radius=1.0,
    )
    updated, diagnostics = dpals_private_item_update(
        users,
        train_users,
        train_items,
        num_items=4,
        implicit_penalty=0.1,
        regularization=0.1,
        user_clip_radius=1.0,
        entry_clip=1.0,
        noise_multiplier=0.0,
        generator=torch.Generator().manual_seed(12),
    )
    assert torch.isfinite(updated).all()
    assert torch.allclose(
        updated.T @ updated, torch.eye(2, dtype=torch.float64), atol=1e-10
    )
    assert diagnostics.item_gram_noise_norm == 0.0
    assert diagnostics.item_vector_noise_norm == 0.0
    assert diagnostics.shared_gram_noise_norm == 0.0


def test_private_dpals_item_update_is_seed_deterministic() -> None:
    train_users, train_items = _indices()
    users = torch.tensor(
        [[0.2, 0.1], [0.1, -0.2], [-0.1, 0.15]], dtype=torch.float64
    )
    kwargs = dict(
        num_items=4,
        implicit_penalty=0.1,
        regularization=0.1,
        user_clip_radius=1.0,
        entry_clip=1.0,
        noise_multiplier=2.0,
    )
    first, _ = dpals_private_item_update(
        users,
        train_users,
        train_items,
        generator=torch.Generator().manual_seed(13),
        **kwargs,
    )
    second, _ = dpals_private_item_update(
        users,
        train_users,
        train_items,
        generator=torch.Generator().manual_seed(13),
        **kwargs,
    )
    assert torch.equal(first, second)
