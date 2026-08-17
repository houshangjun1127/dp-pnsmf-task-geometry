import numpy as np
import torch

from src.models import pnsmf_bgd_round_indexed
from src.training import run_uniform_dp_pnsmf_bgd_round, server_ema_delta


def _problem():
    torch.manual_seed(19)
    users = torch.randn(3, 2, dtype=torch.float64) * 0.1
    items = torch.randn(4, 2, dtype=torch.float64) * 0.1
    train_by_user = (
        np.array([0, 2], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
    )
    train_users = torch.tensor(
        [u for u, item_indices in enumerate(train_by_user) for _ in item_indices],
        dtype=torch.long,
    )
    train_items = torch.tensor(
        [item for item_indices in train_by_user for item in item_indices],
        dtype=torch.long,
    )
    return users, items, train_users, train_items


def test_full_participation_no_noise_large_bound_matches_nonprivate_round() -> None:
    users, items, train_users, train_items = _problem()
    expected_users, expected_items = pnsmf_bgd_round_indexed(
        users,
        items,
        train_users,
        train_items,
        learning_rate=0.1,
        omega=4.0,
        regularization=0.01,
    )
    actual_users, actual_items, diagnostics = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        torch.arange(len(users)),
        learning_rate=0.1,
        sampling_probability=1.0,
        clip_norm=1e6,
        noise_multiplier=0.0,
        omega=4.0,
        regularization=0.01,
    )
    assert torch.allclose(actual_users, expected_users, atol=1e-12, rtol=1e-10)
    assert torch.allclose(actual_items, expected_items, atol=1e-12, rtol=1e-10)
    assert not diagnostics.contribution.clipped.any()
    assert diagnostics.gaussian_noise_norm == 0.0


def test_direct_gradient_full_participation_matches_nonprivate_round() -> None:
    users, items, train_users, train_items = _problem()
    expected_users, expected_items = pnsmf_bgd_round_indexed(
        users,
        items,
        train_users,
        train_items,
        learning_rate=0.1,
        omega=4.0,
        regularization=0.01,
    )
    actual_users, actual_items, diagnostics = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        torch.arange(len(users)),
        learning_rate=0.1,
        sampling_probability=1.0,
        clip_norm=1e6,
        noise_multiplier=0.0,
        omega=4.0,
        regularization=0.01,
        contribution_space="direct_item_gradient",
    )
    assert torch.allclose(actual_users, expected_users, atol=1e-12, rtol=1e-10)
    assert torch.allclose(actual_items, expected_items, atol=1e-12, rtol=1e-10)
    assert not diagnostics.contribution.clipped.any()


def test_separate_item_learning_rate_changes_only_item_update() -> None:
    users, items, train_users, train_items = _problem()
    kwargs = dict(
        learning_rate=0.1,
        sampling_probability=1.0,
        clip_norm=1e6,
        noise_multiplier=0.0,
        omega=4.0,
        regularization=0.01,
        contribution_space="direct_item_gradient",
    )
    common = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        torch.arange(len(users)),
        **kwargs,
    )
    separated = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        torch.arange(len(users)),
        item_learning_rate=0.01,
        **kwargs,
    )
    assert torch.equal(common[0], separated[0])
    assert not torch.equal(common[1], separated[1])
    expected_items = items + 0.1 * (common[1] - items)
    assert torch.allclose(separated[1], expected_items, atol=1e-12, rtol=1e-10)


def test_only_selected_users_update_and_joint_bound_is_respected() -> None:
    users, items, train_users, train_items = _problem()
    selected = torch.tensor([0, 2], dtype=torch.long)
    updated_users, _, diagnostics = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        selected,
        learning_rate=0.1,
        sampling_probability=2.0 / 3.0,
        clip_norm=0.05,
        noise_multiplier=0.0,
        omega=4.0,
    )
    assert not torch.equal(updated_users[0], users[0])
    assert torch.equal(updated_users[1], users[1])
    assert not torch.equal(updated_users[2], users[2])
    post_norm = (
        diagnostics.contribution.pre_clip_norm
        * diagnostics.contribution.clipping_factor
    )
    assert torch.all(post_norm <= 0.05 + 1e-12)


def test_dp_round_projects_selected_users_before_forming_statistics() -> None:
    users, items, train_users, train_items = _problem()
    selected = torch.tensor([0, 2], dtype=torch.long)
    radius = 0.03
    updated_users, _, diagnostics = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        selected,
        learning_rate=0.1,
        sampling_probability=2.0 / 3.0,
        clip_norm=1e6,
        noise_multiplier=0.0,
        omega=4.0,
        user_factor_radius=radius,
    )

    assert torch.all(updated_users.index_select(0, selected).norm(dim=1) <= radius)
    assert torch.equal(updated_users[1], users[1])
    assert not diagnostics.contribution.clipped.any()


def test_normalized_direct_contributions_have_fixed_nonzero_norm() -> None:
    users, items, train_users, train_items = _problem()
    selected = torch.tensor([0, 2], dtype=torch.long)
    _, _, diagnostics = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        selected,
        learning_rate=0.1,
        sampling_probability=2.0 / 3.0,
        clip_norm=0.05,
        noise_multiplier=0.0,
        omega=4.0,
        contribution_space="direct_item_gradient",
        contribution_rule="normalize",
    )
    post_norm = (
        diagnostics.contribution.pre_clip_norm
        * diagnostics.contribution.clipping_factor
    )
    assert torch.allclose(post_norm, torch.full_like(post_norm, 0.05), atol=1e-12)


def test_dp_round_noise_is_reproducible_with_fixed_generator() -> None:
    users, items, train_users, train_items = _problem()
    kwargs = dict(
        learning_rate=0.1,
        sampling_probability=1.0,
        clip_norm=0.2,
        noise_multiplier=1.1,
        omega=4.0,
    )
    first = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        torch.arange(len(users)),
        generator=torch.Generator().manual_seed(23),
        **kwargs,
    )
    second = run_uniform_dp_pnsmf_bgd_round(
        users,
        items,
        train_users,
        train_items,
        torch.arange(len(users)),
        generator=torch.Generator().manual_seed(23),
        **kwargs,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2].gaussian_noise_norm == second[2].gaussian_noise_norm
    assert first[2].gaussian_noise_norm > 0.0


def test_server_ema_delta_is_identity_at_zero_momentum() -> None:
    current = torch.tensor([1.0, 2.0], dtype=torch.float64)
    proposed = torch.tensor([1.5, 1.0], dtype=torch.float64)
    velocity = torch.zeros_like(current)
    updated, next_velocity = server_ema_delta(
        current, proposed, velocity, momentum=0.0
    )
    assert torch.equal(updated, proposed)
    assert torch.equal(next_velocity, proposed - current)


def test_server_ema_delta_smooths_released_delta() -> None:
    current = torch.tensor([1.0, 2.0], dtype=torch.float64)
    proposed = torch.tensor([3.0, 0.0], dtype=torch.float64)
    velocity = torch.tensor([0.5, -0.5], dtype=torch.float64)
    updated, next_velocity = server_ema_delta(
        current, proposed, velocity, momentum=0.5
    )
    assert torch.equal(next_velocity, torch.tensor([1.25, -1.25], dtype=torch.float64))
    assert torch.equal(updated, torch.tensor([2.25, 0.75], dtype=torch.float64))
