import torch

from src.privacy import (
    add_gaussian_noise_to_pnsmf_statistics,
    aggregate_clipped_pnsmf_item_gradients,
    aggregate_clipped_pnsmf_statistics,
    isometric_vector_to_symmetric,
    symmetric_to_isometric_vector,
)


def _problem():
    users = torch.tensor(
        [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]], dtype=torch.float64
    )
    items = torch.tensor(
        [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2], [0.2, 0.1]],
        dtype=torch.float64,
    )
    train_users = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    train_items = torch.tensor([0, 2, 1, 0, 3], dtype=torch.long)
    return users, items, train_users, train_items


def test_symmetric_half_vectorization_is_frobenius_isometry() -> None:
    matrix = torch.tensor(
        [[2.0, -3.0, 4.0], [-3.0, 5.0, 6.0], [4.0, 6.0, -1.0]],
        dtype=torch.float64,
    )
    vector = symmetric_to_isometric_vector(matrix)
    recovered = isometric_vector_to_symmetric(vector, 3)
    assert torch.allclose(vector.square().sum(), matrix.square().sum(), atol=1e-12)
    assert torch.equal(recovered, matrix)


def test_sparse_joint_clipping_matches_explicit_fixed_dimension_vectors() -> None:
    users, items, train_users, train_items = _problem()
    selected = torch.tensor([0, 2], dtype=torch.long)
    max_norm = 0.35
    item_sum, moment_sum, diagnostics = aggregate_clipped_pnsmf_statistics(
        users,
        items,
        train_users,
        train_items,
        selected,
        omega=4.0,
        max_norm=max_norm,
    )

    explicit = []
    for user in selected.tolist():
        dense_rows = torch.zeros_like(items)
        mask = train_users == user
        positive_items = train_items[mask]
        selected_items = items.index_select(0, positive_items)
        predictions = selected_items @ users[user]
        rows = (3.0 * predictions - 4.0).unsqueeze(1) * users[user]
        dense_rows.index_add_(0, positive_items, rows)
        moment = torch.outer(users[user], users[user])
        explicit.append(
            torch.cat([dense_rows.flatten(), symmetric_to_isometric_vector(moment)])
        )
    explicit = torch.stack(explicit)
    norms = torch.linalg.vector_norm(explicit, dim=1)
    factors = torch.clamp(max_norm / norms, max=1.0)
    explicit_sum = (explicit * factors[:, None]).sum(dim=0)
    split = items.numel()

    assert torch.allclose(diagnostics.pre_clip_norm, norms, atol=1e-12)
    assert torch.allclose(diagnostics.clipping_factor, factors, atol=1e-12)
    assert torch.allclose(item_sum.flatten(), explicit_sum[:split], atol=1e-12)
    assert torch.allclose(
        symmetric_to_isometric_vector(moment_sum), explicit_sum[split:], atol=1e-12
    )
    assert torch.all(norms * factors <= max_norm + 1e-12)


def test_joint_gaussian_noise_is_reproducible_and_symmetric() -> None:
    item_sum = torch.zeros((4, 2), dtype=torch.float64)
    moment_sum = torch.zeros((2, 2), dtype=torch.float64)
    first = add_gaussian_noise_to_pnsmf_statistics(
        item_sum,
        moment_sum,
        max_norm=0.5,
        noise_multiplier=1.2,
        generator=torch.Generator().manual_seed(7),
    )
    second = add_gaussian_noise_to_pnsmf_statistics(
        item_sum,
        moment_sum,
        max_norm=0.5,
        noise_multiplier=1.2,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[1], first[1].T)


def test_zero_noise_preserves_aggregates() -> None:
    item_sum = torch.randn(4, 2, dtype=torch.float64)
    raw = torch.randn(2, 2, dtype=torch.float64)
    moment_sum = raw + raw.T
    noisy_items, noisy_moment = add_gaussian_noise_to_pnsmf_statistics(
        item_sum, moment_sum, max_norm=1.0, noise_multiplier=0.0
    )
    assert torch.equal(noisy_items, item_sum)
    assert torch.equal(noisy_moment, moment_sum)


def test_direct_item_gradient_clipping_matches_explicit_dense_vectors() -> None:
    users, items, train_users, train_items = _problem()
    selected = torch.tensor([0, 2], dtype=torch.long)
    max_norm = 0.4
    gradient_sum, diagnostics = aggregate_clipped_pnsmf_item_gradients(
        users,
        items,
        train_users,
        train_items,
        selected,
        omega=4.0,
        max_norm=max_norm,
    )
    explicit = []
    for user in selected.tolist():
        rows = []
        positives = set(train_items[train_users == user].tolist())
        for item in range(len(items)):
            prediction = users[user] @ items[item]
            scalar = 4.0 * (prediction - 1.0) if item in positives else prediction
            rows.append(scalar * users[user])
        explicit.append(torch.stack(rows).flatten())
    explicit = torch.stack(explicit)
    norms = torch.linalg.vector_norm(explicit, dim=1)
    factors = torch.clamp(max_norm / norms, max=1.0)
    expected = (explicit * factors[:, None]).sum(dim=0).reshape_as(items)
    assert torch.allclose(diagnostics.pre_clip_norm, norms, atol=1e-12)
    assert torch.allclose(diagnostics.clipping_factor, factors, atol=1e-12)
    assert torch.allclose(gradient_sum, expected, atol=1e-12)


def test_direct_item_gradient_normalization_matches_fixed_norm_vectors() -> None:
    users, items, train_users, train_items = _problem()
    selected = torch.tensor([0, 2], dtype=torch.long)
    max_norm = 0.4
    gradient_sum, diagnostics = aggregate_clipped_pnsmf_item_gradients(
        users,
        items,
        train_users,
        train_items,
        selected,
        omega=4.0,
        max_norm=max_norm,
        contribution_rule="normalize",
    )
    explicit = []
    for user in selected.tolist():
        rows = []
        positives = set(train_items[train_users == user].tolist())
        for item in range(len(items)):
            prediction = users[user] @ items[item]
            scalar = 4.0 * (prediction - 1.0) if item in positives else prediction
            rows.append(scalar * users[user])
        explicit.append(torch.stack(rows).flatten())
    explicit = torch.stack(explicit)
    norms = torch.linalg.vector_norm(explicit, dim=1)
    factors = max_norm / norms
    expected = (explicit * factors[:, None]).sum(dim=0).reshape_as(items)
    assert torch.allclose(diagnostics.pre_clip_norm, norms, atol=1e-12)
    assert torch.allclose(diagnostics.clipping_factor, factors, atol=1e-12)
    assert torch.allclose(gradient_sum, expected, atol=1e-12)
    assert torch.allclose(norms * factors, torch.full_like(norms, max_norm))


def test_unclipped_joint_reconstruction_equals_direct_query() -> None:
    users, items, train_users, train_items = _problem()
    selected = torch.tensor([0, 2], dtype=torch.long)
    item_sum, moment_sum, joint_diagnostics = aggregate_clipped_pnsmf_statistics(
        users,
        items,
        train_users,
        train_items,
        selected,
        omega=4.0,
        max_norm=1e9,
    )
    direct_sum, direct_diagnostics = aggregate_clipped_pnsmf_item_gradients(
        users,
        items,
        train_users,
        train_items,
        selected,
        omega=4.0,
        max_norm=1e9,
    )
    assert not joint_diagnostics.clipped.any()
    assert not direct_diagnostics.clipped.any()
    assert torch.allclose(direct_sum, item_sum + items @ moment_sum, atol=1e-12)


def test_single_user_sufficient_statistics_have_full_linear_span() -> None:
    """Small-instance audit of the analytic no-lossless-subspace proposition."""

    generator = torch.Generator().manual_seed(20260802)
    num_items = 3
    dimension = 2
    item_factors = torch.randn(
        (num_items, dimension), dtype=torch.float64, generator=generator
    )
    contributions = []

    for _ in range(20):
        user_factor = torch.randn(
            (1, dimension), dtype=torch.float64, generator=generator
        )
        for item_index in range(num_items):
            item_sum, moment_sum, _ = aggregate_clipped_pnsmf_statistics(
                user_factor,
                item_factors,
                torch.tensor([0], dtype=torch.long),
                torch.tensor([item_index], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                omega=4.0,
                max_norm=1e9,
            )
            contributions.append(
                torch.cat(
                    (
                        item_sum.flatten(),
                        symmetric_to_isometric_vector(moment_sum),
                    )
                )
            )

    contribution_matrix = torch.stack(contributions)
    expected_dimension = (
        num_items * dimension + dimension * (dimension + 1) // 2
    )
    assert torch.linalg.matrix_rank(contribution_matrix) == expected_dimension
