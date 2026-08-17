import math

import torch

from src.privacy import (
    debiased_signal_energy,
    compare_pnsmf_query_noise_energy,
    compare_pnsmf_query_noise_energy_quantile,
    compare_pnsmf_query_noise_energy_tail,
    direct_query_feedback_bound,
    direct_query_noise_accumulation,
    gaussian_ball_containment_certificate,
    minimum_signal_norm_for_energy_power,
    minimum_feasible_radius_for_gaussian_containment,
    minimum_projection_displacement_for_gaussian_update,
    noisy_energy_test_power,
    optimal_two_block_gaussian_allocation,
    pnsmf_catalog_sensitivity_bound,
    pnsmf_affine_sensitivity_bounds,
    gaussian_norm_radius,
    joint_statistics_feedback_bound,
    isometric_vector_to_symmetric,
    poisson_participation_upper_bound,
    public_noiseless_item_map_envelope,
    signal_energy_estimator_variance,
    select_pnsmf_query_noise_energy_quantile,
)


def test_constant_schedule_matches_random_walk_closed_form() -> None:
    users = 1000
    items = 400
    dimension = 8
    rounds = 25
    learning_rate = 0.3
    sampling_probability = 0.1
    clip_norm = 0.8
    noise_multiplier = 1.7

    result = direct_query_noise_accumulation(
        num_users=users,
        num_items=items,
        embedding_dim=dimension,
        item_learning_rates=[learning_rate] * rounds,
        sampling_probabilities=sampling_probability,
        clip_norms=clip_norm,
        noise_multipliers=noise_multiplier,
    )
    expected_rms = (
        2.0
        * learning_rate
        * noise_multiplier
        * clip_norm
        / (sampling_probability * users)
        * math.sqrt(items * dimension * rounds)
    )

    assert result.query_dimension == items * dimension
    assert math.isclose(result.rms_frobenius_norm, expected_rms, rel_tol=1e-12)


def test_regularization_recurrence_matches_geometric_series() -> None:
    rounds = 30
    learning_rate = 0.2
    regularization = 0.05
    contraction = 1.0 - 2.0 * learning_rate * regularization
    users = 600
    sampling_probability = 0.2
    clip_norm = 0.7
    noise_multiplier = 1.3
    update_variance = (
        2.0 * learning_rate / (sampling_probability * users)
    ) ** 2 * (noise_multiplier * clip_norm) ** 2
    expected_coordinate_variance = (
        update_variance * (1.0 - contraction ** (2 * rounds))
        / (1.0 - contraction**2)
    )

    result = direct_query_noise_accumulation(
        num_users=users,
        num_items=20,
        embedding_dim=3,
        item_learning_rates=[learning_rate] * rounds,
        sampling_probabilities=sampling_probability,
        clip_norms=clip_norm,
        noise_multipliers=noise_multiplier,
        regularization=regularization,
    )

    assert math.isclose(
        result.per_coordinate_variance,
        expected_coordinate_variance,
        rel_tol=1e-12,
    )


def test_monte_carlo_noise_energy_matches_analytic_expectation() -> None:
    trials = 20_000
    query_dimension = 12
    learning_rates = [0.4, 0.3, 0.2, 0.1]
    sampling = [0.2, 0.25, 0.3, 0.35]
    clipping = [0.9, 0.8, 0.7, 0.6]
    multipliers = [1.4, 1.3, 1.2, 1.1]
    regularization = [0.03, 0.02, 0.01, 0.0]
    users = 500

    analytic = direct_query_noise_accumulation(
        num_users=users,
        num_items=4,
        embedding_dim=3,
        item_learning_rates=learning_rates,
        sampling_probabilities=sampling,
        clip_norms=clipping,
        noise_multipliers=multipliers,
        regularization=regularization,
    )

    generator = torch.Generator().manual_seed(20260802)
    accumulator = torch.zeros((trials, query_dimension), dtype=torch.float64)
    for eta, q_value, clip, multiplier, reg in zip(
        learning_rates,
        sampling,
        clipping,
        multipliers,
        regularization,
        strict=True,
    ):
        contraction = 1.0 - 2.0 * eta * reg
        update_scale = 2.0 * eta / (q_value * users)
        innovation = torch.randn(
            accumulator.shape, dtype=torch.float64, generator=generator
        ) * (multiplier * clip)
        accumulator = contraction * accumulator - update_scale * innovation

    empirical = float(accumulator.square().sum(dim=1).mean())
    assert math.isclose(
        empirical,
        analytic.expected_squared_frobenius_norm,
        rel_tol=0.02,
    )


def test_schedule_validation_rejects_data_independent_shape_errors() -> None:
    try:
        direct_query_noise_accumulation(
            num_users=10,
            num_items=4,
            embedding_dim=2,
            item_learning_rates=[0.1, 0.1],
            sampling_probabilities=[0.1],
            clip_norms=1.0,
            noise_multipliers=1.0,
        )
    except ValueError as error:
        assert "sampling_probabilities" in str(error)
    else:
        raise AssertionError("Mismatched schedule length should fail.")


def test_debiased_signal_energy_has_correct_exact_moments() -> None:
    query_dimension = 30
    noise_standard_deviation = 1.7
    signal = torch.linspace(-0.5, 0.5, query_dimension, dtype=torch.float64)
    signal_squared_norm = float(signal.square().sum())
    trials = 60_000
    generator = torch.Generator().manual_seed(20260803)
    noise = torch.randn(
        (trials, query_dimension), dtype=torch.float64, generator=generator
    ) * noise_standard_deviation
    noisy_squared_norms = (signal.unsqueeze(0) + noise).square().sum(dim=1)
    estimates = noisy_squared_norms - query_dimension * noise_standard_deviation**2

    analytic_variance = signal_energy_estimator_variance(
        signal_squared_norm,
        query_dimension=query_dimension,
        noise_standard_deviation=noise_standard_deviation,
    )
    assert math.isclose(float(estimates.mean()), signal_squared_norm, abs_tol=0.08)
    assert math.isclose(
        float(estimates.var(unbiased=True)), analytic_variance, rel_tol=0.025
    )
    assert math.isclose(
        debiased_signal_energy(
            float(noisy_squared_norms[0]),
            query_dimension=query_dimension,
            noise_standard_deviation=noise_standard_deviation,
        ),
        float(estimates[0]),
        rel_tol=1e-12,
    )


def test_energy_test_has_nominal_null_size_and_monotone_power() -> None:
    kwargs = {
        "query_dimension": 200,
        "noise_standard_deviation": 2.0,
        "significance_level": 0.05,
    }
    null_power = noisy_energy_test_power(0.0, **kwargs)
    moderate_power = noisy_energy_test_power(10.0, **kwargs)
    strong_power = noisy_energy_test_power(30.0, **kwargs)

    assert math.isclose(null_power, 0.05, rel_tol=1e-10)
    assert null_power < moderate_power < strong_power


def test_minimum_signal_norm_inverts_target_power() -> None:
    required = minimum_signal_norm_for_energy_power(
        query_dimension=1000,
        noise_standard_deviation=1.5,
        target_power=0.8,
        significance_level=0.05,
    )
    achieved = noisy_energy_test_power(
        required,
        query_dimension=1000,
        noise_standard_deviation=1.5,
        significance_level=0.05,
    )
    assert math.isclose(achieved, 0.8, abs_tol=1e-10)


def test_two_block_allocation_saturates_constraint_and_improves_proxy() -> None:
    allocation = optimal_two_block_gaussian_allocation(
        first_block_bound=3.0,
        second_block_bound=1.0,
        first_quadratic_weight=2.0,
        second_quadratic_weight=7.0,
        standardized_sensitivity_budget=0.4,
    )

    assert math.isclose(
        allocation.standardized_sensitivity_squared, 0.4**2, rel_tol=1e-12
    )
    assert allocation.optimized_weighted_variance < allocation.isotropic_weighted_variance
    assert 0.0 < allocation.improvement_ratio < 1.0


def test_two_block_allocation_has_cauchy_equality_case() -> None:
    # Equality holds when block bounds are proportional to sqrt(weights).
    allocation = optimal_two_block_gaussian_allocation(
        first_block_bound=2.0,
        second_block_bound=3.0,
        first_quadratic_weight=4.0,
        second_quadratic_weight=9.0,
        standardized_sensitivity_budget=0.75,
    )

    assert math.isclose(allocation.first_variance, allocation.second_variance)
    assert math.isclose(allocation.improvement_ratio, 1.0, rel_tol=1e-12)


def test_catalog_sensitivity_bound_dominates_exhaustive_contributions() -> None:
    item_factors = torch.tensor(
        [[0.2, -0.1], [0.7, 0.3], [-0.4, 0.5], [0.1, 0.2]],
        dtype=torch.float64,
    )
    radius = 1.3
    omega = 1.0
    alpha = 0.15
    beta = 4.0
    cap = 2
    bound = pnsmf_catalog_sensitivity_bound(
        item_factor_norms=item_factors.norm(dim=1).tolist(),
        interaction_cap=cap,
        user_factor_radius=radius,
        positive_weight=omega,
        missing_weight=alpha,
        catalog_weight=beta,
    )

    # Dense angle/radius enumeration is not a proof; it audits the proved bound.
    observed_max = 0.0
    for radial_step in range(21):
        radial = radius * radial_step / 20.0
        for angle_step in range(720):
            angle = 2.0 * math.pi * angle_step / 720.0
            user = radial * torch.tensor(
                [math.cos(angle), math.sin(angle)], dtype=torch.float64
            )
            item_terms = ((omega - alpha * beta) * (item_factors @ user) - omega) ** 2
            sparse_squared = radial**2 * float(torch.topk(item_terms, cap).values.sum())
            moment_squared = alpha**2 * radial**4
            observed_max = max(observed_max, sparse_squared + moment_squared)

    assert observed_max <= bound.catalog_dependent_squared_bound * (1.0 + 1e-12)
    assert bound.catalog_dependent_squared_bound <= bound.max_norm_squared_bound


def test_catalog_sensitivity_bound_is_strict_for_heterogeneous_norms() -> None:
    bound = pnsmf_catalog_sensitivity_bound(
        item_factor_norms=[2.0, 0.6, 0.4, 0.2],
        interaction_cap=3,
        user_factor_radius=1.0,
        positive_weight=1.0,
        missing_weight=0.1,
        catalog_weight=2.0,
    )

    assert bound.catalog_dependent_squared_bound < bound.max_norm_squared_bound
    assert 0.0 < bound.bound_ratio < 1.0


def test_affine_sensitivity_bounds_dominate_exact_catalog_bound() -> None:
    factors = torch.tensor(
        [[0.2, -0.1], [0.7, 0.3], [-0.4, 0.5], [0.1, 0.2]],
        dtype=torch.float64,
    )
    exact_catalog = pnsmf_catalog_sensitivity_bound(
        item_factor_norms=factors.norm(dim=1).tolist(),
        interaction_cap=2,
        user_factor_radius=1.3,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
    )
    affine = pnsmf_affine_sensitivity_bounds(
        interaction_cap=2,
        user_factor_radius=1.3,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
    )
    item_scale = float(factors.norm())
    assert (
        affine.joint_intercept + affine.joint_slope * item_scale
        >= exact_catalog.catalog_dependent_squared_bound**0.5
    )
    assert affine.direct_slope >= affine.joint_slope


def test_gaussian_norm_radius_increases_with_events_and_confidence() -> None:
    baseline = gaussian_norm_radius(
        dimension=100, events=5, failure_probability=0.05
    )
    more_events = gaussian_norm_radius(
        dimension=100, events=50, failure_probability=0.05
    )
    more_confident = gaussian_norm_radius(
        dimension=100, events=5, failure_probability=0.005
    )
    assert more_events > baseline
    assert more_confident > baseline


def test_direct_feedback_has_affine_stability_threshold() -> None:
    bound = direct_query_feedback_bound(
        num_items=20,
        embedding_dim=3,
        rounds=10,
        failure_probability=0.05,
        update_scale=0.001,
        noise_multiplier=1.0,
        sensitivity_intercept=0.2,
        sensitivity_slope=0.01,
        noiseless_intercept=0.05,
        noiseless_slope=0.7,
    )
    assert bound.zeta == 0.0
    assert bound.has_bounded_region
    assert bound.attracting_bound is not None
    assert math.isinf(bound.escape_radius)
    assert math.isclose(
        bound.attracting_bound,
        bound.kappa / (1.0 - bound.rho),
        rel_tol=1e-12,
    )


def test_joint_feedback_has_quadratic_escape_radius_when_stable() -> None:
    bound = joint_statistics_feedback_bound(
        num_items=20,
        embedding_dim=3,
        rounds=10,
        failure_probability=0.05,
        update_scale=0.0001,
        noise_multiplier=0.5,
        catalog_weight=1.0,
        sensitivity_intercept=0.1,
        sensitivity_slope=0.01,
        noiseless_intercept=0.01,
        noiseless_slope=0.5,
    )
    assert bound.zeta > 0.0
    assert bound.has_bounded_region
    assert bound.attracting_bound is not None
    assert bound.escape_radius is not None
    assert bound.attracting_bound < bound.escape_radius
    for root in (bound.attracting_bound, bound.escape_radius):
        assert math.isclose(
            bound.kappa + bound.rho * root + bound.zeta * root**2,
            root,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )


def test_gate_d8_public_configuration_fails_noise_only_stability_gate() -> None:
    sensitivity = pnsmf_affine_sensitivity_bounds(
        interaction_cap=50,
        user_factor_radius=1.0,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
    )
    bound = joint_statistics_feedback_bound(
        num_items=3952,
        embedding_dim=20,
        rounds=2,
        failure_probability=0.05,
        update_scale=2.0 * 3.0 / (0.1 * 6040),
        noise_multiplier=3.0,
        catalog_weight=1.0,
        sensitivity_intercept=sensitivity.joint_intercept,
        sensitivity_slope=sensitivity.joint_slope,
        noiseless_intercept=0.0,
        noiseless_slope=0.0,
    )
    assert bound.rho > 1.0
    assert bound.zeta > 0.0
    assert not bound.has_bounded_region


def test_participation_upper_bound_is_simultaneous_and_capped() -> None:
    one_round = poisson_participation_upper_bound(
        num_users=1000,
        sampling_probability=0.1,
        rounds=1,
        failure_probability=0.05,
    )
    many_rounds = poisson_participation_upper_bound(
        num_users=1000,
        sampling_probability=0.1,
        rounds=100,
        failure_probability=0.05,
    )
    all_users = poisson_participation_upper_bound(
        num_users=10,
        sampling_probability=1.0,
        rounds=100,
        failure_probability=0.05,
    )
    assert one_round.expected_participants == 100.0
    assert one_round.upper_participants < many_rounds.upper_participants
    assert many_rounds.upper_participants <= 1000
    assert all_users.upper_participants == 10.0


def test_public_item_map_envelope_matches_scalar_extreme_eigenvalues() -> None:
    envelope = public_noiseless_item_map_envelope(
        num_users=1000,
        sampling_probability=0.1,
        participation_upper=125.0,
        interaction_cap=20,
        user_factor_radius=0.5,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
        regularization=0.02,
        item_learning_rate=0.1,
    )
    lower = 2.0 * 0.02
    expected_upper = 2.0 * 4.0 * 125.0 * 0.5**2 / 100.0 + lower
    assert math.isclose(envelope.hessian_eigenvalue_upper, expected_upper)
    assert math.isclose(
        envelope.item_map_slope,
        max(abs(1.0 - 0.1 * lower), abs(1.0 - 0.1 * expected_upper)),
    )
    assert envelope.item_map_intercept > 0.0


def test_gate_d8_public_noiseless_envelope_rejects_known_large_step() -> None:
    participation = poisson_participation_upper_bound(
        num_users=6040,
        sampling_probability=0.1,
        rounds=2,
        failure_probability=0.05,
    )
    envelope = public_noiseless_item_map_envelope(
        num_users=6040,
        sampling_probability=0.1,
        participation_upper=participation.upper_participants,
        interaction_cap=50,
        user_factor_radius=1.0,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
        regularization=0.01,
        item_learning_rate=3.0,
    )
    assert envelope.item_map_slope > 1.0
    assert envelope.item_map_intercept > 100.0


def test_gaussian_containment_certificate_matches_chi_square_quantile() -> None:
    dimension = 40
    noise_standard_deviation = 0.7
    target_probability = 0.9
    radius = minimum_feasible_radius_for_gaussian_containment(
        query_dimension=dimension,
        update_noise_standard_deviation=noise_standard_deviation,
        target_containment_probability=target_probability,
    )
    certificate = gaussian_ball_containment_certificate(
        query_dimension=dimension,
        update_noise_standard_deviation=noise_standard_deviation,
        feasible_radius=radius,
    )
    assert math.isclose(
        certificate.maximum_containment_probability,
        target_probability,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert math.isclose(
        certificate.minimum_exit_probability,
        1.0 - target_probability,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert math.isclose(
        certificate.noise_rms_norm,
        noise_standard_deviation * math.sqrt(dimension),
    )


def test_gate_d8_noise_requires_catalog_scale_radius() -> None:
    dimension = 3952 * 20
    public_clip_norm = 28.642566978892766
    update_noise_standard_deviation = (
        2.0 * 3.0 / (0.1 * 6040)
    ) * 3.0 * public_clip_norm
    radius = minimum_feasible_radius_for_gaussian_containment(
        query_dimension=dimension,
        update_noise_standard_deviation=update_noise_standard_deviation,
        target_containment_probability=0.95,
    )
    row_rms_radius = radius / math.sqrt(3952)
    assert 240.0 < radius < 242.0
    assert 3.8 < row_rms_radius < 3.9


def test_gate_d8_unit_row_projection_would_strongly_intervene() -> None:
    dimension = 3952 * 20
    public_clip_norm = 28.642566978892766
    update_noise_standard_deviation = (
        2.0 * 3.0 / (0.1 * 6040)
    ) * 3.0 * public_clip_norm
    displacement = minimum_projection_displacement_for_gaussian_update(
        query_dimension=dimension,
        update_noise_standard_deviation=update_noise_standard_deviation,
        projection_radius=math.sqrt(3952),
        target_probability=0.95,
    )
    assert 175.0 < displacement < 177.0


def test_joint_noise_energy_accounts_for_isometric_symmetric_block() -> None:
    comparison = compare_pnsmf_query_noise_energy(
        num_items=7,
        embedding_dim=3,
        item_factor_norm=2.0,
        catalog_weight=1.5,
        joint_clip_norm=0.8,
        direct_clip_norm=1.1,
        update_scale=0.2,
        noise_multiplier=1.3,
    )
    common = (0.2 * 1.3) ** 2
    expected_joint = common * 0.8**2 * (7 * 3 + 1.5**2 * 4 * 2.0**2 / 2)
    expected_direct = common * 1.1**2 * 7 * 3
    assert math.isclose(comparison.joint_expected_noise_energy, expected_joint)
    assert math.isclose(comparison.direct_expected_noise_energy, expected_direct)


def test_affine_bounds_create_nontrivial_representation_switching_region() -> None:
    sensitivity = pnsmf_affine_sensitivity_bounds(
        interaction_cap=50,
        user_factor_radius=1.0,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
    )

    def preferred(item_norm: float) -> str:
        return compare_pnsmf_query_noise_energy(
            num_items=3952,
            embedding_dim=20,
            item_factor_norm=item_norm,
            catalog_weight=1.0,
            joint_clip_norm=(
                sensitivity.joint_intercept + sensitivity.joint_slope * item_norm
            ),
            direct_clip_norm=(
                sensitivity.direct_intercept + sensitivity.direct_slope * item_norm
            ),
            update_scale=2.0 * 3.0 / (0.1 * 6040),
            noise_multiplier=3.0,
        ).preferred_representation

    assert preferred(0.5) == "direct_item_gradient"
    assert preferred(20.0) == "joint_sufficient_statistics"
    assert preferred(100.0) == "direct_item_gradient"


def test_joint_noise_energy_formula_matches_monte_carlo() -> None:
    generator = torch.Generator().manual_seed(20260802)
    trials, num_items, dimension = 20000, 5, 3
    catalog_weight = 1.4
    item_factors = torch.randn(
        (num_items, dimension), dtype=torch.float64, generator=generator
    )
    sparse_noise = torch.randn(
        (trials, num_items, dimension),
        dtype=torch.float64,
        generator=generator,
    )
    packed_moment_noise = torch.randn(
        (trials, dimension * (dimension + 1) // 2),
        dtype=torch.float64,
        generator=generator,
    )
    moment_noise = isometric_vector_to_symmetric(
        packed_moment_noise, dimension
    )
    reconstructed_noise = sparse_noise + catalog_weight * torch.einsum(
        "md,tdk->tmk", item_factors, moment_noise
    )
    empirical_energy = float(
        reconstructed_noise.square().sum(dim=(1, 2)).mean()
    )
    analytic = compare_pnsmf_query_noise_energy(
        num_items=num_items,
        embedding_dim=dimension,
        item_factor_norm=float(item_factors.norm()),
        catalog_weight=catalog_weight,
        joint_clip_norm=1.0,
        direct_clip_norm=1.0,
        update_scale=1.0,
        noise_multiplier=1.0,
    ).joint_expected_noise_energy
    assert math.isclose(empirical_energy, analytic, rel_tol=0.015)


def test_symmetric_reconstruction_spectrum_has_expected_trace() -> None:
    gram_eigenvalues = [0.4, 1.1, 2.3]
    moment_spectrum = list(gram_eigenvalues) + [
        (gram_eigenvalues[i] + gram_eigenvalues[j]) / 2.0
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    assert math.isclose(
        sum(moment_spectrum),
        (3 + 1) / 2 * sum(gram_eigenvalues),
    )


def test_symmetric_reconstruction_spectrum_matches_explicit_operator() -> None:
    generator = torch.Generator().manual_seed(20260802)
    num_items, dimension = 4, 3
    item_factors = torch.randn(
        (num_items, dimension), dtype=torch.float64, generator=generator
    )
    packed_dimension = dimension * (dimension + 1) // 2
    symmetric_basis = isometric_vector_to_symmetric(
        torch.eye(packed_dimension, dtype=torch.float64), dimension
    )
    operator = torch.stack(
        [(item_factors @ basis).flatten() for basis in symmetric_basis], dim=1
    )
    explicit = torch.linalg.eigvalsh(operator.T @ operator)
    gram = torch.linalg.eigvalsh(item_factors.T @ item_factors)
    predicted = torch.tensor(
        list(gram.tolist())
        + [
            float((gram[i] + gram[j]) / 2.0)
            for i in range(dimension)
            for j in range(i + 1, dimension)
        ],
        dtype=torch.float64,
    ).sort().values
    assert torch.allclose(explicit, predicted, atol=1e-12, rtol=1e-10)


def test_tail_comparison_matches_direct_gaussian_concentration_formula() -> None:
    result = compare_pnsmf_query_noise_energy_tail(
        num_items=7,
        item_gram_eigenvalues=[0.4, 1.1, 2.3],
        catalog_weight=1.2,
        joint_clip_norm=0.8,
        direct_clip_norm=1.1,
        update_scale=0.2,
        noise_multiplier=1.3,
        failure_probability=0.05,
    )
    query_dimension = 7 * 3
    variance = (0.2 * 1.3 * 1.1) ** 2
    log_term = math.log(20.0)
    expected_direct_upper = variance * (
        query_dimension
        + 2.0 * math.sqrt(query_dimension * log_term)
        + 2.0 * log_term
    )
    assert math.isclose(result.direct_noise_energy_upper, expected_direct_upper)
    assert result.joint_noise_energy_upper > result.joint_expected_noise_energy


def test_tail_selection_remains_nontrivial_for_gate_d8_scales() -> None:
    sensitivity = pnsmf_affine_sensitivity_bounds(
        interaction_cap=50,
        user_factor_radius=1.0,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
    )

    def preferred(item_norm: float) -> str:
        isotropic_gram = [item_norm**2 / 20.0] * 20
        return compare_pnsmf_query_noise_energy_tail(
            num_items=3952,
            item_gram_eigenvalues=isotropic_gram,
            catalog_weight=1.0,
            joint_clip_norm=(
                sensitivity.joint_intercept + sensitivity.joint_slope * item_norm
            ),
            direct_clip_norm=(
                sensitivity.direct_intercept + sensitivity.direct_slope * item_norm
            ),
            update_scale=2.0 * 3.0 / (0.1 * 6040),
            noise_multiplier=3.0,
            failure_probability=0.05,
        ).preferred_representation

    assert preferred(0.5) == "direct_item_gradient"
    assert preferred(20.0) == "joint_sufficient_statistics"
    assert preferred(100.0) == "direct_item_gradient"


def test_imhof_quantile_reduces_to_ordinary_chi_square() -> None:
    result = compare_pnsmf_query_noise_energy_quantile(
        num_items=128,
        item_gram_eigenvalues=[0.4, 1.1, 2.3],
        catalog_weight=0.0,
        joint_clip_norm=0.8,
        direct_clip_norm=0.8,
        update_scale=0.2,
        noise_multiplier=1.3,
        quantile_probability=0.95,
    )
    assert math.isclose(
        result.joint_noise_energy_quantile,
        result.direct_noise_energy_quantile,
        rel_tol=2e-9,
    )


def test_exact_quantile_resolves_representation_disagreement_states() -> None:
    sensitivity = pnsmf_affine_sensitivity_bounds(
        interaction_cap=50,
        user_factor_radius=1.0,
        positive_weight=4.0,
        missing_weight=1.0,
        catalog_weight=1.0,
    )

    def preferred(item_norm: float, geometry: str) -> str:
        if geometry == "isotropic":
            gram = [item_norm**2 / 8.0] * 8
        else:
            gram = [item_norm**2] + [0.0] * 7
        return compare_pnsmf_query_noise_energy_quantile(
            num_items=128,
            item_gram_eigenvalues=gram,
            catalog_weight=1.0,
            joint_clip_norm=(
                sensitivity.joint_intercept
                + sensitivity.joint_slope * item_norm
            ),
            direct_clip_norm=(
                sensitivity.direct_intercept
                + sensitivity.direct_slope * item_norm
            ),
            update_scale=1.0,
            noise_multiplier=1.0,
            quantile_probability=0.95,
        ).preferred_representation

    assert preferred(7.0, "isotropic") == "joint_sufficient_statistics"
    assert preferred(5.0, "rank1") == "joint_sufficient_statistics"
    assert preferred(7.0, "rank1") == "direct_item_gradient"


def test_single_cdf_quantile_decision_matches_inverse_cdf_comparison() -> None:
    common = dict(
        num_items=128,
        item_gram_eigenvalues=[49.0] + [0.0] * 7,
        catalog_weight=1.0,
        joint_clip_norm=50.2842712474619,
        direct_clip_norm=56.2842712474619,
        update_scale=0.0005,
        noise_multiplier=1.0,
        quantile_probability=0.95,
    )
    full = compare_pnsmf_query_noise_energy_quantile(**common)
    decision = select_pnsmf_query_noise_energy_quantile(**common)
    assert decision.preferred_representation == full.preferred_representation
    assert 0.0 <= decision.joint_cdf_at_direct_quantile <= 1.0
