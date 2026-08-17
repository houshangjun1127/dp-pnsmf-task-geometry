"""Protocol-specific noise accumulation diagnostics for direct DP P-NSMF.

These functions characterize only the Gaussian innovations propagated through
the explicit L2-regularization contraction.  They do not predict ranking
utility and do not equal the difference between private and non-private
training trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2


@dataclass(frozen=True)
class DirectQueryNoiseAccumulation:
    """Second-moment summary for the auxiliary injected-noise process."""

    query_dimension: int
    per_coordinate_variance: float
    expected_squared_frobenius_norm: float
    rms_frobenius_norm: float


@dataclass(frozen=True)
class NoisySignalEnergyAudit:
    """Power and uncertainty summary for a noisy-vector energy statistic."""

    query_dimension: int
    noise_standard_deviation: float
    signal_norm: float
    signal_energy: float
    estimator_standard_deviation: float
    relative_standard_deviation: float
    energy_test_power: float


@dataclass(frozen=True)
class TwoBlockGaussianAllocation:
    """Optimal public two-block Gaussian variances for a quadratic proxy.

    This is an analytic allocation inside the product-of-balls sensitivity
    outer bound.  It is not an exact characterization of the P-NSMF
    sensitivity body.  Every input must be public, fixed in advance, or a
    valid post-processing of an earlier DP release.
    """

    first_variance: float
    second_variance: float
    standardized_sensitivity_squared: float
    optimized_weighted_variance: float
    isotropic_variance: float
    isotropic_weighted_variance: float
    improvement_ratio: float


@dataclass(frozen=True)
class PNSMFCatalogSensitivityBound:
    """Public catalog-dependent upper bound for one P-NSMF contribution."""

    interaction_cap: int
    topk_item_norm_energy: float
    catalog_dependent_squared_bound: float
    max_norm_squared_bound: float
    bound_ratio: float


@dataclass(frozen=True)
class PNSMFAffineSensitivityBounds:
    """Affine public bounds in the current item Frobenius norm.

    For ``x = ||V||_F``, the joint-statistics sensitivity is bounded by
    ``joint_intercept + joint_slope * x``.  The reconstructed direct item
    gradient has the analogous ``direct_*`` bound.  These are conservative
    triangle-inequality envelopes, not exact sensitivities.
    """

    joint_intercept: float
    joint_slope: float
    direct_intercept: float
    direct_slope: float


@dataclass(frozen=True)
class FeedbackRecurrenceBound:
    """Summary of ``x[t+1] <= kappa + rho*x[t] + zeta*x[t]^2``.

    ``attracting_bound`` and ``escape_radius`` are populated only when the
    non-negative recurrence has a certified bounded region.  For an affine
    recurrence, ``escape_radius`` is infinite.  The statement is conditional
    on the caller-supplied noiseless update envelope.
    """

    kappa: float
    rho: float
    zeta: float
    discriminant: float | None
    has_bounded_region: bool
    attracting_bound: float | None
    escape_radius: float | None


@dataclass(frozen=True)
class ParticipationUpperBound:
    """Simultaneous high-probability cap for Poisson-sampled user counts."""

    expected_participants: float
    upper_participants: float
    rounds: int
    failure_probability: float


@dataclass(frozen=True)
class NoiselessItemMapEnvelope:
    """Public affine envelope for one fixed-user P-NSMF item step."""

    participation_upper: float
    hessian_eigenvalue_upper: float
    item_map_slope: float
    item_map_intercept: float


@dataclass(frozen=True)
class GaussianContainmentCertificate:
    """One-step Gaussian ball-containment rejection certificate."""

    query_dimension: int
    update_noise_standard_deviation: float
    feasible_radius: float
    noise_rms_norm: float
    maximum_containment_probability: float
    minimum_exit_probability: float


@dataclass(frozen=True)
class QueryRepresentationNoiseComparison:
    """Expected item-space Gaussian energy for two P-NSMF releases."""

    item_factor_norm: float
    joint_clip_norm: float
    direct_clip_norm: float
    joint_expected_noise_energy: float
    direct_expected_noise_energy: float
    preferred_representation: str
    preferred_to_alternative_ratio: float


@dataclass(frozen=True)
class QueryRepresentationTailComparison:
    """High-probability Gaussian item-noise bounds for two releases."""

    failure_probability: float
    joint_expected_noise_energy: float
    direct_expected_noise_energy: float
    joint_noise_energy_upper: float
    direct_noise_energy_upper: float
    preferred_representation: str
    preferred_to_alternative_ratio: float


@dataclass(frozen=True)
class QueryRepresentationQuantileComparison:
    """Numerically inverted item-noise energy quantiles for two releases."""

    quantile_probability: float
    joint_noise_energy_quantile: float
    direct_noise_energy_quantile: float
    preferred_representation: str
    preferred_to_alternative_ratio: float
    numerical_integration_tolerance: float


@dataclass(frozen=True)
class QueryRepresentationQuantileDecision:
    """Two-way quantile ordering from one joint CDF evaluation."""

    quantile_probability: float
    direct_noise_energy_quantile: float
    joint_cdf_at_direct_quantile: float
    preferred_representation: str
    numerical_integration_tolerance: float


def _as_schedule(
    value: float | Sequence[float], rounds: int, *, name: str
) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        schedule = (float(value),) * rounds
    else:
        schedule = tuple(float(entry) for entry in value)
        if len(schedule) != rounds:
            raise ValueError(f"{name} must contain exactly {rounds} entries.")
    return schedule


def direct_query_noise_accumulation(
    *,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    item_learning_rates: Sequence[float],
    sampling_probabilities: float | Sequence[float],
    clip_norms: float | Sequence[float],
    noise_multipliers: float | Sequence[float],
    regularization: float | Sequence[float] = 0.0,
) -> DirectQueryNoiseAccumulation:
    """Return the exact second moment of the direct-query noise accumulator.

    For the implementation update

        V[t+1] = (1 - 2 * eta[t] * reg[t]) * V[t]
                 - 2 * eta[t] / (q[t] * N) * (G[t] + Z[t]),

    this tracks the auxiliary recursion containing only ``Z[t]``.  All
    schedules must be fixed independently of private, unnoised training data.
    """

    if num_users <= 0 or num_items <= 0 or embedding_dim <= 0:
        raise ValueError("num_users, num_items, and embedding_dim must be positive.")
    learning_rates = tuple(float(value) for value in item_learning_rates)
    if not learning_rates:
        raise ValueError("item_learning_rates must not be empty.")
    rounds = len(learning_rates)
    sampling = _as_schedule(
        sampling_probabilities, rounds, name="sampling_probabilities"
    )
    clipping = _as_schedule(clip_norms, rounds, name="clip_norms")
    multipliers = _as_schedule(
        noise_multipliers, rounds, name="noise_multipliers"
    )
    regularizers = _as_schedule(regularization, rounds, name="regularization")

    variance = 0.0
    for eta, q_value, clip, multiplier, reg in zip(
        learning_rates, sampling, clipping, multipliers, regularizers, strict=True
    ):
        if eta <= 0.0:
            raise ValueError("item learning rates must be positive.")
        if not 0.0 < q_value <= 1.0:
            raise ValueError("sampling probabilities must be in (0, 1].")
        if clip <= 0.0:
            raise ValueError("clip norms must be positive.")
        if multiplier < 0.0 or reg < 0.0:
            raise ValueError("noise multipliers and regularization must be non-negative.")

        contraction = 1.0 - 2.0 * eta * reg
        update_scale = 2.0 * eta / (q_value * num_users)
        noise_std = multiplier * clip
        variance = (
            contraction * contraction * variance
            + update_scale * update_scale * noise_std * noise_std
        )

    dimension = num_items * embedding_dim
    expected_squared_norm = dimension * variance
    return DirectQueryNoiseAccumulation(
        query_dimension=dimension,
        per_coordinate_variance=variance,
        expected_squared_frobenius_norm=expected_squared_norm,
        rms_frobenius_norm=expected_squared_norm**0.5,
    )


def debiased_signal_energy(
    noisy_squared_norm: float, *, query_dimension: int, noise_standard_deviation: float
) -> float:
    """Estimate ``||G||^2`` from ``Y = G + N(0, tau^2 I)``.

    The estimator is unbiased but may be negative.  Truncating it at zero or
    taking its square root changes the estimator and introduces bias.
    """

    if query_dimension <= 0:
        raise ValueError("query_dimension must be positive.")
    if noise_standard_deviation < 0.0:
        raise ValueError("noise_standard_deviation must be non-negative.")
    return noisy_squared_norm - query_dimension * noise_standard_deviation**2


def signal_energy_estimator_variance(
    signal_squared_norm: float,
    *,
    query_dimension: int,
    noise_standard_deviation: float,
) -> float:
    """Return the exact variance of the unbiased signal-energy estimator."""

    if signal_squared_norm < 0.0:
        raise ValueError("signal_squared_norm must be non-negative.")
    if query_dimension <= 0:
        raise ValueError("query_dimension must be positive.")
    if noise_standard_deviation < 0.0:
        raise ValueError("noise_standard_deviation must be non-negative.")
    tau_squared = noise_standard_deviation**2
    return (
        2.0 * query_dimension * tau_squared**2
        + 4.0 * tau_squared * signal_squared_norm
    )


def noisy_energy_test_power(
    signal_norm: float,
    *,
    query_dimension: int,
    noise_standard_deviation: float,
    significance_level: float = 0.05,
) -> float:
    """Power of the upper-tail energy test for ``G = 0`` versus ``G != 0``.

    The statistic ``||Y||^2 / tau^2`` follows a noncentral chi-square
    distribution with noncentrality ``||G||^2 / tau^2``.  This is an
    orientation-agnostic energy test, not a claim of uniformly most powerful
    detection for every structured alternative.
    """

    if signal_norm < 0.0:
        raise ValueError("signal_norm must be non-negative.")
    if query_dimension <= 0:
        raise ValueError("query_dimension must be positive.")
    if noise_standard_deviation <= 0.0:
        raise ValueError("noise_standard_deviation must be positive.")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must be in (0, 1).")
    threshold = chi2.ppf(1.0 - significance_level, df=query_dimension)
    noncentrality = (signal_norm / noise_standard_deviation) ** 2
    return float(ncx2.sf(threshold, df=query_dimension, nc=noncentrality))


def minimum_signal_norm_for_energy_power(
    *,
    query_dimension: int,
    noise_standard_deviation: float,
    target_power: float = 0.8,
    significance_level: float = 0.05,
) -> float:
    """Solve for the signal norm required by the noisy-vector energy test."""

    if not significance_level < target_power < 1.0:
        raise ValueError("target_power must exceed significance_level and be below 1.")

    def objective(signal_norm: float) -> float:
        return noisy_energy_test_power(
            signal_norm,
            query_dimension=query_dimension,
            noise_standard_deviation=noise_standard_deviation,
            significance_level=significance_level,
        ) - target_power

    upper = noise_standard_deviation * query_dimension**0.5
    while objective(upper) < 0.0:
        upper *= 2.0
    return float(brentq(objective, 0.0, upper))


def audit_noisy_signal_energy(
    signal_norm: float,
    *,
    query_dimension: int,
    noise_standard_deviation: float,
    significance_level: float = 0.05,
) -> NoisySignalEnergyAudit:
    """Summarize uncertainty and detection power at a supplied signal norm."""

    signal_energy = signal_norm**2
    standard_deviation = signal_energy_estimator_variance(
        signal_energy,
        query_dimension=query_dimension,
        noise_standard_deviation=noise_standard_deviation,
    ) ** 0.5
    relative = (
        standard_deviation / signal_energy if signal_energy > 0.0 else float("inf")
    )
    return NoisySignalEnergyAudit(
        query_dimension=query_dimension,
        noise_standard_deviation=noise_standard_deviation,
        signal_norm=signal_norm,
        signal_energy=signal_energy,
        estimator_standard_deviation=standard_deviation,
        relative_standard_deviation=relative,
        energy_test_power=noisy_energy_test_power(
            signal_norm,
            query_dimension=query_dimension,
            noise_standard_deviation=noise_standard_deviation,
            significance_level=significance_level,
        ),
    )


def expected_quadratic_loss_inflation(
    *, hessian_trace: float, update_noise_variance: float
) -> float:
    """Return exact expected loss inflation from isotropic parameter noise.

    For a quadratic objective with Hessian ``H`` and a deterministic candidate
    perturbed by ``xi ~ N(0, v I)``, the conditional expected loss increase is
    ``0.5 * v * tr(H)``. Later optimization feedback is outside this identity.
    """

    if hessian_trace < 0.0 or update_noise_variance < 0.0:
        raise ValueError("hessian_trace and update_noise_variance must be non-negative.")
    return 0.5 * update_noise_variance * hessian_trace


def optimal_two_block_gaussian_allocation(
    *,
    first_block_bound: float,
    second_block_bound: float,
    first_quadratic_weight: float,
    second_quadratic_weight: float,
    standardized_sensitivity_budget: float,
) -> TwoBlockGaussianAllocation:
    """Minimize a two-block quadratic noise proxy under a DP constraint.

    Let the contribution satisfy ``||q_1|| <= C_1`` and ``||q_2|| <= C_2``.
    Independent noises with coordinate variances ``s^2`` and ``t^2`` obey the
    sufficient standardized-sensitivity constraint

        C_1^2 / s^2 + C_2^2 / t^2 <= rho^2.

    For positive public weights ``A`` and ``B``, this function returns the
    closed-form minimizer of ``A s^2 + B t^2``.  The privacy accountant still
    has to translate ``rho`` into the intended DP definition and adjacency.
    """

    values = {
        "first_block_bound": first_block_bound,
        "second_block_bound": second_block_bound,
        "first_quadratic_weight": first_quadratic_weight,
        "second_quadratic_weight": second_quadratic_weight,
        "standardized_sensitivity_budget": standardized_sensitivity_budget,
    }
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")

    c_first = float(first_block_bound)
    c_second = float(second_block_bound)
    weight_first = float(first_quadratic_weight)
    weight_second = float(second_quadratic_weight)
    rho_squared = float(standardized_sensitivity_budget) ** 2

    weighted_radius = (
        c_first * weight_first**0.5 + c_second * weight_second**0.5
    )
    first_variance = (
        c_first * weighted_radius / (rho_squared * weight_first**0.5)
    )
    second_variance = (
        c_second * weighted_radius / (rho_squared * weight_second**0.5)
    )
    optimized_cost = weight_first * first_variance + weight_second * second_variance

    isotropic_variance = (c_first**2 + c_second**2) / rho_squared
    isotropic_cost = (weight_first + weight_second) * isotropic_variance
    standardized_sensitivity_squared = (
        c_first**2 / first_variance + c_second**2 / second_variance
    )
    return TwoBlockGaussianAllocation(
        first_variance=first_variance,
        second_variance=second_variance,
        standardized_sensitivity_squared=standardized_sensitivity_squared,
        optimized_weighted_variance=optimized_cost,
        isotropic_variance=isotropic_variance,
        isotropic_weighted_variance=isotropic_cost,
        improvement_ratio=optimized_cost / isotropic_cost,
    )


def pnsmf_catalog_sensitivity_bound(
    *,
    item_factor_norms: Sequence[float],
    interaction_cap: int,
    user_factor_radius: float,
    positive_weight: float,
    missing_weight: float,
    catalog_weight: float,
) -> PNSMFCatalogSensitivityBound:
    """Bound the joint P-NSMF statistic using public item-factor norms.

    The result is valid only when the supplied norms come from public state or
    an earlier DP transcript, and when the protocol enforces both the user
    radius and interaction cap.  It is an upper bound, not the exact
    sensitivity of the coupled query.
    """

    norms = tuple(float(value) for value in item_factor_norms)
    if not norms:
        raise ValueError("item_factor_norms must not be empty.")
    if any(value < 0.0 for value in norms):
        raise ValueError("item factor norms must be non-negative.")
    if not 1 <= interaction_cap <= len(norms):
        raise ValueError("interaction_cap must be between 1 and the catalog size.")
    if user_factor_radius <= 0.0:
        raise ValueError("user_factor_radius must be positive.")
    if positive_weight <= 0.0 or missing_weight < 0.0 or catalog_weight < 0.0:
        raise ValueError(
            "positive_weight must be positive; other weights must be non-negative."
        )

    radius = float(user_factor_radius)
    omega = float(positive_weight)
    alpha = float(missing_weight)
    beta = float(catalog_weight)
    coefficient = abs(omega - alpha * beta)
    largest_norms = sorted(norms, reverse=True)[:interaction_cap]
    topk_energy = sum(value * value for value in largest_norms)
    catalog_sparse_radius = radius * (
        coefficient * radius * topk_energy**0.5
        + omega * interaction_cap**0.5
    )
    catalog_squared = (alpha * radius**2) ** 2 + catalog_sparse_radius**2

    max_norm = max(norms)
    max_norm_sparse_radius = radius * interaction_cap**0.5 * (
        coefficient * radius * max_norm + omega
    )
    max_norm_squared = (alpha * radius**2) ** 2 + max_norm_sparse_radius**2
    return PNSMFCatalogSensitivityBound(
        interaction_cap=interaction_cap,
        topk_item_norm_energy=topk_energy,
        catalog_dependent_squared_bound=catalog_squared,
        max_norm_squared_bound=max_norm_squared,
        bound_ratio=catalog_squared / max_norm_squared,
    )


def pnsmf_affine_sensitivity_bounds(
    *,
    interaction_cap: int,
    user_factor_radius: float,
    positive_weight: float,
    missing_weight: float,
    catalog_weight: float,
) -> PNSMFAffineSensitivityBounds:
    """Return safe affine sensitivity envelopes in ``||V||_F``.

    The joint query is ``(S, M)`` and the direct query is
    ``S + beta * V @ M``.  The result assumes ``||u|| <= R`` and at most
    ``K`` observed items per user.  It does not account for an additional
    explicit clipping operation; fixed clipping can replace the relevant
    intercept/slope pair by ``(C, 0)``.
    """

    if interaction_cap <= 0:
        raise ValueError("interaction_cap must be positive.")
    if user_factor_radius <= 0.0:
        raise ValueError("user_factor_radius must be positive.")
    if positive_weight <= 0.0 or missing_weight < 0.0 or catalog_weight < 0.0:
        raise ValueError(
            "positive_weight must be positive; other weights must be non-negative."
        )

    radius = float(user_factor_radius)
    omega = float(positive_weight)
    alpha = float(missing_weight)
    beta = float(catalog_weight)
    coefficient = abs(omega - alpha * beta)
    sparse_intercept = omega * radius * interaction_cap**0.5
    sparse_slope = coefficient * radius**2
    return PNSMFAffineSensitivityBounds(
        joint_intercept=alpha * radius**2 + sparse_intercept,
        joint_slope=sparse_slope,
        direct_intercept=sparse_intercept,
        direct_slope=sparse_slope + alpha * beta * radius**2,
    )


def gaussian_norm_radius(
    *, dimension: int, events: int, failure_probability: float
) -> float:
    """A simultaneous upper radius for standard Gaussian vector norms.

    By Gaussian concentration and a union bound, all ``events`` vectors obey
    ``||Z|| <= sqrt(d) + sqrt(2 log(events / xi))`` except with probability
    at most ``xi``.  Adaptive conditioning on earlier DP transcripts is valid
    because the bound is applied to each fresh conditional Gaussian draw.
    """

    if dimension <= 0 or events <= 0:
        raise ValueError("dimension and events must be positive.")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must be in (0, 1).")
    return dimension**0.5 + (
        2.0 * math.log(events / failure_probability)
    ) ** 0.5


def poisson_participation_upper_bound(
    *,
    num_users: int,
    sampling_probability: float,
    rounds: int,
    failure_probability: float,
) -> ParticipationUpperBound:
    """Return a simultaneous Bernstein cap for sampled user counts.

    Each round count is ``Binomial(num_users, sampling_probability)`` under
    independent Poisson user sampling.  The cap uses
    ``mu + sqrt(2 mu log(T/xi)) + 2 log(T/xi)/3`` and is truncated at ``N``.
    """

    if num_users <= 0 or rounds <= 0:
        raise ValueError("num_users and rounds must be positive.")
    if not 0.0 < sampling_probability <= 1.0:
        raise ValueError("sampling_probability must be in (0, 1].")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must be in (0, 1).")
    expected = num_users * sampling_probability
    log_term = math.log(rounds / failure_probability)
    upper = min(
        float(num_users),
        expected + (2.0 * expected * log_term) ** 0.5 + 2.0 * log_term / 3.0,
    )
    return ParticipationUpperBound(
        expected_participants=expected,
        upper_participants=upper,
        rounds=rounds,
        failure_probability=failure_probability,
    )


def public_noiseless_item_map_envelope(
    *,
    num_users: int,
    sampling_probability: float,
    participation_upper: float,
    interaction_cap: int,
    user_factor_radius: float,
    positive_weight: float,
    missing_weight: float,
    catalog_weight: float,
    regularization: float,
    item_learning_rate: float,
) -> NoiselessItemMapEnvelope:
    """Bound a fixed-user noiseless P-NSMF item-gradient step.

    Conditional on bounded updated user factors and a sampled-user count no
    larger than ``participation_upper``, the item subproblem is quadratic.
    For ``omega >= alpha*beta``, its Hessian eigenvalues lie in
    ``[2*lambda, L]`` with the public ``L`` returned here.  The affine right
    hand side is bounded using at most ``K`` positive rows per user.

    The envelope is deliberately worst-case.  It may be too loose to serve as
    a practical predictor, which must be checked rather than assumed.
    """

    if num_users <= 0 or interaction_cap <= 0:
        raise ValueError("num_users and interaction_cap must be positive.")
    if not 0.0 < sampling_probability <= 1.0:
        raise ValueError("sampling_probability must be in (0, 1].")
    if not 0.0 <= participation_upper <= num_users:
        raise ValueError("participation_upper must lie between zero and num_users.")
    if user_factor_radius <= 0.0 or positive_weight <= 0.0:
        raise ValueError("user_factor_radius and positive_weight must be positive.")
    if min(missing_weight, catalog_weight, regularization) < 0.0:
        raise ValueError("weights and regularization must be non-negative.")
    if item_learning_rate <= 0.0:
        raise ValueError("item_learning_rate must be positive.")
    if positive_weight < missing_weight * catalog_weight:
        raise ValueError(
            "The public PSD Hessian envelope requires omega >= alpha * beta."
        )

    expected = sampling_probability * num_users
    radius_squared = user_factor_radius**2
    # alpha*beta plus the observed correction omega-alpha*beta equals omega.
    hessian_upper = (
        2.0
        * positive_weight
        * participation_upper
        * radius_squared
        / expected
        + 2.0 * regularization
    )
    minimum_eigenvalue = 2.0 * regularization
    slope = max(
        abs(1.0 - item_learning_rate * minimum_eigenvalue),
        abs(1.0 - item_learning_rate * hessian_upper),
    )
    update_scale = 2.0 * item_learning_rate / expected
    intercept = (
        update_scale
        * participation_upper
        * positive_weight
        * user_factor_radius
        * interaction_cap**0.5
    )
    return NoiselessItemMapEnvelope(
        participation_upper=participation_upper,
        hessian_eigenvalue_upper=hessian_upper,
        item_map_slope=slope,
        item_map_intercept=intercept,
    )


def _summarize_feedback_recurrence(
    *, kappa: float, rho: float, zeta: float
) -> FeedbackRecurrenceBound:
    if min(kappa, rho, zeta) < 0.0:
        raise ValueError("recurrence coefficients must be non-negative.")
    if zeta == 0.0:
        stable = rho < 1.0
        return FeedbackRecurrenceBound(
            kappa=kappa,
            rho=rho,
            zeta=zeta,
            discriminant=None,
            has_bounded_region=stable,
            attracting_bound=(kappa / (1.0 - rho) if stable else None),
            escape_radius=(float("inf") if stable else None),
        )

    discriminant = (1.0 - rho) ** 2 - 4.0 * zeta * kappa
    stable = rho < 1.0 and discriminant >= 0.0
    if not stable:
        lower = None
        upper = None
    else:
        root = discriminant**0.5
        lower = (1.0 - rho - root) / (2.0 * zeta)
        upper = (1.0 - rho + root) / (2.0 * zeta)
    return FeedbackRecurrenceBound(
        kappa=kappa,
        rho=rho,
        zeta=zeta,
        discriminant=discriminant,
        has_bounded_region=stable,
        attracting_bound=lower,
        escape_radius=upper,
    )


def direct_query_feedback_bound(
    *,
    num_items: int,
    embedding_dim: int,
    rounds: int,
    failure_probability: float,
    update_scale: float,
    noise_multiplier: float,
    sensitivity_intercept: float,
    sensitivity_slope: float,
    noiseless_intercept: float,
    noiseless_slope: float,
) -> FeedbackRecurrenceBound:
    """Bound direct-query scale feedback under an affine sensitivity.

    The caller must establish the noiseless envelope
    ``||F_t(V)|| <= noiseless_intercept + noiseless_slope * ||V||_F``.
    The returned coefficients hold simultaneously for the requested number
    of fresh Gaussian releases with the stated failure probability.
    """

    values = (
        update_scale,
        noise_multiplier,
        sensitivity_intercept,
        sensitivity_slope,
        noiseless_intercept,
        noiseless_slope,
    )
    if any(value < 0.0 for value in values):
        raise ValueError("scales, bounds, and envelope coefficients must be non-negative.")
    radius = gaussian_norm_radius(
        dimension=num_items * embedding_dim,
        events=rounds,
        failure_probability=failure_probability,
    )
    noise_factor = update_scale * noise_multiplier * radius
    return _summarize_feedback_recurrence(
        kappa=noiseless_intercept + noise_factor * sensitivity_intercept,
        rho=noiseless_slope + noise_factor * sensitivity_slope,
        zeta=0.0,
    )


def joint_statistics_feedback_bound(
    *,
    num_items: int,
    embedding_dim: int,
    rounds: int,
    failure_probability: float,
    update_scale: float,
    noise_multiplier: float,
    catalog_weight: float,
    sensitivity_intercept: float,
    sensitivity_slope: float,
    noiseless_intercept: float,
    noiseless_slope: float,
) -> FeedbackRecurrenceBound:
    """Bound joint-statistics scale feedback including ``V @ Z_M`` noise.

    In isometric symmetric coordinates, the moment block has dimension
    ``d(d+1)/2``.  The proof uses ``||V Z_M||_F <= ||V||_F ||Z_M||_F``.
    Consequently an affine catalog sensitivity produces a quadratic scale
    recurrence, unlike direct-gradient release.
    """

    values = (
        update_scale,
        noise_multiplier,
        catalog_weight,
        sensitivity_intercept,
        sensitivity_slope,
        noiseless_intercept,
        noiseless_slope,
    )
    if any(value < 0.0 for value in values):
        raise ValueError("scales, bounds, and envelope coefficients must be non-negative.")
    events = 2 * rounds
    sparse_radius = gaussian_norm_radius(
        dimension=num_items * embedding_dim,
        events=events,
        failure_probability=failure_probability,
    )
    moment_radius = gaussian_norm_radius(
        dimension=embedding_dim * (embedding_dim + 1) // 2,
        events=events,
        failure_probability=failure_probability,
    )
    noise_factor = update_scale * noise_multiplier
    kappa = (
        noiseless_intercept
        + noise_factor * sensitivity_intercept * sparse_radius
    )
    rho = noiseless_slope + noise_factor * (
        sensitivity_slope * sparse_radius
        + catalog_weight * sensitivity_intercept * moment_radius
    )
    zeta = (
        noise_factor
        * catalog_weight
        * sensitivity_slope
        * moment_radius
    )
    return _summarize_feedback_recurrence(kappa=kappa, rho=rho, zeta=zeta)


def gaussian_ball_containment_certificate(
    *,
    query_dimension: int,
    update_noise_standard_deviation: float,
    feasible_radius: float,
) -> GaussianContainmentCertificate:
    """Certify a one-step lower bound on leaving a Frobenius ball.

    For a conditional update ``mu + tau * Z`` with ``Z ~ N(0, I_D)``, a
    centered Gaussian maximizes the probability of every origin-centered
    Euclidean ball.  Hence

        P(||mu + tau Z|| <= B) <= P(||tau Z|| <= B).

    The right side is an exact chi-square CDF.  A joint-statistics release also
    inherits this certificate from its independent isotropic sparse block;
    extra independent moment-block covariance cannot improve containment.
    The radius must be justified independently of private outcomes.
    """

    if query_dimension <= 0:
        raise ValueError("query_dimension must be positive.")
    if update_noise_standard_deviation < 0.0:
        raise ValueError("update_noise_standard_deviation must be non-negative.")
    if feasible_radius < 0.0:
        raise ValueError("feasible_radius must be non-negative.")
    if update_noise_standard_deviation == 0.0:
        containment = 1.0
    else:
        threshold = (feasible_radius / update_noise_standard_deviation) ** 2
        containment = float(chi2.cdf(threshold, df=query_dimension))
    return GaussianContainmentCertificate(
        query_dimension=query_dimension,
        update_noise_standard_deviation=update_noise_standard_deviation,
        feasible_radius=feasible_radius,
        noise_rms_norm=(
            update_noise_standard_deviation * query_dimension**0.5
        ),
        maximum_containment_probability=containment,
        minimum_exit_probability=1.0 - containment,
    )


def minimum_feasible_radius_for_gaussian_containment(
    *,
    query_dimension: int,
    update_noise_standard_deviation: float,
    target_containment_probability: float,
) -> float:
    """Return the radius needed by centered isotropic Gaussian noise alone."""

    if query_dimension <= 0:
        raise ValueError("query_dimension must be positive.")
    if update_noise_standard_deviation < 0.0:
        raise ValueError("update_noise_standard_deviation must be non-negative.")
    if not 0.0 < target_containment_probability < 1.0:
        raise ValueError("target_containment_probability must be in (0, 1).")
    return float(
        update_noise_standard_deviation
        * chi2.ppf(target_containment_probability, df=query_dimension) ** 0.5
    )


def minimum_projection_displacement_for_gaussian_update(
    *,
    query_dimension: int,
    update_noise_standard_deviation: float,
    projection_radius: float,
    target_probability: float,
) -> float:
    """Lower-bound radial ball-projection displacement with given probability.

    For ``Y = mu + tau Z`` and Euclidean projection onto the radius-``B``
    ball, Anderson's inequality gives, for every conditional mean ``mu``,

        P(||Y - Proj_B(Y)|| >= r_p - B) >= p,

    where ``r_p`` is the lower ``p`` survival quantile of ``||tau Z||``.
    The returned bound is truncated at zero.  It describes intervention by
    post-noise projection, not recommendation utility.
    """

    if query_dimension <= 0:
        raise ValueError("query_dimension must be positive.")
    if update_noise_standard_deviation < 0.0:
        raise ValueError("update_noise_standard_deviation must be non-negative.")
    if projection_radius < 0.0:
        raise ValueError("projection_radius must be non-negative.")
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be in (0, 1).")
    lower_survival_radius = update_noise_standard_deviation * chi2.ppf(
        1.0 - target_probability,
        df=query_dimension,
    ) ** 0.5
    return float(max(0.0, lower_survival_radius - projection_radius))


def compare_pnsmf_query_noise_energy(
    *,
    num_items: int,
    embedding_dim: int,
    item_factor_norm: float,
    catalog_weight: float,
    joint_clip_norm: float,
    direct_clip_norm: float,
    update_scale: float,
    noise_multiplier: float,
) -> QueryRepresentationNoiseComparison:
    """Compare exact expected Gaussian energy after item reconstruction.

    The joint release adds isotropic noise in the direct sum of the sparse
    item block and the isometrically half-vectorized symmetric moment block.
    If each packed coordinate has standard deviation ``tau_J``, then

        E||Z_S + beta V Z_M||_F^2
        = tau_J^2 [m d + beta^2 (d + 1) ||V||_F^2 / 2].

    A direct item-gradient release has energy ``tau_G^2 m d``.  The function
    compares noise only; it does not compare clipping bias or utility.  All
    supplied norms must be public, fixed in advance, or based on a previous DP
    transcript before it is used for adaptive mechanism selection.
    """

    if num_items <= 0 or embedding_dim <= 0:
        raise ValueError("num_items and embedding_dim must be positive.")
    if min(item_factor_norm, catalog_weight) < 0.0:
        raise ValueError("item_factor_norm and catalog_weight must be non-negative.")
    if joint_clip_norm <= 0.0 or direct_clip_norm <= 0.0:
        raise ValueError("clip norms must be positive.")
    if update_scale < 0.0 or noise_multiplier < 0.0:
        raise ValueError("update_scale and noise_multiplier must be non-negative.")

    common = (update_scale * noise_multiplier) ** 2
    sparse_dimension = num_items * embedding_dim
    moment_reconstruction_factor = (
        catalog_weight**2
        * (embedding_dim + 1.0)
        * item_factor_norm**2
        / 2.0
    )
    joint_energy = common * joint_clip_norm**2 * (
        sparse_dimension + moment_reconstruction_factor
    )
    direct_energy = common * direct_clip_norm**2 * sparse_dimension
    if joint_energy < direct_energy:
        preferred = "joint_sufficient_statistics"
        ratio = joint_energy / direct_energy if direct_energy > 0.0 else 0.0
    else:
        preferred = "direct_item_gradient"
        ratio = direct_energy / joint_energy if joint_energy > 0.0 else 0.0
    return QueryRepresentationNoiseComparison(
        item_factor_norm=float(item_factor_norm),
        joint_clip_norm=float(joint_clip_norm),
        direct_clip_norm=float(direct_clip_norm),
        joint_expected_noise_energy=float(joint_energy),
        direct_expected_noise_energy=float(direct_energy),
        preferred_representation=preferred,
        preferred_to_alternative_ratio=float(ratio),
    )


def compare_pnsmf_query_noise_energy_tail(
    *,
    num_items: int,
    item_gram_eigenvalues: Sequence[float],
    catalog_weight: float,
    joint_clip_norm: float,
    direct_clip_norm: float,
    update_scale: float,
    noise_multiplier: float,
    failure_probability: float,
) -> QueryRepresentationTailComparison:
    """Compare Gaussian quadratic-form upper tails in item-update space.

    Let ``g_i`` be the eigenvalues of the public Gram matrix ``V.T @ V``.
    In the Frobenius-isometric symmetric coordinates, the squared singular
    values of ``Z -> V Z`` are ``g_i`` and ``(g_i + g_j) / 2`` for ``i < j``.
    This yields the exact trace, squared trace, and operator norm of the joint
    Gaussian covariance without constructing an ``(m d) x (m d)`` matrix.

    For ``X ~ N(0, Sigma)``, the returned upper bound applies

        ||X||^2 <= tr(Sigma)
                    + 2 sqrt(tr(Sigma^2) log(1 / delta))
                    + 2 ||Sigma||_op log(1 / delta)

    with probability at least ``1 - delta``.  Both representations use the
    same concentration inequality, so the comparison is calibrated on equal
    terms.  It remains a noise-only certificate, not a utility guarantee.
    """

    gram = tuple(float(value) for value in item_gram_eigenvalues)
    if num_items <= 0 or not gram:
        raise ValueError("num_items and item_gram_eigenvalues must be non-empty.")
    if any(value < -1e-12 for value in gram):
        raise ValueError("item Gram eigenvalues must be non-negative.")
    gram = tuple(max(0.0, value) for value in gram)
    if catalog_weight < 0.0:
        raise ValueError("catalog_weight must be non-negative.")
    if joint_clip_norm <= 0.0 or direct_clip_norm <= 0.0:
        raise ValueError("clip norms must be positive.")
    if update_scale < 0.0 or noise_multiplier < 0.0:
        raise ValueError("update_scale and noise_multiplier must be non-negative.")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must be in (0, 1).")

    dimension = len(gram)
    query_dimension = num_items * dimension
    moment_spectrum = list(gram)
    moment_spectrum.extend(
        (gram[first] + gram[second]) / 2.0
        for first in range(dimension)
        for second in range(first + 1, dimension)
    )
    spectrum_sum = sum(moment_spectrum)
    spectrum_square_sum = sum(value * value for value in moment_spectrum)
    spectrum_max = max(moment_spectrum)

    joint_variance = (
        update_scale * noise_multiplier * joint_clip_norm
    ) ** 2
    direct_variance = (
        update_scale * noise_multiplier * direct_clip_norm
    ) ** 2
    beta_squared = catalog_weight**2
    joint_trace = joint_variance * (
        query_dimension + beta_squared * spectrum_sum
    )
    joint_trace_square = joint_variance**2 * (
        query_dimension
        + 2.0 * beta_squared * spectrum_sum
        + beta_squared**2 * spectrum_square_sum
    )
    joint_operator = joint_variance * (
        1.0 + beta_squared * spectrum_max
    )
    direct_trace = direct_variance * query_dimension
    direct_trace_square = direct_variance**2 * query_dimension
    direct_operator = direct_variance
    log_inverse_failure = math.log(1.0 / failure_probability)

    def upper(trace: float, trace_square: float, operator: float) -> float:
        return (
            trace
            + 2.0 * math.sqrt(trace_square * log_inverse_failure)
            + 2.0 * operator * log_inverse_failure
        )

    joint_upper = upper(joint_trace, joint_trace_square, joint_operator)
    direct_upper = upper(direct_trace, direct_trace_square, direct_operator)
    if joint_upper < direct_upper:
        preferred = "joint_sufficient_statistics"
        ratio = joint_upper / direct_upper if direct_upper > 0.0 else 0.0
    else:
        preferred = "direct_item_gradient"
        ratio = direct_upper / joint_upper if joint_upper > 0.0 else 0.0
    return QueryRepresentationTailComparison(
        failure_probability=float(failure_probability),
        joint_expected_noise_energy=float(joint_trace),
        direct_expected_noise_energy=float(direct_trace),
        joint_noise_energy_upper=float(joint_upper),
        direct_noise_energy_upper=float(direct_upper),
        preferred_representation=preferred,
        preferred_to_alternative_ratio=float(ratio),
    )


def _positive_weighted_chi_square_cdf(
    value: float,
    *,
    weights: Sequence[float],
    degrees_of_freedom: Sequence[float],
    absolute_tolerance: float,
) -> float:
    """Evaluate a positive weighted chi-square CDF by Imhof inversion.

    This is a deterministic numerical use of the classical characteristic-
    function inversion formula, not a new distributional algorithm.  Terms
    with the same weight may be grouped by adding their degrees of freedom.
    """

    weight_array = tuple(float(weight) for weight in weights)
    degree_array = tuple(float(degree) for degree in degrees_of_freedom)
    if len(weight_array) != len(degree_array) or not weight_array:
        raise ValueError("weights and degrees_of_freedom must have equal nonzero length.")
    if any(weight <= 0.0 for weight in weight_array):
        raise ValueError("all weights must be positive.")
    if any(degree <= 0.0 for degree in degree_array):
        raise ValueError("all degrees of freedom must be positive.")
    if value <= 0.0:
        return 0.0
    if absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be positive.")

    grouped_degrees: dict[float, float] = {}
    for weight, degree in zip(weight_array, degree_array):
        grouped_degrees[weight] = grouped_degrees.get(weight, 0.0) + degree
    weight_array = tuple(grouped_degrees)
    degree_array = tuple(grouped_degrees.values())
    total_degrees = sum(degree_array)
    mean = sum(
        weight * degree
        for weight, degree in zip(weight_array, degree_array)
    )
    scale = mean / total_degrees
    normalized_weights = np.asarray(
        [weight / scale for weight in weight_array], dtype=np.float64
    )
    degree_vector = np.asarray(degree_array, dtype=np.float64)
    normalized_value = value / scale

    def integrand(frequency: float) -> float:
        if frequency == 0.0:
            return total_degrees - normalized_value
        log_characteristic = -0.5 * np.dot(
            degree_vector,
            np.log(1.0 - 2.0j * normalized_weights * frequency),
        )
        phase = log_characteristic - 1j * frequency * normalized_value
        return math.e ** phase.real * math.sin(phase.imag) / frequency

    upper = 0.125 / float(np.max(normalized_weights))
    while upper < 128.0:
        log_amplitude = -0.25 * float(
            np.dot(
                degree_vector,
                np.log1p(4.0 * (normalized_weights * upper) ** 2),
            )
        )
        if math.exp(log_amplitude) / upper < absolute_tolerance * 0.1:
            break
        upper *= 2.0
    integral, _ = quad(
        integrand,
        0.0,
        upper,
        epsabs=absolute_tolerance * math.pi / 4.0,
        epsrel=absolute_tolerance,
        limit=500,
        points=(0.0,),
    )
    return float(min(1.0, max(0.0, 0.5 - integral / math.pi)))


def compare_pnsmf_query_noise_energy_quantile(
    *,
    num_items: int,
    item_gram_eigenvalues: Sequence[float],
    catalog_weight: float,
    joint_clip_norm: float,
    direct_clip_norm: float,
    update_scale: float,
    noise_multiplier: float,
    quantile_probability: float = 0.95,
    numerical_integration_tolerance: float = 1e-9,
) -> QueryRepresentationQuantileComparison:
    """Compare numerically inverted Gaussian item-noise energy quantiles.

    The joint covariance has ``D-r`` repeated baseline eigenvalues and only
    ``r=d(d+1)/2`` modified eigenvalues.  Hence its squared norm is a positive
    weighted sum of one grouped chi-square variable and ``r`` chi-square(1)
    variables.  Imhof inversion depends on ``O(d^2)`` weights, not ``m*d``
    explicit covariance eigenvalues.  The direct quantile is an ordinary
    chi-square quantile.  This function is a noise-only diagnostic.
    """

    gram = tuple(float(value) for value in item_gram_eigenvalues)
    if num_items <= 0 or not gram:
        raise ValueError("num_items and item_gram_eigenvalues must be non-empty.")
    if any(value < -1e-12 for value in gram):
        raise ValueError("item Gram eigenvalues must be non-negative.")
    gram = tuple(max(0.0, value) for value in gram)
    if catalog_weight < 0.0:
        raise ValueError("catalog_weight must be non-negative.")
    if joint_clip_norm <= 0.0 or direct_clip_norm <= 0.0:
        raise ValueError("clip norms must be positive.")
    if update_scale < 0.0 or noise_multiplier < 0.0:
        raise ValueError("update_scale and noise_multiplier must be non-negative.")
    if not 0.0 < quantile_probability < 1.0:
        raise ValueError("quantile_probability must be in (0, 1).")
    if numerical_integration_tolerance <= 0.0:
        raise ValueError("numerical_integration_tolerance must be positive.")

    dimension = len(gram)
    query_dimension = num_items * dimension
    moment_spectrum = list(gram)
    moment_spectrum.extend(
        (gram[first] + gram[second]) / 2.0
        for first in range(dimension)
        for second in range(first + 1, dimension)
    )
    modified_rank = len(moment_spectrum)
    residual_degrees = query_dimension - modified_rank
    if residual_degrees <= 0:
        raise ValueError("num_items * dimension must exceed d(d+1)/2.")

    beta_squared = catalog_weight**2
    shape_weights = [1.0]
    shape_degrees = [float(residual_degrees)]
    shape_weights.extend(1.0 + beta_squared * value for value in moment_spectrum)
    shape_degrees.extend([1.0] * modified_rank)
    shape_mean = sum(
        weight * degree
        for weight, degree in zip(shape_weights, shape_degrees)
    )
    shape_trace_square = sum(
        weight * weight * degree
        for weight, degree in zip(shape_weights, shape_degrees)
    )
    failure_probability = 1.0 - quantile_probability
    log_inverse_failure = math.log(1.0 / failure_probability)
    upper = (
        shape_mean
        + 2.0 * math.sqrt(shape_trace_square * log_inverse_failure)
        + 2.0 * max(shape_weights) * log_inverse_failure
    )
    cdf = lambda value: _positive_weighted_chi_square_cdf(
        value,
        weights=shape_weights,
        degrees_of_freedom=shape_degrees,
        absolute_tolerance=numerical_integration_tolerance,
    )
    while cdf(upper) < quantile_probability:
        upper *= 2.0
    joint_shape_quantile = brentq(
        lambda value: cdf(value) - quantile_probability,
        0.0,
        upper,
        xtol=numerical_integration_tolerance,
        rtol=max(4.0 * 2.220446049250313e-16, numerical_integration_tolerance),
    )
    joint_variance = (
        update_scale * noise_multiplier * joint_clip_norm
    ) ** 2
    direct_variance = (
        update_scale * noise_multiplier * direct_clip_norm
    ) ** 2
    joint_quantile = joint_variance * joint_shape_quantile
    direct_quantile = direct_variance * chi2.ppf(
        quantile_probability, df=query_dimension
    )
    if joint_quantile < direct_quantile:
        preferred = "joint_sufficient_statistics"
        ratio = joint_quantile / direct_quantile if direct_quantile > 0.0 else 0.0
    else:
        preferred = "direct_item_gradient"
        ratio = direct_quantile / joint_quantile if joint_quantile > 0.0 else 0.0
    return QueryRepresentationQuantileComparison(
        quantile_probability=float(quantile_probability),
        joint_noise_energy_quantile=float(joint_quantile),
        direct_noise_energy_quantile=float(direct_quantile),
        preferred_representation=preferred,
        preferred_to_alternative_ratio=float(ratio),
        numerical_integration_tolerance=float(numerical_integration_tolerance),
    )


def select_pnsmf_query_noise_energy_quantile(
    *,
    num_items: int,
    item_gram_eigenvalues: Sequence[float],
    catalog_weight: float,
    joint_clip_norm: float,
    direct_clip_norm: float,
    update_scale: float,
    noise_multiplier: float,
    quantile_probability: float = 0.95,
    numerical_integration_tolerance: float = 1e-9,
) -> QueryRepresentationQuantileDecision:
    """Order two noise-energy quantiles with one Imhof CDF evaluation.

    If ``q_G`` is the direct-interface quantile, monotonicity gives
    ``q_J <= q_G`` exactly when ``F_J(q_G) >= probability``.  This avoids a
    per-round inverse-CDF solve.  It reuses a classical numerical distribution
    method and should not be described as a new generalized-chi-square method.
    """

    gram = tuple(float(value) for value in item_gram_eigenvalues)
    if num_items <= 0 or not gram:
        raise ValueError("num_items and item_gram_eigenvalues must be non-empty.")
    if any(value < -1e-12 for value in gram):
        raise ValueError("item Gram eigenvalues must be non-negative.")
    gram = tuple(max(0.0, value) for value in gram)
    if catalog_weight < 0.0:
        raise ValueError("catalog_weight must be non-negative.")
    if joint_clip_norm <= 0.0 or direct_clip_norm <= 0.0:
        raise ValueError("clip norms must be positive.")
    if update_scale < 0.0 or noise_multiplier < 0.0:
        raise ValueError("update_scale and noise_multiplier must be non-negative.")
    if not 0.0 < quantile_probability < 1.0:
        raise ValueError("quantile_probability must be in (0, 1).")
    if numerical_integration_tolerance <= 0.0:
        raise ValueError("numerical_integration_tolerance must be positive.")

    dimension = len(gram)
    query_dimension = num_items * dimension
    moment_spectrum = list(gram)
    moment_spectrum.extend(
        (gram[first] + gram[second]) / 2.0
        for first in range(dimension)
        for second in range(first + 1, dimension)
    )
    residual_degrees = query_dimension - len(moment_spectrum)
    if residual_degrees <= 0:
        raise ValueError("num_items * dimension must exceed d(d+1)/2.")
    beta_squared = catalog_weight**2
    shape_weights = [1.0]
    shape_degrees = [float(residual_degrees)]
    shape_weights.extend(1.0 + beta_squared * value for value in moment_spectrum)
    shape_degrees.extend([1.0] * len(moment_spectrum))

    joint_variance = (
        update_scale * noise_multiplier * joint_clip_norm
    ) ** 2
    direct_variance = (
        update_scale * noise_multiplier * direct_clip_norm
    ) ** 2
    direct_quantile = direct_variance * chi2.ppf(
        quantile_probability, df=query_dimension
    )
    if joint_variance == 0.0:
        joint_cdf = 1.0
    else:
        joint_cdf = _positive_weighted_chi_square_cdf(
            direct_quantile / joint_variance,
            weights=shape_weights,
            degrees_of_freedom=shape_degrees,
            absolute_tolerance=numerical_integration_tolerance,
        )
    preferred = (
        "joint_sufficient_statistics"
        if joint_cdf >= quantile_probability and direct_quantile > 0.0
        else "direct_item_gradient"
    )
    return QueryRepresentationQuantileDecision(
        quantile_probability=float(quantile_probability),
        direct_noise_energy_quantile=float(direct_quantile),
        joint_cdf_at_direct_quantile=float(joint_cdf),
        preferred_representation=preferred,
        numerical_integration_tolerance=float(numerical_integration_tolerance),
    )
