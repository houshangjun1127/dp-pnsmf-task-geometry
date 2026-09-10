"""User-level privacy accounting for Poisson-sampled Gaussian mechanisms.

This module deliberately supports only the sampling assumption that has been
verified here. Fixed-size sampling without replacement must not be silently
treated as Poisson sampling in paper experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dp_accounting import dp_event, rdp


DEFAULT_RDP_ORDERS: tuple[float, ...] = tuple(range(2, 65)) + (128.0, 256.0)


@dataclass(frozen=True)
class RdpPrivacyConfig:
    """Configuration for user-level RDP accounting."""

    sampling_probability: float
    noise_multiplier: float
    steps: int
    delta: float
    orders: Sequence[float] = DEFAULT_RDP_ORDERS

    def validate(self) -> None:
        if not 0.0 < self.sampling_probability <= 1.0:
            raise ValueError("sampling_probability must lie in (0, 1].")
        if self.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be positive.")
        if self.steps <= 0:
            raise ValueError("steps must be a positive integer.")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1).")
        if not self.orders:
            raise ValueError("orders must not be empty.")
        if any(order <= 1.0 for order in self.orders):
            raise ValueError("Every Renyi order must be greater than 1.")


@dataclass(frozen=True)
class GaussianCompositionPrivacyConfig:
    """RDP accounting for repeated, non-subsampled Gaussian mechanisms."""

    noise_multiplier: float
    compositions: int
    delta: float
    orders: Sequence[float] = DEFAULT_RDP_ORDERS

    def validate(self) -> None:
        if self.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be positive.")
        if self.compositions <= 0:
            raise ValueError("compositions must be a positive integer.")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1).")
        if not self.orders or any(order <= 1.0 for order in self.orders):
            raise ValueError("Every Renyi order must be greater than 1.")


def compute_epsilon(config: RdpPrivacyConfig) -> float:
    """Return epsilon for repeated Poisson-sampled Gaussian mechanisms."""

    config.validate()
    accountant = rdp.RdpAccountant(orders=list(config.orders))
    event = dp_event.PoissonSampledDpEvent(
        config.sampling_probability,
        dp_event.GaussianDpEvent(config.noise_multiplier),
    )
    accountant.compose(event, count=config.steps)
    epsilon = float(accountant.get_epsilon(config.delta))
    if epsilon <= 0.0:
        raise RuntimeError("The accountant returned a non-positive epsilon.")
    return epsilon


def compute_composed_gaussian_epsilon(
    config: GaussianCompositionPrivacyConfig,
) -> float:
    """Return epsilon for repeated Gaussian mechanisms without subsampling."""

    config.validate()
    accountant = rdp.RdpAccountant(orders=list(config.orders))
    accountant.compose(
        dp_event.GaussianDpEvent(config.noise_multiplier),
        count=config.compositions,
    )
    epsilon = float(accountant.get_epsilon(config.delta))
    if epsilon <= 0.0:
        raise RuntimeError("The accountant returned a non-positive epsilon.")
    return epsilon


def calibrate_composed_gaussian_noise_multiplier(
    *,
    target_epsilon: float,
    compositions: int,
    delta: float,
    orders: Sequence[float] = DEFAULT_RDP_ORDERS,
    relative_tolerance: float = 1e-10,
) -> float:
    """Calibrate a Gaussian multiplier to a target epsilon by bisection.

    The returned value is the smallest endpoint found whose computed epsilon
    does not exceed the target, up to the requested relative tolerance.
    """

    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive.")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive.")
    template = dict(compositions=compositions, delta=delta, orders=orders)
    lower = 1e-6
    upper = 1.0
    while compute_composed_gaussian_epsilon(
        GaussianCompositionPrivacyConfig(noise_multiplier=upper, **template)
    ) > target_epsilon:
        upper *= 2.0
        if upper > 1e9:
            raise RuntimeError("Failed to bracket a Gaussian noise multiplier.")
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        epsilon = compute_composed_gaussian_epsilon(
            GaussianCompositionPrivacyConfig(
                noise_multiplier=midpoint, **template
            )
        )
        if epsilon > target_epsilon:
            lower = midpoint
        else:
            upper = midpoint
        if (upper - lower) / upper <= relative_tolerance:
            break
    return upper
