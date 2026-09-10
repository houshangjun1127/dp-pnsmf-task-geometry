import pytest
import torch

from src.privacy import (
    GaussianCompositionPrivacyConfig,
    RdpPrivacyConfig,
    calibrate_composed_gaussian_noise_multiplier,
    clip_client_updates,
    compute_composed_gaussian_epsilon,
    compute_epsilon,
    public_model_scaled_clip_norm,
)


def test_uniform_clipping_respects_shared_bound() -> None:
    generator = torch.Generator().manual_seed(20260729)
    updates = torch.randn(64, 128, generator=generator) * 4.0

    clipped, diagnostics = clip_client_updates(updates, max_norm=2.0)

    assert clipped.shape == updates.shape
    assert diagnostics.post_clip_norm.max().item() <= 2.0 + 1e-5
    assert torch.all((diagnostics.clipping_factor >= 0.0) & (diagnostics.clipping_factor <= 1.0))
    assert torch.allclose(
        diagnostics.post_clip_norm,
        torch.minimum(diagnostics.pre_clip_norm, torch.tensor(2.0)),
        atol=1e-5,
    )


def test_zero_update_is_stable() -> None:
    updates = torch.zeros(3, 5)

    clipped, diagnostics = clip_client_updates(updates, max_norm=1.0)

    assert torch.equal(clipped, updates)
    assert torch.equal(diagnostics.clipping_factor, torch.ones(3))
    assert torch.equal(diagnostics.cosine_similarity, torch.ones(3))


def test_privacy_epsilon_is_monotone() -> None:
    base = dict(sampling_probability=0.1, delta=1e-5)
    epsilon_20 = compute_epsilon(RdpPrivacyConfig(**base, noise_multiplier=1.2, steps=20))
    epsilon_40 = compute_epsilon(RdpPrivacyConfig(**base, noise_multiplier=1.2, steps=40))
    epsilon_more_noise = compute_epsilon(
        RdpPrivacyConfig(**base, noise_multiplier=2.0, steps=20)
    )

    assert epsilon_40 > epsilon_20 > epsilon_more_noise > 0.0
    assert epsilon_20 == pytest.approx(3.0144711402, rel=1e-8)


def test_privacy_epsilon_increases_with_poisson_sampling_probability() -> None:
    base = dict(noise_multiplier=1.2, steps=20, delta=1e-5)
    epsilon_low_sampling = compute_epsilon(
        RdpPrivacyConfig(**base, sampling_probability=0.05)
    )
    epsilon_high_sampling = compute_epsilon(
        RdpPrivacyConfig(**base, sampling_probability=0.10)
    )
    assert epsilon_high_sampling > epsilon_low_sampling > 0.0


def test_composed_gaussian_calibration_reaches_target_conservatively() -> None:
    target = 2.193078238018666
    sigma = calibrate_composed_gaussian_noise_multiplier(
        target_epsilon=target,
        compositions=210,
        delta=1e-5,
    )
    epsilon = compute_composed_gaussian_epsilon(
        GaussianCompositionPrivacyConfig(
            noise_multiplier=sigma,
            compositions=210,
            delta=1e-5,
        )
    )
    epsilon_less_noise = compute_composed_gaussian_epsilon(
        GaussianCompositionPrivacyConfig(
            noise_multiplier=sigma * (1.0 - 1e-6),
            compositions=210,
            delta=1e-5,
        )
    )
    assert epsilon <= target
    assert epsilon == pytest.approx(target, rel=1e-8)
    assert epsilon_less_noise > epsilon


def test_public_model_clip_schedule_obeys_growth_and_absolute_caps() -> None:
    items = torch.full((4, 2), 10.0)
    growth_limited = public_model_scaled_clip_norm(
        items,
        scale_ratio=0.5,
        previous_clip_norm=1.0,
        max_growth_factor=1.1,
        max_clip_norm=5.0,
    )
    absolute_limited = public_model_scaled_clip_norm(
        items,
        scale_ratio=0.5,
        max_clip_norm=5.0,
    )
    assert growth_limited == pytest.approx(1.1)
    assert absolute_limited == pytest.approx(5.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sampling_probability", 0.0),
        ("noise_multiplier", 0.0),
        ("steps", 0),
        ("delta", 1.0),
    ],
)
def test_invalid_privacy_config_is_rejected(field: str, value: float) -> None:
    kwargs = {
        "sampling_probability": 0.1,
        "noise_multiplier": 1.2,
        "steps": 20,
        "delta": 1e-5,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        compute_epsilon(RdpPrivacyConfig(**kwargs))
