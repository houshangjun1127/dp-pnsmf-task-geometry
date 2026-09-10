"""Validate expected DP objective inflation at real final factor states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import get_pnsmf_fixed_dataset_spec, load_pnsmf_fixed_split
from src.models import (
    item_subproblem_hessian_trace,
    weighted_nsmf_loss_indexed,
)
from src.privacy import expected_quadratic_loss_inflation


DATA_DIRECTORIES = {
    "ml1m": "ML1M-TXT-FORMAT",
    "amazon_kindle": "Amazon_Kindle_Store-TXT-FORMAT",
    "netflix5k5k": "Netflix5K5K-TXT-FORAMT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registered-result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260931)
    return parser.parse_args()


def _load_factors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    archive = np.load(path)
    return (
        torch.as_tensor(archive["user_factors"], dtype=torch.float64),
        torch.as_tensor(archive["item_factors"], dtype=torch.float64),
    )


def _item_hessians_and_gradient(
    users: torch.Tensor,
    items: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    *,
    omega: float,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_users, dimension = users.shape
    num_items = items.shape[0]
    interaction_users = users.index_select(0, train_users)
    observed_moments = torch.zeros(
        (num_items, dimension, dimension), dtype=users.dtype
    )
    observed_moments.index_add_(
        0,
        train_items,
        interaction_users.unsqueeze(2) * interaction_users.unsqueeze(1),
    )
    observed_rhs = torch.zeros_like(items)
    observed_rhs.index_add_(0, train_items, omega * interaction_users)
    unnormalized_systems = (
        (users.T @ users).unsqueeze(0) + (omega - 1.0) * observed_moments
    )
    identity = torch.eye(dimension, dtype=users.dtype)
    hessians = (
        2.0 * unnormalized_systems / num_users
        + 2.0 * regularization * identity.unsqueeze(0)
    )
    gradient = (
        2.0
        * (
            torch.einsum("mde,me->md", unnormalized_systems, items)
            - observed_rhs
        )
        / num_users
        + 2.0 * regularization * items
    )
    return hessians, gradient


def validate_dataset(
    *,
    dataset: str,
    data_root: Path,
    result_root: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    spec = get_pnsmf_fixed_dataset_spec(dataset)
    data_dir = data_root / DATA_DIRECTORIES[dataset]
    train_name, validation_name, test_name = spec.filenames(1)
    data = load_pnsmf_fixed_split(
        data_dir / train_name,
        data_dir / validation_name,
        data_dir / test_name,
        num_users=spec.num_users,
        num_items=spec.num_items,
    )
    factor_path = result_root / "factors" / (
        f"{dataset}_copy1_eps2p193_seed20260821.npz"
    )
    run_path = result_root / "runs" / (
        f"{dataset}_copy1_eps2p193_seed20260821.json"
    )
    users, items = _load_factors(factor_path)
    train_users = torch.as_tensor(data.train_users, dtype=torch.long)
    train_items = torch.as_tensor(data.train_items, dtype=torch.long)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    last_diagnostic = run["round_diagnostics"][-1]
    update_noise_variance = float(last_diagnostic["update_noise_variance"])

    hessians, gradient = _item_hessians_and_gradient(
        users,
        items,
        train_users,
        train_items,
        omega=spec.paper_omega,
        regularization=spec.paper_regularization,
    )
    trace_direct = float(torch.diagonal(hessians, dim1=1, dim2=2).sum())
    trace_helper = float(
        item_subproblem_hessian_trace(
            users,
            train_users,
            num_items=data.num_items,
            omega=spec.paper_omega,
            regularization=spec.paper_regularization,
        )
    )
    expected = expected_quadratic_loss_inflation(
        hessian_trace=trace_direct,
        update_noise_variance=update_noise_variance,
    )

    if draws <= 0 or draws % 2:
        raise ValueError("draws must be a positive even integer for antithetic sampling.")
    eigenvalues = torch.linalg.eigvalsh(hessians).flatten()
    generator = torch.Generator().manual_seed(seed)
    half_draws = draws // 2
    quadratic_values = []
    batch_size = 25
    for start in range(0, half_draws, batch_size):
        count = min(batch_size, half_draws - start)
        standard_normal = torch.randn(
            (count, eigenvalues.numel()),
            dtype=torch.float64,
            generator=generator,
        )
        values = (
            0.5
            * update_noise_variance
            * (standard_normal.square() * eigenvalues.unsqueeze(0)).sum(dim=1)
        )
        quadratic_values.extend(values.tolist())
    empirical = float(np.mean(quadratic_values))
    standard_error = float(np.std(quadratic_values, ddof=1) / math.sqrt(half_draws))

    baseline = float(
        weighted_nsmf_loss_indexed(
            users,
            items,
            train_users,
            train_items,
            omega=spec.paper_omega,
            regularization=spec.paper_regularization,
        )
        / data.num_users
    )
    identity_errors = []
    identity_generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(4):
        noise = torch.randn(
            items.shape, dtype=torch.float64, generator=identity_generator
        ) * update_noise_variance**0.5
        perturbed = float(
            weighted_nsmf_loss_indexed(
                users,
                items + noise,
                train_users,
                train_items,
                omega=spec.paper_omega,
                regularization=spec.paper_regularization,
            )
            / data.num_users
        )
        quadratic_identity = float(
            (gradient * noise).sum()
            + 0.5 * torch.einsum("md,mde,me->", noise, hessians, noise)
        )
        identity_errors.append(abs((perturbed - baseline) - quadratic_identity))

    return {
        "dataset": dataset,
        "copy": 1,
        "factor_state": str(factor_path),
        "run_state": str(run_path),
        "coordinates": int(items.numel()),
        "update_noise_variance": update_noise_variance,
        "hessian_trace_direct": trace_direct,
        "hessian_trace_helper": trace_helper,
        "hessian_trace_absolute_difference": abs(trace_direct - trace_helper),
        "analytic_expected_conditional_loss_inflation": expected,
        "runner_recorded_expected_conditional_loss_inflation": float(
            last_diagnostic["expected_conditional_loss_inflation"]
        ),
        "monte_carlo_draws": draws,
        "effective_antithetic_pairs": half_draws,
        "empirical_mean_conditional_loss_inflation": empirical,
        "monte_carlo_standard_error": standard_error,
        "monte_carlo_95pct_interval": [
            empirical - 1.96 * standard_error,
            empirical + 1.96 * standard_error,
        ],
        "relative_error_percent": 100.0 * (empirical - expected) / expected,
        "maximum_absolute_quadratic_identity_error": max(identity_errors),
    }


def main() -> None:
    args = parse_args()
    rows = [
        validate_dataset(
            dataset=dataset,
            data_root=args.data_root,
            result_root=args.registered_result_dir,
            draws=args.draws,
            seed=args.seed + index,
        )
        for index, dataset in enumerate(DATA_DIRECTORIES)
    ]
    result = {
        "status": "real_state_quadratic_inflation_validation",
        "protocol_id": "REVIEW-REVISION-SENSITIVITY-UNCERTAINTY-20260831-V2",
        "rows": rows,
        "interpretation": (
            "Conditional fixed-user item-subproblem validation. It does not "
            "predict multi-round ranking utility."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

