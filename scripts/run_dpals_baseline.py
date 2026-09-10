"""Run the controlled DPALS-style stress test on ML-1M."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.data import cap_pnsmf_user_interactions, load_pnsmf_fixed_split
from src.evaluation import evaluate_pnsmf_factors
from src.models import dpals_implicit_loss_indexed, random_orthonormal_item_factors
from src.privacy import (
    GaussianCompositionPrivacyConfig,
    calibrate_composed_gaussian_noise_multiplier,
    compute_composed_gaussian_epsilon,
)
from src.training import dpals_private_item_update, dpals_user_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--copy", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--public-cap-seed", type=int, default=20260900)
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--interaction-cap", type=int, required=True)
    parser.add_argument("--user-clip-radius", type=float, required=True)
    parser.add_argument("--entry-clip", type=float, default=1.0)
    parser.add_argument("--implicit-penalty", type=float, required=True)
    parser.add_argument("--regularization", type=float, required=True)
    parser.add_argument("--noise-multiplier", type=float)
    parser.add_argument("--target-epsilon", type=float)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--eigenvalue-rcond", type=float, default=1e-8)
    parser.add_argument("--evaluation-every", type=int, default=1)
    parser.add_argument(
        "--evaluation-target", choices=("validation", "test"), required=True
    )
    parser.add_argument("--cutoff", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--study-phase",
        choices=("validation_tuning", "review_baseline"),
        required=True,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_sha256(users: np.ndarray, items: np.ndarray) -> str:
    digest = hashlib.sha256()
    for user, item in zip(users, items, strict=True):
        digest.update(f"{int(user) + 1} {int(item) + 1}\n".encode("ascii"))
    return digest.hexdigest()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0], dtype=values.dtype, device=values.device
    )
    result = torch.quantile(values, probabilities).cpu().tolist()
    return {
        label: float(value)
        for label, value in zip(("min", "q25", "q50", "q75", "max"), result)
    }


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.interaction_cap <= 0:
        raise ValueError("rounds and interaction_cap must be positive.")
    if args.embedding_dim <= 0 or args.user_clip_radius <= 0.0:
        raise ValueError("embedding_dim and user_clip_radius must be positive.")
    if args.implicit_penalty < 0.0 or args.regularization <= 0.0:
        raise ValueError("implicit_penalty must be non-negative and regularization positive.")
    if args.evaluation_every <= 0 or args.torch_threads <= 0:
        raise ValueError("evaluation_every and torch_threads must be positive.")
    if args.noise_multiplier is not None and args.target_epsilon is not None:
        raise ValueError("Specify either noise_multiplier or target_epsilon, not both.")
    if args.noise_multiplier is None and args.target_epsilon is None:
        args.noise_multiplier = 0.0
    if args.study_phase == "validation_tuning":
        if args.evaluation_target != "validation" or args.target_epsilon is not None:
            raise ValueError("Validation tuning must use validation and no target epsilon.")
        if args.noise_multiplier != 0.0:
            raise ValueError("Validation selection is frozen to the matched no-noise arm.")

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    initialization_generator = torch.Generator().manual_seed(args.seed)
    noise_generator = torch.Generator().manual_seed(args.seed + 1)
    paths = {
        split: args.data_dir / f"ML1M-copy{args.copy}-{split}"
        for split in ("train", "valid", "test")
    }
    data = load_pnsmf_fixed_split(
        paths["train"],
        paths["valid"],
        paths["test"],
        num_users=6040,
        num_items=3952,
    )
    capped_users_np, capped_items_np = cap_pnsmf_user_interactions(
        data.train_users,
        data.train_items,
        num_users=data.num_users,
        interaction_cap=args.interaction_cap,
        public_seed=args.public_cap_seed,
    )
    full_users = torch.as_tensor(data.train_users, dtype=torch.long)
    full_items = torch.as_tensor(data.train_items, dtype=torch.long)
    capped_users = torch.as_tensor(capped_users_np, dtype=torch.long)
    capped_items = torch.as_tensor(capped_items_np, dtype=torch.long)
    capped_counts = np.bincount(capped_users_np, minlength=data.num_users)
    if int(capped_counts.max(initial=0)) > args.interaction_cap:
        raise RuntimeError("Interaction cap invariant failed.")

    compositions = (args.interaction_cap + 1) * args.rounds
    if args.target_epsilon is not None:
        args.noise_multiplier = calibrate_composed_gaussian_noise_multiplier(
            target_epsilon=args.target_epsilon,
            compositions=compositions,
            delta=args.delta,
        )
    assert args.noise_multiplier is not None
    epsilon = (
        compute_composed_gaussian_epsilon(
            GaussianCompositionPrivacyConfig(
                noise_multiplier=args.noise_multiplier,
                compositions=compositions,
                delta=args.delta,
            )
        )
        if args.noise_multiplier > 0.0
        else None
    )

    items = random_orthonormal_item_factors(
        data.num_items,
        args.embedding_dim,
        generator=initialization_generator,
    )
    targets = (
        data.validation_by_user
        if args.evaluation_target == "validation"
        else data.test_by_user
    )
    history: list[dict[str, object]] = []
    round_diagnostics: list[dict[str, object]] = []
    evaluation_users = torch.empty(
        (data.num_users, args.embedding_dim), dtype=torch.float64
    )

    def evaluate(round_index: int) -> None:
        nonlocal evaluation_users
        evaluation_started = time.perf_counter()
        evaluation_users, local_diagnostics = dpals_user_update(
            items,
            full_users,
            full_items,
            num_users=data.num_users,
            implicit_penalty=args.implicit_penalty,
            regularization=args.regularization,
            clip_radius=None,
        )
        objective = dpals_implicit_loss_indexed(
            evaluation_users,
            items,
            full_users,
            full_items,
            implicit_penalty=args.implicit_penalty,
            regularization=args.regularization,
        )
        metrics = evaluate_pnsmf_factors(
            evaluation_users,
            items,
            targets,
            data.train_by_user,
            cutoff=args.cutoff,
        )
        history.append(
            {
                "round": round_index,
                "normalized_training_objective": float(objective / data.num_users),
                "metrics": metrics,
                "evaluation_user_norm": _quantiles(local_diagnostics.pre_clip_norm),
                "item_factor_norm": float(torch.linalg.vector_norm(items)),
                "item_whitening_error_frobenius": float(
                    torch.linalg.matrix_norm(
                        items.T @ items
                        - torch.eye(args.embedding_dim, dtype=items.dtype)
                    )
                ),
                "evaluation_seconds": time.perf_counter() - evaluation_started,
            }
        )

    started = time.perf_counter()
    evaluate(0)
    with torch.no_grad():
        for round_index in range(1, args.rounds + 1):
            round_started = time.perf_counter()
            training_users, user_diagnostics = dpals_user_update(
                items,
                full_users,
                full_items,
                num_users=data.num_users,
                implicit_penalty=args.implicit_penalty,
                regularization=args.regularization,
                clip_radius=args.user_clip_radius,
            )
            items, item_diagnostics = dpals_private_item_update(
                training_users,
                capped_users,
                capped_items,
                num_items=data.num_items,
                implicit_penalty=args.implicit_penalty,
                regularization=args.regularization,
                user_clip_radius=args.user_clip_radius,
                entry_clip=args.entry_clip,
                noise_multiplier=args.noise_multiplier,
                eigenvalue_rcond=args.eigenvalue_rcond,
                generator=noise_generator,
            )
            if not torch.isfinite(items).all():
                raise FloatingPointError(f"Non-finite item factors at round {round_index}.")
            clip_factors = user_diagnostics.clipping_factor
            round_diagnostics.append(
                {
                    "round": round_index,
                    "training_user_pre_clip_norm": _quantiles(
                        user_diagnostics.pre_clip_norm
                    ),
                    "user_clip_rate": float((clip_factors < 1.0).double().mean()),
                    "mean_user_clipping_factor": float(clip_factors.mean()),
                    "item_gram_noise_norm": item_diagnostics.item_gram_noise_norm,
                    "item_vector_noise_norm": item_diagnostics.item_vector_noise_norm,
                    "shared_gram_noise_norm": item_diagnostics.shared_gram_noise_norm,
                    "systems_with_nonpositive_eigenvalue": item_diagnostics.systems_with_nonpositive_eigenvalue,
                    "retained_rank_min": item_diagnostics.retained_rank_min,
                    "retained_rank_mean": item_diagnostics.retained_rank_mean,
                    "retained_rank_max": item_diagnostics.retained_rank_max,
                    "round_seconds": time.perf_counter() - round_started,
                }
            )
            if round_index % args.evaluation_every == 0 or round_index == args.rounds:
                evaluate(round_index)

    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    is_private = args.noise_multiplier > 0.0
    result = {
        "status": (
            "reviewer_requested_dpals_style_user_level_joint_dp_baseline"
            if is_private
            else "reviewer_requested_matched_dpals_no_noise_control"
            if args.study_phase == "review_baseline"
            else "validation_only_dpals_no_noise_parameter_selection"
        ),
        "protocol_id": "REVIEW-REVISION-DPALS-ML1M-20260831-V2",
        "method_scope": {
            "label": "controlled_DPALS_style_implicit_feedback_baseline",
            "retained": [
                "local_user_ALS",
                "user_factor_l2_clipping",
                "at_most_k_item_contributions_per_user",
                "Gaussian_item_Gram_and_vector_statistics",
                "noisy_shared_implicit_Gram_term",
                "PSD_projection_and_pseudoinverse",
                "item_whitening",
            ],
            "omitted": [
                "private_popularity_filtering",
                "adaptive_item_sampling",
                "private_global_centering",
            ],
            "comparison_claim": "mechanism_calibration_not_performance_superiority",
        },
        "config": serialized_args,
        "privacy_accounting": {
            "scope": (
                "user_level_joint_DP_for_public_item_factors"
                if is_private
                else "not_private_matched_no_noise_control"
            ),
            "mechanism": "non_subsampled_Gaussian_RDP_composition",
            "item_participations_per_user_per_round": args.interaction_cap,
            "shared_implicit_Gram_releases_per_round": 1,
            "compositions": compositions,
            "noise_multiplier": args.noise_multiplier,
            "epsilon": epsilon,
            "delta": args.delta,
            "individual_user_factor": "privileged_output_for_that_user",
            "offline_metric_warning": "test_metrics_are_not_DP_release_outputs",
        },
        "data": {
            "copy": args.copy,
            "train": data.train_count,
            "validation": data.validation_count,
            "test": data.test_count,
            "capped_train_interactions": int(len(capped_users_np)),
            "maximum_capped_interactions_per_user": int(
                capped_counts.max(initial=0)
            ),
            "users_at_cap": int(np.sum(capped_counts == args.interaction_cap)),
            "capped_pair_sha256": _pair_sha256(capped_users_np, capped_items_np),
            "file_sha256": {
                split: _sha256(path) for split, path in paths.items()
            },
        },
        "protocol": {
            "dataset": "MovieLens-1M fixed P-NSMF split",
            "declared_users": data.num_users,
            "declared_items": data.num_items,
            "binary_event_value": 1.0,
            "ranking_catalog": "all_declared_3952_items",
            "seen_mask": "train_file_only",
            "validation_rejoined_for_test": False,
            "user_item_cap_selection": "uniform_without_replacement_once_per_user_under_fixed_seed",
            "initialization": "random_orthonormal_independent_of_private_data",
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "round_diagnostics": round_diagnostics,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.factor_output is not None:
        args.factor_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.factor_output,
            evaluation_user_factors=evaluation_users.cpu().numpy(),
            public_item_factors=items.cpu().numpy(),
            metadata=np.asarray(json.dumps(result["privacy_accounting"])),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
