"""Run a traceable uniform user-level DP-P-NSMF(BGD) diagnostic."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.data import (
    PNSMF_FIXED_DATASET_SPECS,
    build_pnsmf_public_interaction_order,
    get_pnsmf_fixed_dataset_spec,
    load_pnsmf_fixed_split,
    pnsmf_training_strata,
    rotating_pnsmf_user_interactions,
)
from src.evaluation import evaluate_pnsmf_factors, evaluate_pnsmf_factors_by_strata
from src.models import item_subproblem_hessian_trace, weighted_nsmf_loss_indexed
from src.privacy import (
    RdpPrivacyConfig,
    compute_epsilon,
    expected_quadratic_loss_inflation,
    pnsmf_catalog_sensitivity_bound,
    public_model_scaled_clip_norm,
)
from src.training import (
    run_uniform_dp_pnsmf_bgd_round,
    sample_poisson_clients,
    server_ema_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=tuple(PNSMF_FIXED_DATASET_SPECS),
        default="ml1m",
    )
    parser.add_argument("--copy", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--factor-output",
        type=Path,
        help="Optional compressed NPZ path for final factors used by offline diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--item-learning-rate",
        type=float,
        help="Server item-step size; defaults to --learning-rate.",
    )
    parser.add_argument("--learning-rate-decay", type=float, default=0.999)
    parser.add_argument(
        "--server-momentum",
        type=float,
        default=0.0,
        help="EMA coefficient applied to the already released item delta.",
    )
    parser.add_argument("--omega", type=float)
    parser.add_argument("--regularization", type=float)
    parser.add_argument("--sampling-probability", type=float, required=True)
    parser.add_argument("--clip-norm", type=float)
    parser.add_argument(
        "--clip-mode",
        choices=("fixed", "public_item_norm", "catalog_energy"),
        default="fixed",
    )
    parser.add_argument("--clip-ratio", type=float)
    parser.add_argument("--clip-max-growth", type=float, default=1.05)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=float("inf"))
    parser.add_argument("--interaction-cap", type=int)
    parser.add_argument("--user-factor-radius", type=float)
    parser.add_argument("--public-cap-seed", type=int, default=20260802)
    parser.add_argument("--noise-multiplier", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--evaluation-every", type=int, default=1)
    parser.add_argument("--evaluation-target", choices=("validation", "test"), default="validation")
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--report-strata", action="store_true")
    parser.add_argument(
        "--popularity-strata",
        choices=("item_count_quintile", "training_mass_tertile"),
        default="item_count_quintile",
    )
    parser.add_argument("--clip-source", type=str, required=True)
    parser.add_argument(
        "--study-phase",
        choices=("diagnostic", "confirmatory", "registered_followup"),
        default="diagnostic",
        help="Evidence label. Confirmatory runs require a frozen, public clip source.",
    )
    parser.add_argument(
        "--contribution-space",
        choices=("joint_sufficient_statistics", "direct_item_gradient"),
        default="joint_sufficient_statistics",
    )
    parser.add_argument(
        "--contribution-rule",
        choices=("clip", "normalize"),
        default="clip",
        help="Use ordinary clipping or the published DP-NormFedAvg-style fixed-norm baseline.",
    )
    return parser.parse_args()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    probabilities = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0], dtype=values.dtype, device=values.device
    )
    quantiles = torch.quantile(values, probabilities).cpu().tolist()
    return {label: float(value) for label, value in zip(("min", "q25", "q50", "q75", "max"), quantiles)}


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.evaluation_every <= 0:
        raise ValueError("rounds and evaluation_every must be positive.")
    if not 0.0 <= args.server_momentum < 1.0:
        raise ValueError("server_momentum must be in [0, 1).")
    if args.clip_mode == "fixed" and (args.clip_norm is None or args.clip_norm <= 0.0):
        raise ValueError("fixed clip mode requires a positive --clip-norm.")
    if args.study_phase == "confirmatory" and not args.clip_source.startswith("frozen_"):
        raise ValueError("confirmatory runs require --clip-source beginning with 'frozen_'.")
    if args.clip_mode == "public_item_norm" and (
        args.clip_ratio is None or args.clip_ratio <= 0.0
    ):
        raise ValueError("public_item_norm mode requires a positive --clip-ratio.")
    if args.clip_mode == "catalog_energy":
        if args.interaction_cap is None or args.interaction_cap <= 0:
            raise ValueError("catalog_energy mode requires a positive interaction cap.")
        if args.user_factor_radius is None or args.user_factor_radius <= 0.0:
            raise ValueError("catalog_energy mode requires a positive user factor radius.")
        if args.contribution_space != "joint_sufficient_statistics":
            raise ValueError("catalog_energy mode requires joint sufficient statistics.")
        if args.contribution_rule != "clip":
            raise ValueError("catalog_energy mode requires ordinary clipping.")
    np_rng = np.random.default_rng(args.seed)
    noise_generator = torch.Generator().manual_seed(args.seed + 1)
    torch.manual_seed(args.seed)
    if args.torch_threads <= 0:
        raise ValueError("torch_threads must be positive.")
    torch.set_num_threads(args.torch_threads)
    data_dir = args.data_dir
    dataset_spec = get_pnsmf_fixed_dataset_spec(args.dataset)
    if args.learning_rate is None:
        args.learning_rate = dataset_spec.paper_bgd_learning_rate
    if args.omega is None:
        args.omega = dataset_spec.paper_omega
    if args.regularization is None:
        args.regularization = dataset_spec.paper_regularization
    train_name, validation_name, test_name = dataset_spec.filenames(args.copy)
    data = load_pnsmf_fixed_split(
        data_dir / train_name,
        data_dir / validation_name,
        data_dir / test_name,
        num_users=dataset_spec.num_users,
        num_items=dataset_spec.num_items,
    )
    train_users = torch.as_tensor(data.train_users, dtype=torch.long)
    train_items = torch.as_tensor(data.train_items, dtype=torch.long)
    objective_train_users = train_users
    objective_train_items = train_items
    public_order_by_user = (
        build_pnsmf_public_interaction_order(
            data.train_users,
            num_users=data.num_users,
            public_seed=args.public_cap_seed,
        )
        if args.clip_mode == "catalog_energy"
        else None
    )
    if args.report_strata:
        (
            activity_groups,
            item_popularity_groups,
            user_train_counts,
            item_train_counts,
        ) = pnsmf_training_strata(
            data, popularity_scheme=args.popularity_strata
        )
    else:
        activity_groups = item_popularity_groups = None
        user_train_counts = item_train_counts = None
    users = (torch.rand((data.num_users, args.embedding_dim), dtype=torch.float64) - 0.5) * 0.01
    items = (torch.rand((data.num_items, args.embedding_dim), dtype=torch.float64) - 0.5) * 0.01
    targets = data.validation_by_user if args.evaluation_target == "validation" else data.test_by_user
    history = []

    def evaluate(round_index: int) -> None:
        normalized_objective = float(
            weighted_nsmf_loss_indexed(
                users,
                items,
                objective_train_users,
                objective_train_items,
                omega=args.omega,
                regularization=args.regularization,
            )
            / data.num_users
        )
        if args.report_strata:
            grouped = evaluate_pnsmf_factors_by_strata(
                users,
                items,
                targets,
                data.train_by_user,
                activity_groups,
                item_popularity_groups,
                cutoff=args.cutoff,
            )
            history.append(
                {
                    "round": round_index,
                    "normalized_training_objective": normalized_objective,
                    "metrics": grouped["overall"],
                    "strata": {
                        key: value
                        for key, value in grouped.items()
                        if key != "overall"
                    },
                }
            )
        else:
            history.append({
                "round": round_index,
                "normalized_training_objective": normalized_objective,
                "metrics": evaluate_pnsmf_factors(
                    users, items, targets, data.train_by_user, cutoff=args.cutoff
                ),
            })

    started = time.perf_counter()
    evaluate(0)
    learning_rate = args.learning_rate
    item_learning_rate = (
        args.learning_rate
        if args.item_learning_rate is None
        else args.item_learning_rate
    )
    round_diagnostics = []
    previous_clip_norm = None
    item_velocity = torch.zeros_like(items)
    with torch.no_grad():
        for round_index in range(1, args.rounds + 1):
            catalog_bound = None
            if public_order_by_user is not None:
                round_users, round_items = rotating_pnsmf_user_interactions(
                    data.train_users,
                    data.train_items,
                    public_order_by_user,
                    interaction_cap=args.interaction_cap,
                    round_index=round_index - 1,
                )
                train_users = torch.as_tensor(round_users, dtype=torch.long)
                train_items = torch.as_tensor(round_items, dtype=torch.long)
                catalog_bound = pnsmf_catalog_sensitivity_bound(
                    item_factor_norms=items.norm(dim=1).tolist(),
                    interaction_cap=args.interaction_cap,
                    user_factor_radius=args.user_factor_radius,
                    positive_weight=args.omega,
                    missing_weight=1.0,
                    catalog_weight=1.0,
                )
                applied_clip_norm = catalog_bound.catalog_dependent_squared_bound**0.5
            elif args.clip_mode == "fixed":
                applied_clip_norm = float(args.clip_norm)
            else:
                applied_clip_norm = public_model_scaled_clip_norm(
                    items,
                    scale_ratio=float(args.clip_ratio),
                    previous_clip_norm=previous_clip_norm,
                    max_growth_factor=args.clip_max_growth,
                    min_clip_norm=args.clip_min,
                    max_clip_norm=args.clip_max,
                )
            previous_clip_norm = applied_clip_norm
            selected_numpy = sample_poisson_clients(
                data.num_users, args.sampling_probability, np_rng
            )
            selected = torch.as_tensor(selected_numpy, dtype=torch.long)
            previous_items = items
            users, proposed_items, diagnostics = run_uniform_dp_pnsmf_bgd_round(
                users,
                previous_items,
                train_users,
                train_items,
                selected,
                learning_rate=learning_rate,
                item_learning_rate=item_learning_rate,
                sampling_probability=args.sampling_probability,
                clip_norm=applied_clip_norm,
                noise_multiplier=args.noise_multiplier,
                omega=args.omega,
                regularization=args.regularization,
                contribution_space=args.contribution_space,
                contribution_rule=args.contribution_rule,
                user_factor_radius=(
                    args.user_factor_radius
                    if args.clip_mode == "catalog_energy"
                    else None
                ),
                generator=noise_generator,
            )
            items, item_velocity = server_ema_delta(
                previous_items,
                proposed_items,
                item_velocity,
                momentum=args.server_momentum,
            )
            if not torch.isfinite(users).all() or not torch.isfinite(items).all():
                raise FloatingPointError(f"Non-finite factors after round {round_index}.")
            contribution = diagnostics.contribution
            contribution_by_activity = {}
            if args.report_strata:
                selected_groups = activity_groups[selected_numpy]
                for group in sorted(set(selected_groups)):
                    mask_numpy = selected_groups == group
                    mask = torch.as_tensor(mask_numpy, dtype=torch.bool)
                    group_norms = contribution.pre_clip_norm[mask]
                    group_clipped = contribution.clipped[mask]
                    group_factors = contribution.clipping_factor[mask]
                    contribution_by_activity[str(group)] = {
                        "selected_clients": int(mask_numpy.sum()),
                        "pre_clip_norm_mean": float(group_norms.mean()),
                        "pre_clip_norm_median": float(group_norms.median()),
                        "clip_rate": float(group_clipped.double().mean()),
                        "rescale_rate": float(
                            (~torch.isclose(group_factors, torch.ones_like(group_factors))).double().mean()
                        ),
                        "mean_clipping_factor": float(group_factors.mean()),
                    }
            hessian_trace = float(
                item_subproblem_hessian_trace(
                    users,
                    train_users,
                    num_items=items.shape[0],
                    omega=args.omega,
                    regularization=args.regularization,
                )
            )
            update_noise_variance = (
                2.0
                * item_learning_rate
                * args.noise_multiplier
                * applied_clip_norm
                / (args.sampling_probability * users.shape[0])
            ) ** 2
            round_diagnostics.append({
                "round": round_index,
                "user_learning_rate": learning_rate,
                "item_learning_rate": item_learning_rate,
                "selected_clients": diagnostics.selected_clients,
                "applied_clip_norm": applied_clip_norm,
                **(
                    {
                        "catalog_max_norm_bound": catalog_bound.max_norm_squared_bound**0.5,
                        "catalog_noise_standard_deviation_ratio": catalog_bound.bound_ratio**0.5,
                    }
                    if catalog_bound is not None
                    else {}
                ),
                "observed_rows": contribution.observed_rows,
                "clip_rate": float(contribution.clipped.double().mean()) if diagnostics.selected_clients else 0.0,
                "rescale_rate": (
                    float(
                        (~torch.isclose(
                            contribution.clipping_factor,
                            torch.ones_like(contribution.clipping_factor),
                        )).double().mean()
                    )
                    if diagnostics.selected_clients
                    else 0.0
                ),
                "pre_clip_norm": _quantiles(contribution.pre_clip_norm),
                "aggregate_signal_norm": diagnostics.aggregate_signal_norm,
                "gaussian_noise_norm": diagnostics.gaussian_noise_norm,
                "item_subproblem_hessian_trace": hessian_trace,
                "update_noise_variance": update_noise_variance,
                "expected_conditional_loss_inflation": expected_quadratic_loss_inflation(
                    hessian_trace=hessian_trace,
                    update_noise_variance=update_noise_variance,
                ),
                "signal_to_noise_ratio": (
                    diagnostics.signal_to_noise_ratio
                    if np.isfinite(diagnostics.signal_to_noise_ratio)
                    else None
                ),
                "released_item_signal_norm": diagnostics.released_item_signal_norm,
                "released_item_noise_norm": diagnostics.released_item_noise_norm,
                "released_item_signal_to_noise_ratio": (
                    diagnostics.released_item_signal_to_noise_ratio
                    if np.isfinite(diagnostics.released_item_signal_to_noise_ratio)
                    else None
                ),
                "user_factor_norm": float(torch.linalg.vector_norm(users)),
                "item_factor_norm": float(torch.linalg.vector_norm(items)),
                "server_velocity_norm": float(torch.linalg.vector_norm(item_velocity)),
                **(
                    {"contribution_by_activity": contribution_by_activity}
                    if args.report_strata
                    else {}
                ),
            })
            learning_rate *= args.learning_rate_decay
            item_learning_rate *= args.learning_rate_decay
            if round_index % args.evaluation_every == 0 or round_index == args.rounds:
                evaluate(round_index)

    epsilon = (
        compute_epsilon(
            RdpPrivacyConfig(
                sampling_probability=args.sampling_probability,
                noise_multiplier=args.noise_multiplier,
                steps=args.rounds,
                delta=args.delta,
            )
        )
        if args.noise_multiplier > 0.0
        else None
    )
    serialized_args = {
        key: (
            str(value)
            if isinstance(value, Path)
            else None
            if isinstance(value, float) and not np.isfinite(value)
            else value
        )
        for key, value in vars(args).items()
    }
    result = {
        "status": (
            "confirmatory_user_level_dp_result"
            if args.study_phase == "confirmatory" and args.noise_multiplier > 0.0
            else "confirmatory_control_not_private"
            if args.study_phase == "confirmatory"
            else "registered_followup_user_level_dp_result"
            if args.study_phase == "registered_followup" and args.noise_multiplier > 0.0
            else "registered_followup_control_not_private"
            if args.study_phase == "registered_followup"
            else "dp_smoke_test_not_a_paper_result"
            if args.noise_multiplier > 0.0
            else "paired_clipping_without_noise_not_private"
        ),
        "privacy_scope": (
            "simulated_secure_aggregation_supported_central_user_level_DP"
            if args.noise_multiplier > 0.0
            else "not_private_no_gaussian_noise"
        ),
        "warning": (
            "catalog_energy uses only the prior public/DP item transcript; private diagnostics and evaluation are not mechanism outputs"
            if args.clip_mode == "catalog_energy"
            else "clip_source must be independently public or privately selected before a formal DP claim"
        ),
        "config": {
            **serialized_args,
            "data_dir": str(args.data_dir),
            "output": str(args.output),
            "clip_source": args.clip_source,
        },
        "privacy_accounting": {
            "mechanism": "Poisson-sampled Gaussian",
            "epsilon": epsilon,
            "delta": args.delta,
        },
        "protocol": {
            "dataset": dataset_spec.display_name,
            "declared_users": dataset_spec.num_users,
            "declared_items": dataset_spec.num_items,
            "contribution_space": args.contribution_space,
            "contribution_rule": args.contribution_rule,
            "normalization": "expected_sample_count_q_times_n",
            "ranking_catalog": f"all_declared_{dataset_spec.num_items}_items",
            "seen_mask": "train_file_only",
            "activity_definition": (
                "training_interaction_count_qcut_quintiles_Q1_to_Q5"
                if args.report_strata
                else None
            ),
            "item_popularity_definition": (
                (
                    "training_count_average_rank_quintiles_P1_tail_to_P5_head"
                    if args.popularity_strata == "item_count_quintile"
                    else "cold_plus_warm_training_interaction_mass_tertiles"
                )
                if args.report_strata
                else None
            ),
            **(
                {
                    "activity_group_summary": {
                        group: {
                            "users": int(np.sum(activity_groups == group)),
                            "min_train_interactions": int(
                                user_train_counts[activity_groups == group].min()
                            ),
                            "max_train_interactions": int(
                                user_train_counts[activity_groups == group].max()
                            ),
                        }
                        for group in sorted(set(activity_groups))
                    },
                    "item_popularity_group_summary": {
                        group: {
                            "items": int(np.sum(item_popularity_groups == group)),
                            "min_train_interactions": int(
                                item_train_counts[item_popularity_groups == group].min()
                            ),
                            "max_train_interactions": int(
                                item_train_counts[item_popularity_groups == group].max()
                            ),
                        }
                        for group in sorted(set(item_popularity_groups))
                    },
                }
                if args.report_strata
                else {}
            ),
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
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
        factor_metadata = {
            "dataset": args.dataset,
            "copy": args.copy,
            "seed": args.seed,
            "rounds": args.rounds,
            "sampling_probability": args.sampling_probability,
            "clip_norm": args.clip_norm,
            "noise_multiplier": args.noise_multiplier,
            "delta": args.delta,
            "evaluation_target": args.evaluation_target,
            "cutoff": args.cutoff,
            "study_phase": args.study_phase,
            "result_json": str(args.output),
        }
        np.savez_compressed(
            args.factor_output,
            user_factors=users.cpu().numpy(),
            item_factors=items.cpu().numpy(),
            metadata=np.asarray(json.dumps(factor_metadata, ensure_ascii=False)),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
