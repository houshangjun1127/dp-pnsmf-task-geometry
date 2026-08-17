"""Summarize the post-result geometry and budget analysis without treating users as replicates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import get_pnsmf_fixed_dataset_spec, load_pnsmf_fixed_split
from src.evaluation import paired_task_geometry_rows


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "task_geometry_budget_analysis_v1"
RUNS = SOURCE / "runs"
FACTORS = SOURCE / "factors"
OUTPUT = SOURCE / "summary"

DATA_DIRS = {
    "ml1m": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "ML1M-TXT-FORMAT",
    "amazon_kindle": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Amazon_Kindle_Store-TXT-FORMAT",
    "netflix5k5k": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Netflix5K5K-TXT-FORAMT",
}
ARMS = ("matched", "eps1", "eps2p193", "eps4", "eps8")
PRIVATE_ARMS = ARMS[1:]
SEEDS = {1: 20260821, 2: 20260822, 3: 20260823}


def finite_median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def load_factors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        users = torch.from_numpy(archive["user_factors"])
        items = torch.from_numpy(archive["item_factors"])
    return users, items


def run_stem(dataset: str, copy: int, arm: str) -> str:
    return f"{dataset}_copy{copy}_{arm}_seed{SEEDS[copy]}"


def summarize_user_geometry(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "users": len(frame),
        "margin_at_20_median": float(frame["margin_at_20"].median()),
        "local_boundary_perturbation_median": float(frame["local_boundary_perturbation"].median()),
        "margin_to_local_perturbation_median": float(frame["margin_to_local_perturbation"].median()),
        "full_catalog_linf_perturbation_median": float(frame["full_catalog_linf_perturbation"].median()),
        "conservative_stability_ratio_median": float(frame["conservative_stability_ratio"].median()),
        "conservative_certificate_fraction": float((frame["conservative_stability_ratio"] > 1.0).mean()),
        "top20_replacement_fraction_mean": float(frame["top20_replacement_fraction"].mean()),
        "top20_replacement_fraction_median": float(frame["top20_replacement_fraction"].median()),
        "top20_jaccard_mean": float(frame["top20_jaccard"].mean()),
        "best_target_absolute_rank_change_median": float(frame["best_target_absolute_rank_change"].median()),
        "best_target_hit_loss_fraction": float(
            ((frame["best_target_hit_control"] == 1) & (frame["best_target_hit_private"] == 0)).mean()
        ),
        "best_target_hit_gain_fraction": float(
            ((frame["best_target_hit_control"] == 0) & (frame["best_target_hit_private"] == 1)).mean()
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    user_output = SOURCE / "geometry_user_level"
    user_output.mkdir(parents=True, exist_ok=True)
    expected = len(DATA_DIRS) * len(SEEDS) * len(ARMS)
    run_files = sorted(RUNS.glob("*.json"))
    factor_files = sorted(FACTORS.glob("*.npz"))
    if len(run_files) != expected or len(factor_files) != expected:
        raise RuntimeError(
            f"Expected {expected} result/factor pairs; found {len(run_files)} JSON and {len(factor_files)} NPZ."
        )

    run_rows: list[dict[str, object]] = []
    run_lookup: dict[tuple[str, int, str], dict[str, object]] = {}
    for dataset in DATA_DIRS:
        for copy in SEEDS:
            for arm in ARMS:
                result = json.loads((RUNS / f"{run_stem(dataset, copy, arm)}.json").read_text(encoding="utf-8"))
                final = result["history"][-1]["metrics"]
                diagnostics = result["round_diagnostics"]
                row = {
                    "dataset": dataset,
                    "copy": copy,
                    "seed": SEEDS[copy],
                    "arm": arm,
                    "noise_multiplier": float(result["config"]["noise_multiplier"]),
                    "epsilon": result["privacy_accounting"]["epsilon"],
                    "delta": float(result["privacy_accounting"]["delta"]),
                    "ndcg_at_20": float(final["ndcg@20"]),
                    "recall_at_20": float(final["recall@20"]),
                    "catalog_coverage_at_20": float(final["catalog_coverage@20"]),
                    "released_snr_median": finite_median(
                        [entry["released_item_signal_to_noise_ratio"] for entry in diagnostics]
                    ),
                    "clip_rate_mean": float(np.mean([entry["clip_rate"] for entry in diagnostics])),
                    "elapsed_seconds": float(result["runtime"]["elapsed_seconds"]),
                }
                run_rows.append(row)
                run_lookup[(dataset, copy, arm)] = row
    run_frame = pd.DataFrame(run_rows).sort_values(["dataset", "copy", "noise_multiplier"])
    run_frame.to_csv(OUTPUT / "run_level_results.csv", index=False)

    budget_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for dataset, data_dir in DATA_DIRS.items():
        spec = get_pnsmf_fixed_dataset_spec(dataset)
        for copy in SEEDS:
            train_name, validation_name, test_name = spec.filenames(copy)
            data = load_pnsmf_fixed_split(
                data_dir / train_name,
                data_dir / validation_name,
                data_dir / test_name,
                num_users=spec.num_users,
                num_items=spec.num_items,
            )
            matched = run_lookup[(dataset, copy, "matched")]
            control_users, control_items = load_factors(
                FACTORS / f"{run_stem(dataset, copy, 'matched')}.npz"
            )
            for arm in PRIVATE_ARMS:
                private = run_lookup[(dataset, copy, arm)]
                private_users, private_items = load_factors(
                    FACTORS / f"{run_stem(dataset, copy, arm)}.npz"
                )
                user_rows = paired_task_geometry_rows(
                    control_users,
                    control_items,
                    private_users,
                    private_items,
                    data.test_by_user,
                    data.train_by_user,
                    cutoff=20,
                    batch_size=128,
                )
                user_frame = pd.DataFrame(user_rows)
                user_frame.insert(0, "arm", arm)
                user_frame.insert(0, "copy", copy)
                user_frame.insert(0, "dataset", dataset)
                user_frame.to_csv(
                    user_output / f"{dataset}_copy{copy}_{arm}.csv.gz",
                    index=False,
                    compression="gzip",
                )
                geometry_rows.append(
                    {
                        "dataset": dataset,
                        "copy": copy,
                        "arm": arm,
                        "epsilon": float(private["epsilon"]),
                        **summarize_user_geometry(user_frame),
                    }
                )
                matched_ndcg = float(matched["ndcg_at_20"])
                private_ndcg = float(private["ndcg_at_20"])
                budget_rows.append(
                    {
                        "dataset": dataset,
                        "copy": copy,
                        "arm": arm,
                        "epsilon": float(private["epsilon"]),
                        "noise_multiplier": float(private["noise_multiplier"]),
                        "matched_ndcg_at_20": matched_ndcg,
                        "private_ndcg_at_20": private_ndcg,
                        "dp_minus_matched_ndcg": private_ndcg - matched_ndcg,
                        "relative_ndcg_loss": 1.0 - private_ndcg / matched_ndcg,
                        "released_snr_median": private["released_snr_median"],
                    }
                )

    geometry_frame = pd.DataFrame(geometry_rows).sort_values(["dataset", "copy", "epsilon"])
    geometry_frame.to_csv(OUTPUT / "copy_level_task_geometry.csv", index=False)
    budget_frame = pd.DataFrame(budget_rows).sort_values(["dataset", "copy", "epsilon"])
    budget_frame.to_csv(OUTPUT / "copy_level_budget_effects.csv", index=False)

    copy_summary = (
        budget_frame.groupby(["dataset", "epsilon"], sort=True)
        .agg(
            relative_ndcg_loss_mean=("relative_ndcg_loss", "mean"),
            relative_ndcg_loss_sd=("relative_ndcg_loss", "std"),
            private_ndcg_mean=("private_ndcg_at_20", "mean"),
            private_ndcg_sd=("private_ndcg_at_20", "std"),
            released_snr_median_across_copies=("released_snr_median", "median"),
        )
        .reset_index()
    )
    copy_summary.to_csv(OUTPUT / "budget_group_summary.csv", index=False)

    original_epsilon = 2.193078238018666
    gate_frame = geometry_frame[np.isclose(geometry_frame["epsilon"], original_epsilon)]
    gate_checks: dict[str, dict[str, bool]] = {}
    for copy in SEEDS:
        indexed = gate_frame[gate_frame["copy"] == copy].set_index("dataset")
        amazon = indexed.loc["amazon_kindle"]
        gate_checks[str(copy)] = {
            "amazon_ratio_lower_than_both": bool(
                amazon["conservative_stability_ratio_median"]
                < min(
                    indexed.loc["ml1m", "conservative_stability_ratio_median"],
                    indexed.loc["netflix5k5k", "conservative_stability_ratio_median"],
                )
            ),
            "amazon_replacement_higher_than_both": bool(
                amazon["top20_replacement_fraction_mean"]
                > max(
                    indexed.loc["ml1m", "top20_replacement_fraction_mean"],
                    indexed.loc["netflix5k5k", "top20_replacement_fraction_mean"],
                )
            ),
        }
    mechanism_gate_passed = all(all(check.values()) for check in gate_checks.values())

    decisions: dict[str, object] = {}
    for dataset in DATA_DIRS:
        decisions[dataset] = {}
        for epsilon in sorted(budget_frame["epsilon"].unique()):
            selected = budget_frame[
                (budget_frame["dataset"] == dataset)
                & np.isclose(budget_frame["epsilon"], epsilon)
            ]
            conditions = {
                "all_copy_median_snr_below_1": bool((selected["released_snr_median"] < 1).all()),
                "dp_ndcg_lower_in_every_copy": bool((selected["dp_minus_matched_ndcg"] < 0).all()),
                "mean_relative_ndcg_loss_at_least_50_percent": bool(
                    selected["relative_ndcg_loss"].mean() >= 0.5
                ),
            }
            decisions[dataset][f"epsilon={epsilon:.12g}"] = {
                "conditions": conditions,
                "noise_dominated_by_frozen_rule": bool(all(conditions.values())),
                "mean_relative_ndcg_loss": float(selected["relative_ndcg_loss"].mean()),
            }

    audit = {
        "protocol_id": "TASK-GEOMETRY-BUDGET-ANALYSIS-20260808-V1",
        "study_phase": "post_result_analysis_after_main_results_were_inspected",
        "runs": len(run_frame),
        "independent_unit": "fixed_dataset_copy",
        "copies_per_dataset": 3,
        "mechanism_upgrade_gate": {
            "original_epsilon": original_epsilon,
            "copy_checks": gate_checks,
            "passed": mechanism_gate_passed,
        },
        "budget_decisions": decisions,
        "user_level_statistics_are_descriptive": True,
    }
    (OUTPUT / "decision_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
