"""Summarize cross-dataset learning-rate, multi-cutoff, geometry, and dataset audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import get_pnsmf_fixed_dataset_spec, load_pnsmf_fixed_split, pnsmf_training_strata
from src.evaluation import evaluate_pnsmf_factors, paired_task_geometry_rows


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_BUDGET = ROOT / "results" / "task_geometry_budget_analysis_v1"
AMAZON_LR = ROOT / "results" / "amazon_lr_cutoff_sensitivity_v1"
NEW_LR = ROOT / "results" / "cross_dataset_lr_cutoff_sensitivity_v1"
OUTPUT = NEW_LR / "summary"
SEEDS = {1: 20260821, 2: 20260822, 3: 20260823}
CUTOFFS = (5, 10, 20, 50)

DATA_DIRS = {
    "ml1m": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "ML1M-TXT-FORMAT",
    "amazon_kindle": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Amazon_Kindle_Store-TXT-FORMAT",
    "netflix5k5k": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Netflix5K5K-TXT-FORAMT",
}


def load_factors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        users = torch.as_tensor(archive["user_factors"], dtype=torch.float64)
        items = torch.as_tensor(archive["item_factors"], dtype=torch.float64)
    return users, items


def source_paths(dataset: str, setting: str, copy: int, arm: str) -> tuple[Path, Path]:
    seed = SEEDS[copy]
    registered_arm = "matched" if arm == "matched" else "eps2p193"
    if setting == "common_lr3" and dataset == "ml1m":
        stem = f"{dataset}_copy{copy}_{registered_arm}_seed{seed}"
        return GEOMETRY_BUDGET / "runs" / f"{stem}.json", GEOMETRY_BUDGET / "factors" / f"{stem}.npz"
    if setting == "common_lr3" and dataset == "amazon_kindle":
        stem = f"amazon_kindle_equalized_both_lr3_copy{copy}_{arm}_seed{seed}"
        return AMAZON_LR / "runs" / f"{stem}.json", AMAZON_LR / "factors" / f"{stem}.npz"
    if setting == "high_lr7_stress" and dataset == "amazon_kindle":
        stem = f"{dataset}_copy{copy}_{registered_arm}_seed{seed}"
        return GEOMETRY_BUDGET / "runs" / f"{stem}.json", GEOMETRY_BUDGET / "factors" / f"{stem}.npz"
    stem = f"{dataset}_{setting}_copy{copy}_{arm}_seed{seed}"
    return NEW_LR / "runs" / f"{stem}.json", NEW_LR / "factors" / f"{stem}.npz"


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0 or float(x.sum()) == 0.0:
        return 0.0
    index = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x) / (x.size * np.sum(x))) - (x.size + 1.0) / x.size)


def paired_summary(frame: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, selected in frame.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        pivot = selected.pivot(index="copy", columns="arm", values=metric)
        matched = pivot["matched"]
        private = pivot["user_dp"]
        difference = private - matched
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_copies": int(len(pivot)),
            "matched_mean": float(matched.mean()),
            "matched_sd": float(matched.std(ddof=1)),
            "dp_mean": float(private.mean()),
            "dp_sd": float(private.std(ddof=1)),
            "paired_difference_mean": float(difference.mean()),
            "paired_difference_sd": float(difference.std(ddof=1)),
            "relative_loss_percent": float((matched.mean() - private.mean()) / matched.mean() * 100.0),
            "all_copies_dp_below_matched": bool((private < matched).all()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    factor_cache: dict[tuple[str, str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    data_cache = {}
    dataset_rows: list[dict[str, object]] = []
    lr_rows: list[dict[str, object]] = []
    cutoff_rows: list[dict[str, object]] = []
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
            data_cache[(dataset, copy)] = data
            _activity, _popularity, user_counts, item_counts = pnsmf_training_strata(data)
            dataset_rows.append({
                "dataset": dataset,
                "copy": copy,
                "users": spec.num_users,
                "items": spec.num_items,
                "train_events": data.train_count,
                "validation_events": data.validation_count,
                "test_events": data.test_count,
                "density": data.train_count / (spec.num_users * spec.num_items),
                "user_train_median": float(np.median(user_counts)),
                "item_train_median": float(np.median(item_counts)),
                "user_activity_gini": gini(user_counts),
                "item_popularity_gini": gini(item_counts),
                "cold_user_fraction": float(np.mean(user_counts == 0)),
                "cold_item_fraction": float(np.mean(item_counts == 0)),
            })

        for setting in ("common_lr3", "high_lr7_stress"):
            for copy in SEEDS:
                data = data_cache[(dataset, copy)]
                for arm in ("matched", "user_dp"):
                    run_path, factor_path = source_paths(dataset, setting, copy, arm)
                    if not run_path.exists() or not factor_path.exists():
                        raise FileNotFoundError(f"Missing source pair: {run_path} / {factor_path}")
                    payload = json.loads(run_path.read_text(encoding="utf-8"))
                    final = payload["history"][-1]["metrics"]
                    users, items = load_factors(factor_path)
                    factor_cache[(dataset, setting, copy, arm)] = (users, items)
                    lr_rows.append({
                        "dataset": dataset,
                        "setting": setting,
                        "copy": copy,
                        "seed": SEEDS[copy],
                        "arm": arm,
                        "ndcg_at_20": float(final["ndcg@20"]),
                        "recall_at_20": float(final["recall@20"]),
                        "user_learning_rate": float(payload["config"]["learning_rate"]),
                        "item_learning_rate": float(payload["config"]["item_learning_rate"]),
                    })
                    for cutoff in CUTOFFS:
                        metrics = evaluate_pnsmf_factors(
                            users,
                            items,
                            data.test_by_user,
                            data.train_by_user,
                            cutoff=cutoff,
                        )
                        cutoff_rows.append({
                            "dataset": dataset,
                            "setting": setting,
                            "copy": copy,
                            "arm": arm,
                            "cutoff": cutoff,
                            "ndcg": float(metrics[f"ndcg@{cutoff}"]),
                            "recall": float(metrics[f"recall@{cutoff}"]),
                        })

                control_users, control_items = factor_cache[(dataset, setting, copy, "matched")]
                private_users, private_items = factor_cache[(dataset, setting, copy, "user_dp")]
                user_geometry = pd.DataFrame(paired_task_geometry_rows(
                    control_users,
                    control_items,
                    private_users,
                    private_items,
                    data.test_by_user,
                    data.train_by_user,
                    cutoff=20,
                    batch_size=128,
                ))
                geometry_rows.append({
                    "dataset": dataset,
                    "setting": setting,
                    "copy": copy,
                    "users_evaluated": int(len(user_geometry)),
                    "margin_median": float(user_geometry["margin_at_20"].median()),
                    "linf_perturbation_median": float(user_geometry["full_catalog_linf_perturbation"].median()),
                    "stability_ratio_median": float(user_geometry["conservative_stability_ratio"].median()),
                    "top20_replacement_fraction_mean": float(user_geometry["top20_replacement_fraction"].mean()),
                })

    dataset_frame = pd.DataFrame(dataset_rows)
    lr_frame = pd.DataFrame(lr_rows)
    cutoff_frame = pd.DataFrame(cutoff_rows)
    geometry_frame = pd.DataFrame(geometry_rows)
    lr_summary = paired_summary(lr_frame, ["dataset", "setting", "user_learning_rate", "item_learning_rate"], "ndcg_at_20")
    cutoff_summary = paired_summary(cutoff_frame, ["dataset", "setting", "cutoff"], "ndcg")
    geometry_summary = geometry_frame.groupby(["dataset", "setting"], sort=True).agg(
        margin_median_across_copies=("margin_median", "median"),
        linf_perturbation_median_across_copies=("linf_perturbation_median", "median"),
        stability_ratio_median_across_copies=("stability_ratio_median", "median"),
        replacement_fraction_mean=("top20_replacement_fraction_mean", "mean"),
    ).reset_index()

    dataset_frame.to_csv(OUTPUT / "dataset_copy_statistics.csv", index=False)
    lr_frame.to_csv(OUTPUT / "cross_dataset_lr_run_level.csv", index=False)
    lr_summary.to_csv(OUTPUT / "cross_dataset_lr_paired_summary.csv", index=False)
    cutoff_frame.to_csv(OUTPUT / "multi_cutoff_run_level.csv", index=False)
    cutoff_summary.to_csv(OUTPUT / "multi_cutoff_paired_summary.csv", index=False)
    geometry_frame.to_csv(OUTPUT / "lr_geometry_copy_level.csv", index=False)
    geometry_summary.to_csv(OUTPUT / "lr_geometry_summary.csv", index=False)

    audit = {
        "protocol": "XDATA-LR-CUTOFF-SENSITIVITY-20260809-V1",
        "new_run_pairs_expected": 18,
        "dataset_copy_rows": len(dataset_frame),
        "lr_run_rows": len(lr_frame),
        "cutoff_evaluations": len(cutoff_frame),
        "geometry_copy_rows": len(geometry_frame),
        "lr_summary": lr_summary.to_dict(orient="records"),
        "cutoff_summary": cutoff_summary.to_dict(orient="records"),
        "geometry_summary": geometry_summary.to_dict(orient="records"),
    }
    (OUTPUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
