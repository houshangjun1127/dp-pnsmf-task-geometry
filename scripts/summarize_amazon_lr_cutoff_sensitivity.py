"""Summarize learning-rate and cutoff robustness follow-up analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import get_pnsmf_fixed_dataset_spec, load_pnsmf_fixed_split
from src.evaluation import evaluate_pnsmf_factors


ROOT = Path(__file__).resolve().parents[1]
LR_ROOT = ROOT / "results" / "amazon_lr_cutoff_sensitivity_v1"
GEOMETRY_BUDGET_ROOT = ROOT / "results" / "task_geometry_budget_analysis_v1"
SUMMARY_ROOT = LR_ROOT / "summary"
SEEDS = {1: 20260821, 2: 20260822, 3: 20260823}

DATA_DIRS = {
    "ml1m": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "ML1M-TXT-FORMAT",
    "amazon_kindle": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Amazon_Kindle_Store-TXT-FORMAT",
    "netflix5k5k": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Netflix5K5K-TXT-FORAMT",
}


def sample_sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def summarize_paired(
    rows: pd.DataFrame,
    *,
    group_columns: list[str],
    arm_column: str,
    matched_label: str,
    dp_label: str,
    metric: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_key, selected in rows.groupby(group_columns, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        keyed = dict(zip(group_columns, key_values))
        pivot = selected.pivot(index="copy", columns=arm_column, values=metric)
        matched = pivot[matched_label]
        private = pivot[dp_label]
        difference = private - matched
        relative_loss = (matched - private) / matched * 100.0
        records.append(
            {
                **keyed,
                "n_copies": int(len(pivot)),
                "matched_mean": float(matched.mean()),
                "matched_sd": sample_sd(matched),
                "dp_mean": float(private.mean()),
                "dp_sd": sample_sd(private),
                "paired_difference_mean": float(difference.mean()),
                "paired_difference_sd": sample_sd(difference),
                "relative_loss_percent_from_group_means": float(
                    (matched.mean() - private.mean()) / matched.mean() * 100.0
                ),
                "copy_relative_loss_percent_mean": float(relative_loss.mean()),
                "all_copies_dp_below_matched": bool((private < matched).all()),
            }
        )
    return pd.DataFrame.from_records(records)


def learning_rate_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted((LR_ROOT / "runs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        final = payload["history"][-1]["metrics"]
        stem = path.stem
        setting = (
            "equalized_both_lr3"
            if "equalized_both_lr3" in stem
            else "reduced_item_only"
        )
        arm = "user_dp" if "_user_dp_" in stem else "matched"
        records.append(
            {
                "setting": setting,
                "copy": int(config["copy"]),
                "seed": int(config["seed"]),
                "arm": arm,
                "user_learning_rate": float(config["learning_rate"]),
                "item_learning_rate": float(config["item_learning_rate"]),
                "ndcg@20": float(final["ndcg@20"]),
                "recall@20": float(final["recall@20"]),
                "one_call@20": float(final["one_call@20"]),
                "elapsed_seconds": float(payload["runtime"]["elapsed_seconds"]),
            }
        )
    if len(records) != 12:
        raise RuntimeError(f"Expected 12 learning-rate runs, found {len(records)}.")
    return pd.DataFrame.from_records(records)


def cutoff_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, data_dir in DATA_DIRS.items():
        spec = get_pnsmf_fixed_dataset_spec(dataset)
        for copy, seed in SEEDS.items():
            train_name, validation_name, test_name = spec.filenames(copy)
            data = load_pnsmf_fixed_split(
                data_dir / train_name,
                data_dir / validation_name,
                data_dir / test_name,
                num_users=spec.num_users,
                num_items=spec.num_items,
            )
            for arm in ("matched", "eps2p193"):
                factor_path = (
                    GEOMETRY_BUDGET_ROOT
                    / "factors"
                    / f"{dataset}_copy{copy}_{arm}_seed{seed}.npz"
                )
                with np.load(factor_path, allow_pickle=False) as factors:
                    users = torch.as_tensor(factors["user_factors"], dtype=torch.float64)
                    items = torch.as_tensor(factors["item_factors"], dtype=torch.float64)
                for cutoff in (10, 20):
                    metrics = evaluate_pnsmf_factors(
                        users,
                        items,
                        data.test_by_user,
                        data.train_by_user,
                        cutoff=cutoff,
                    )
                    records.append(
                        {
                            "dataset": dataset,
                            "copy": copy,
                            "seed": seed,
                            "arm": arm,
                            "cutoff": cutoff,
                            f"ndcg@{cutoff}": float(metrics[f"ndcg@{cutoff}"]),
                            f"recall@{cutoff}": float(metrics[f"recall@{cutoff}"]),
                            f"one_call@{cutoff}": float(metrics[f"one_call@{cutoff}"]),
                        }
                    )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 36:
        raise RuntimeError(f"Expected 36 cutoff rows, found {len(frame)}.")
    return frame


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    lr = learning_rate_rows()
    lr.to_csv(SUMMARY_ROOT / "learning_rate_run_level.csv", index=False)
    lr_summary = summarize_paired(
        lr,
        group_columns=["setting", "user_learning_rate", "item_learning_rate"],
        arm_column="arm",
        matched_label="matched",
        dp_label="user_dp",
        metric="ndcg@20",
    )
    lr_summary.to_csv(SUMMARY_ROOT / "learning_rate_paired_summary.csv", index=False)

    cutoffs = cutoff_rows()
    cutoffs.to_csv(SUMMARY_ROOT / "cutoff_run_level.csv", index=False)
    cutoff_summary_parts = []
    for cutoff in (10, 20):
        selected = cutoffs.loc[cutoffs["cutoff"] == cutoff].copy()
        selected["metric"] = selected[f"ndcg@{cutoff}"]
        cutoff_summary_parts.append(
            summarize_paired(
                selected,
                group_columns=["dataset", "cutoff"],
                arm_column="arm",
                matched_label="matched",
                dp_label="eps2p193",
                metric="metric",
            )
        )
    cutoff_summary = pd.concat(cutoff_summary_parts, ignore_index=True)
    cutoff_summary.to_csv(SUMMARY_ROOT / "cutoff_paired_summary.csv", index=False)

    audit = {
        "protocol": "AMAZON-LR-CUTOFF-SENSITIVITY-20260809-V1",
        "learning_rate_runs": int(len(lr)),
        "cutoff_evaluations": int(len(cutoffs)),
        "learning_rate_summary": lr_summary.to_dict(orient="records"),
        "cutoff_summary": cutoff_summary.to_dict(orient="records"),
    }
    (SUMMARY_ROOT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
