"""Summarize the frozen confirmatory query-space experiment without tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "confirmatory_query_feasibility_v1"
OUTPUT = ROOT / "results" / "confirmatory_query_feasibility_v1_summary"


def finite_median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def main() -> None:
    files = sorted(SOURCE.glob("*.json"))
    if len(files) != 27:
        raise RuntimeError(f"Expected 27 frozen result files, found {len(files)}.")
    rows: list[dict[str, object]] = []
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        final = result["history"][-1]["metrics"]
        diagnostics = result["round_diagnostics"]
        arm = (
            "user_dp_full_catalog"
            if float(config["noise_multiplier"]) > 0
            else "reference_full_nonprivate"
            if float(config["sampling_probability"]) == 1.0
            else "matched_clip_no_noise"
        )
        rows.append(
            {
                "dataset": config["dataset"],
                "copy": int(config["copy"]),
                "seed": int(config["seed"]),
                "arm": arm,
                "status": result["status"],
                "epsilon": result["privacy_accounting"]["epsilon"],
                "delta": result["privacy_accounting"]["delta"],
                "ndcg_at_20": float(final["ndcg@20"]),
                "recall_at_20": float(final["recall@20"]),
                "precision_at_20": float(final["precision@20"]),
                "catalog_coverage_at_20": float(final["catalog_coverage@20"]),
                "released_snr_median": finite_median(
                    [entry["released_item_signal_to_noise_ratio"] for entry in diagnostics]
                ),
                "clip_rate_mean": float(np.mean([entry["clip_rate"] for entry in diagnostics])),
                "final_item_factor_norm": float(diagnostics[-1]["item_factor_norm"]),
                "elapsed_seconds": float(result["runtime"]["elapsed_seconds"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["dataset", "copy", "arm"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "run_level_results.csv", index=False)

    metrics = [
        "ndcg_at_20",
        "recall_at_20",
        "precision_at_20",
        "catalog_coverage_at_20",
        "released_snr_median",
        "clip_rate_mean",
    ]
    summary = (
        frame.groupby(["dataset", "arm"], sort=True)[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(OUTPUT / "group_summary.csv", index=False)

    wide = frame.pivot(index=["dataset", "copy"], columns="arm", values="ndcg_at_20")
    effects = wide.reset_index()
    effects["dp_minus_matched"] = (
        effects["user_dp_full_catalog"] - effects["matched_clip_no_noise"]
    )
    effects["dp_relative_loss_vs_matched"] = 1.0 - (
        effects["user_dp_full_catalog"] / effects["matched_clip_no_noise"]
    )
    effects["matched_minus_reference"] = (
        effects["matched_clip_no_noise"] - effects["reference_full_nonprivate"]
    )
    effects.to_csv(OUTPUT / "paired_ndcg_effects.csv", index=False)

    decisions: dict[str, object] = {}
    for dataset in sorted(frame["dataset"].unique()):
        dataset_effects = effects[effects["dataset"] == dataset]
        dp_rows = frame[(frame["dataset"] == dataset) & (frame["arm"] == "user_dp_full_catalog")]
        conditions = {
            "all_copy_median_snr_below_1": bool((dp_rows["released_snr_median"] < 1).all()),
            "dp_ndcg_lower_in_every_copy": bool((dataset_effects["dp_minus_matched"] < 0).all()),
            "mean_relative_ndcg_loss_at_least_50_percent": bool(
                dataset_effects["dp_relative_loss_vs_matched"].mean() >= 0.5
            ),
        }
        decisions[dataset] = {
            "conditions": conditions,
            "noise_dominated_by_frozen_rule": bool(all(conditions.values())),
            "mean_dp_minus_matched_ndcg_at_20": float(dataset_effects["dp_minus_matched"].mean()),
            "mean_relative_loss": float(dataset_effects["dp_relative_loss_vs_matched"].mean()),
        }

    epsilon_values = sorted(
        frame.loc[frame["arm"] == "user_dp_full_catalog", "epsilon"].dropna().unique().tolist()
    )
    audit = {
        "protocol_id": "CONF-QSPACE-20260802-V1",
        "number_of_runs": len(frame),
        "independent_unit": "fixed_dataset_copy",
        "copies_per_dataset": 3,
        "privacy": {"epsilon_values": epsilon_values, "delta": 1e-5},
        "decisions": decisions,
        "no_null_primary_metrics": bool(frame[["ndcg_at_20", "recall_at_20"]].notna().all().all()),
    }
    (OUTPUT / "decision_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
