"""Run the original robustness protocol with the dated 2026-09-07 amendment.

The amendment corrects shared-seed resampling and completes convergence
coverage. Original run outputs are reused; the original frozen YAML is retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = {
    "ml1m": {
        "directory": "ML1M-TXT-FORMAT",
        "user_lr": 3.0,
        "item_lr": 0.3,
    },
    "amazon_kindle": {
        "directory": "Amazon_Kindle_Store-TXT-FORMAT",
        "user_lr": 7.0,
        "item_lr": 0.7,
    },
    "netflix5k5k": {
        "directory": "Netflix5K5K-TXT-FORAMT",
        "user_lr": 2.4,
        "item_lr": 0.24,
    },
}
COPIES = (1, 2, 3)
MULTISEEDS = tuple(range(20260901, 20260911))
SENSITIVITY_SEEDS = {1: 20260921, 2: 20260922, 3: 20260923}
CONVERGENCE_SEEDS = {1: 20260924, 2: 20260925, 3: 20260926}
AMENDMENT_ID = "REVIEW-COMPLETION-20260907-V2.1"
SENSITIVITY = (
    ("d10_C1", 10, 1.0),
    ("d20_C0p5", 20, 0.5),
    ("d20_C1", 20, 1.0),
    ("d20_C2", 20, 2.0),
    ("d40_C1", 40, 1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("multiseed", "sensitivity", "convergence", "all", "summarize"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _run_child(command: list[str], output: Path) -> str:
    if output.is_file():
        return f"reuse {output.name}"
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for {output} ({completed.returncode}):\n{completed.stdout}"
        )
    return f"completed {output.name}"


def _base_command(
    *,
    data_root: Path,
    dataset: str,
    copy: int,
    output: Path,
    seed: int,
    arm: str,
    embedding_dim: int,
    clip_norm: float,
    user_lr: float,
    item_lr: float,
    rounds: int,
    evaluation_every: int,
) -> list[str]:
    spec = DATASETS[dataset]
    return [
        sys.executable,
        "-m",
        "scripts.run_dp_pnsmf_bgd",
        "--data-dir",
        str(data_root / spec["directory"]),
        "--dataset",
        dataset,
        "--copy",
        str(copy),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--rounds",
        str(rounds),
        "--embedding-dim",
        str(embedding_dim),
        "--learning-rate",
        str(user_lr),
        "--item-learning-rate",
        str(item_lr),
        "--learning-rate-decay",
        "0.999",
        "--sampling-probability",
        "0.1",
        "--clip-norm",
        str(clip_norm),
        "--noise-multiplier",
        "3.0" if arm == "dp" else "0.0",
        "--delta",
        "1e-5",
        "--evaluation-every",
        str(evaluation_every),
        "--evaluation-target",
        "test",
        "--cutoff",
        "20",
        "--clip-source",
        "frozen_REVIEW_REVISION_20260831_V2",
        "--study-phase",
        "registered_followup",
        "--contribution-space",
        "direct_item_gradient",
        "--contribution-rule",
        "clip",
        "--torch-threads",
        "1",
    ]


def _execute(jobs: list[tuple[list[str], Path]], workers: int) -> None:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_child, command, output): output
            for command, output in jobs
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            message = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(f"{completed}/{len(jobs)}: {message}", flush=True)


def build_multiseed_jobs(args: argparse.Namespace) -> list[tuple[list[str], Path]]:
    output_dir = args.result_dir / "multiseed" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, copy, seed, arm in (
        (dataset, copy, seed, arm)
        for dataset in DATASETS
        for copy in COPIES
        for seed in MULTISEEDS
        for arm in ("matched", "dp")
    ):
        spec = DATASETS[dataset]
        output = output_dir / f"{dataset}_copy{copy}_seed{seed}_{arm}.json"
        jobs.append(
            (
                _base_command(
                    data_root=args.data_root,
                    dataset=dataset,
                    copy=copy,
                    output=output,
                    seed=seed,
                    arm=arm,
                    embedding_dim=20,
                    clip_norm=1.0,
                    user_lr=spec["user_lr"],
                    item_lr=spec["item_lr"],
                    rounds=200,
                    evaluation_every=200,
                ),
                output,
            )
        )
    return jobs


def build_sensitivity_jobs(args: argparse.Namespace) -> list[tuple[list[str], Path]]:
    output_dir = args.result_dir / "sensitivity" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, copy, (label, dimension, clip_norm), arm in (
        (dataset, copy, config, arm)
        for dataset in DATASETS
        for copy in COPIES
        for config in SENSITIVITY
        for arm in ("matched", "dp")
    ):
        spec = DATASETS[dataset]
        seed = SENSITIVITY_SEEDS[copy]
        output = output_dir / f"{dataset}_copy{copy}_{label}_seed{seed}_{arm}.json"
        jobs.append(
            (
                _base_command(
                    data_root=args.data_root,
                    dataset=dataset,
                    copy=copy,
                    output=output,
                    seed=seed,
                    arm=arm,
                    embedding_dim=dimension,
                    clip_norm=clip_norm,
                    user_lr=spec["user_lr"],
                    item_lr=spec["item_lr"],
                    rounds=200,
                    evaluation_every=200,
                ),
                output,
            )
        )
    return jobs


def _rate_configs(dataset: str) -> tuple[tuple[str, float, float], ...]:
    spec = DATASETS[dataset]
    values = [("published", spec["user_lr"], spec["item_lr"])]
    if not (math.isclose(spec["user_lr"], 3.0) and math.isclose(spec["item_lr"], 0.3)):
        values.append(("common_low", 3.0, 0.3))
    if dataset == "amazon_kindle":
        values.append(("mixed_item_low", 7.0, 0.3))
    else:
        values.append(("common_high", 7.0, 0.7))
    return tuple(values)


def build_convergence_jobs(args: argparse.Namespace) -> list[tuple[list[str], Path]]:
    output_dir = args.result_dir / "convergence" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in DATASETS:
        for copy in COPIES:
            for label, user_lr, item_lr in _rate_configs(dataset):
                for arm in ("matched", "dp"):
                    seed = CONVERGENCE_SEEDS[copy]
                    output = output_dir / (
                        f"{dataset}_copy{copy}_{label}_seed{seed}_{arm}.json"
                    )
                    jobs.append(
                        (
                            _base_command(
                                data_root=args.data_root,
                                dataset=dataset,
                                copy=copy,
                                output=output,
                                seed=seed,
                                arm=arm,
                                embedding_dim=20,
                                clip_norm=1.0,
                                user_lr=user_lr,
                                item_lr=item_lr,
                                rounds=200,
                                evaluation_every=10,
                            ),
                            output,
                        )
                    )
    return jobs


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ndcg(result: dict[str, Any]) -> float:
    return float(result["history"][-1]["metrics"]["ndcg@20"])


def seed_block_bootstrap(
    values: np.ndarray, *, replicates: int = 10000, seed: int = 20260930
) -> np.ndarray:
    """Resample seed columns together across all fixed-copy rows.

    Copies share initialization and sampling seeds, so independent within-copy
    resampling would discard cross-copy dependence. Copies themselves are fixed.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Expected a finite nonempty copy-by-seed matrix.")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    indices = np.random.default_rng(seed).integers(
        0, values.shape[1], size=(replicates, values.shape[1])
    )
    return values.mean(axis=0)[indices].mean(axis=1)


def summarize_multiseed(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.result_dir / "multiseed" / "runs"
    records = []
    for dataset in DATASETS:
        for copy in COPIES:
            for seed in MULTISEEDS:
                matched_path = run_dir / f"{dataset}_copy{copy}_seed{seed}_matched.json"
                dp_path = run_dir / f"{dataset}_copy{copy}_seed{seed}_dp.json"
                matched = _ndcg(_read(matched_path))
                private = _ndcg(_read(dp_path))
                records.append(
                    {
                        "dataset": dataset,
                        "copy": copy,
                        "seed": seed,
                        "matched_ndcg@20": matched,
                        "private_ndcg@20": private,
                        "private_minus_matched": private - matched,
                        "relative_loss_percent": 100.0 * (matched - private) / matched,
                    }
                )
    summaries = {}
    for dataset in DATASETS:
        selected = [record for record in records if record["dataset"] == dataset]
        effects = np.asarray(
            [record["private_minus_matched"] for record in selected], dtype=float
        ).reshape(len(COPIES), len(MULTISEEDS))
        relative = np.asarray(
            [record["relative_loss_percent"] for record in selected], dtype=float
        ).reshape(len(COPIES), len(MULTISEEDS))
        bootstrap_effect = seed_block_bootstrap(effects)
        bootstrap_relative = seed_block_bootstrap(relative)
        summaries[dataset] = {
            "fixed_copy_seed_pairs": len(selected),
            "matched_ndcg@20_mean": statistics.mean(
                record["matched_ndcg@20"] for record in selected
            ),
            "private_ndcg@20_mean": statistics.mean(
                record["private_ndcg@20"] for record in selected
            ),
            "private_minus_matched_mean": float(effects.mean()),
            "private_minus_matched_algorithmic_95pct_interval": np.quantile(
                bootstrap_effect, [0.025, 0.975]
            ).tolist(),
            "relative_loss_percent_mean": float(relative.mean()),
            "relative_loss_algorithmic_95pct_interval": np.quantile(
                bootstrap_relative, [0.025, 0.975]
            ).tolist(),
            "copy_means": {
                str(copy): {
                    "private_minus_matched": float(effects[index].mean()),
                    "relative_loss_percent": float(relative[index].mean()),
                }
                for index, copy in enumerate(COPIES)
            },
        }
    result = {
        "status": "fixed_copy_conditional_algorithmic_uncertainty_summary",
        "protocol_id": "REVIEW-REVISION-SENSITIVITY-UNCERTAINTY-20260831-V2",
        "analysis_amendment": AMENDMENT_ID,
        "resampling": {
            "unit": "seed_block_shared_across_three_fixed_copies",
            "blocks": len(MULTISEEDS),
            "replicates": 10000,
            "rng_seed_per_dataset": 20260930,
            "copy_weighting": "equal",
            "interval": "percentile_2.5_97.5",
        },
        "records": records,
        "dataset_summary": summaries,
        "uncertainty_scope": "algorithmic randomness conditional on three fixed non-independent copies",
        "not_a_claim": "not a dataset-population confidence interval",
    }
    output = args.result_dir / "multiseed" / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def summarize_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.result_dir / "sensitivity" / "runs"
    rows = []
    for dataset in DATASETS:
        for label, dimension, clip_norm in SENSITIVITY:
            copy_rows = []
            for copy in COPIES:
                seed = SENSITIVITY_SEEDS[copy]
                prefix = f"{dataset}_copy{copy}_{label}_seed{seed}"
                matched = _ndcg(_read(run_dir / f"{prefix}_matched.json"))
                private = _ndcg(_read(run_dir / f"{prefix}_dp.json"))
                copy_rows.append(
                    {
                        "copy": copy,
                        "matched": matched,
                        "private": private,
                        "relative_loss_percent": 100.0 * (matched - private) / matched,
                    }
                )
            rows.append(
                {
                    "dataset": dataset,
                    "configuration": label,
                    "embedding_dim": dimension,
                    "clip_norm": clip_norm,
                    "matched_ndcg@20_mean": statistics.mean(
                        row["matched"] for row in copy_rows
                    ),
                    "private_ndcg@20_mean": statistics.mean(
                        row["private"] for row in copy_rows
                    ),
                    "relative_loss_percent_mean": statistics.mean(
                        row["relative_loss_percent"] for row in copy_rows
                    ),
                    "copy_results": copy_rows,
                }
            )
    result = {
        "status": "one_factor_C_d_sensitivity_summary",
        "protocol_id": "REVIEW-REVISION-SENSITIVITY-UNCERTAINTY-20260831-V2",
        "rows": rows,
        "interpretation": "post-decision local one-factor sensitivity, not global robustness",
    }
    output = args.result_dir / "sensitivity" / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _history_at(result: dict[str, Any], round_index: int) -> dict[str, Any]:
    return next(row for row in result["history"] if row["round"] == round_index)


def summarize_convergence_and_build_extensions(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[tuple[list[str], Path]]]:
    run_dir = args.result_dir / "convergence" / "runs"
    extension_dir = args.result_dir / "convergence" / "extensions"
    extension_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    extension_jobs = []
    for dataset in DATASETS:
        for copy in COPIES:
            for label, user_lr, item_lr in _rate_configs(dataset):
                seed = CONVERGENCE_SEEDS[copy]
                arm_results = {}
                for arm in ("matched", "dp"):
                    path = run_dir / f"{dataset}_copy{copy}_{label}_seed{seed}_{arm}.json"
                    arm_results[arm] = _read(path)
                matched_150 = _history_at(arm_results["matched"], 150)
                matched_200 = _history_at(arm_results["matched"], 200)
                ndcg150 = float(matched_150["metrics"]["ndcg@20"])
                ndcg200 = float(matched_200["metrics"]["ndcg@20"])
                objective150 = float(matched_150["normalized_training_objective"])
                objective200 = float(matched_200["normalized_training_objective"])
                ndcg_gain = (
                    (ndcg200 - ndcg150) / abs(ndcg150) if ndcg150 != 0.0 else math.inf
                )
                objective_drop = (objective150 - objective200) / abs(objective150)
                trigger = ndcg_gain > 0.05 or objective_drop > 0.01
                row = {
                    "dataset": dataset,
                    "copy": copy,
                    "rate_label": label,
                    "user_learning_rate": user_lr,
                    "item_learning_rate": item_lr,
                    "matched_ndcg@20_round150": ndcg150,
                    "matched_ndcg@20_round200": ndcg200,
                    "matched_relative_ndcg_gain_150_to_200": ndcg_gain,
                    "matched_relative_objective_drop_150_to_200": objective_drop,
                    "extension_triggered": trigger,
                    "private_ndcg@20_round200": float(
                        _history_at(arm_results["dp"], 200)["metrics"]["ndcg@20"]
                    ),
                }
                rows.append(row)
                if trigger:
                    output = extension_dir / (
                        f"{dataset}_copy{copy}_{label}_seed{seed}_matched_r800.json"
                    )
                    extension_jobs.append(
                        (
                            _base_command(
                                data_root=args.data_root,
                                dataset=dataset,
                                copy=copy,
                                output=output,
                                seed=seed,
                                arm="matched",
                                embedding_dim=20,
                                clip_norm=1.0,
                                user_lr=user_lr,
                                item_lr=item_lr,
                                rounds=800,
                                evaluation_every=50,
                            ),
                            output,
                        )
                    )
    result = {
        "status": "learning_rate_convergence_trigger_summary",
        "protocol_id": "REVIEW-REVISION-SENSITIVITY-UNCERTAINTY-20260831-V2",
        "coverage_amendment": AMENDMENT_ID,
        "rows": rows,
        "extension_rule": "matched arm only; >5% NDCG gain or >1% objective drop from round 150 to 200",
    }
    output = args.result_dir / "convergence" / "trigger_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result, extension_jobs


def summarize_extensions(args: argparse.Namespace, trigger: dict[str, Any]) -> dict[str, Any]:
    extension_dir = args.result_dir / "convergence" / "extensions"
    rows = []
    for row in trigger["rows"]:
        if not row["extension_triggered"]:
            continue
        dataset = row["dataset"]
        copy = row["copy"]
        label = row["rate_label"]
        seed = CONVERGENCE_SEEDS[copy]
        path = extension_dir / f"{dataset}_copy{copy}_{label}_seed{seed}_matched_r800.json"
        result = _read(path)
        rows.append(
            {
                "dataset": dataset,
                "copy": copy,
                "rate_label": label,
                "ndcg@20_round200": float(_history_at(result, 200)["metrics"]["ndcg@20"]),
                "ndcg@20_round400": float(_history_at(result, 400)["metrics"]["ndcg@20"]),
                "ndcg@20_round800": float(_history_at(result, 800)["metrics"]["ndcg@20"]),
                "objective_round200": float(_history_at(result, 200)["normalized_training_objective"]),
                "objective_round800": float(_history_at(result, 800)["normalized_training_objective"]),
                "result_file": str(path),
            }
        )
    result = {
        "status": "matched_control_long_horizon_convergence_summary",
        "protocol_id": "REVIEW-REVISION-SENSITIVITY-UNCERTAINTY-20260831-V2",
        "coverage_amendment": AMENDMENT_ID,
        "rows": rows,
        "privacy_scope": "not private; optimization diagnostic only",
    }
    output = args.result_dir / "convergence" / "extension_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("multiseed", "all"):
        _execute(build_multiseed_jobs(args), args.workers)
        summarize_multiseed(args)
    if args.phase in ("sensitivity", "all"):
        _execute(build_sensitivity_jobs(args), args.workers)
        summarize_sensitivity(args)
    if args.phase in ("convergence", "all"):
        _execute(build_convergence_jobs(args), args.workers)
        trigger, extension_jobs = summarize_convergence_and_build_extensions(args)
        _execute(extension_jobs, args.workers)
        summarize_extensions(args, trigger)
    if args.phase == "summarize":
        summarize_multiseed(args)
        summarize_sensitivity(args)
        trigger, _ = summarize_convergence_and_build_extensions(args)
        summarize_extensions(args, trigger)


if __name__ == "__main__":
    main()
