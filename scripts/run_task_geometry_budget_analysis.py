"""Run the fixed post-result privacy-budget and task-geometry analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "task_geometry_budget_analysis_v1"

DATASETS = {
    "ml1m": {
        "data_dir": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "ML1M-TXT-FORMAT",
        "item_lr": "0.3",
    },
    "amazon_kindle": {
        "data_dir": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Amazon_Kindle_Store-TXT-FORMAT",
        "item_lr": "0.7",
    },
    "netflix5k5k": {
        "data_dir": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Netflix5K5K-TXT-FORAMT",
        "item_lr": "0.24",
    },
}

ARMS = {
    "matched": {"sigma": "0.0"},
    "eps1": {"sigma": "5.888829162483715"},
    "eps2p193": {"sigma": "3.0"},
    "eps4": {"sigma": "1.8856350365894983"},
    "eps8": {"sigma": "1.2158932001187188"},
}

SEEDS = {1: 20260821, 2: 20260822, 3: 20260823}


def main() -> None:
    runs = OUTPUT / "runs"
    factors = OUTPUT / "factors"
    runs.mkdir(parents=True, exist_ok=True)
    factors.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_config in DATASETS.items():
        for copy, seed in SEEDS.items():
            for arm, arm_config in ARMS.items():
                stem = f"{dataset}_copy{copy}_{arm}_seed{seed}"
                result_path = runs / f"{stem}.json"
                factor_path = factors / f"{stem}.npz"
                if result_path.exists() and factor_path.exists():
                    print(f"SKIP complete {stem}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "-m", "scripts.run_dp_pnsmf_bgd",
                    "--data-dir", str(dataset_config["data_dir"]),
                    "--dataset", dataset,
                    "--copy", str(copy),
                    "--output", str(result_path),
                    "--factor-output", str(factor_path),
                    "--seed", str(seed),
                    "--rounds", "200",
                    "--embedding-dim", "20",
                    "--item-learning-rate", dataset_config["item_lr"],
                    "--learning-rate-decay", "0.999",
                    "--sampling-probability", "0.1",
                    "--clip-norm", "1.0",
                    "--clip-mode", "fixed",
                    "--noise-multiplier", arm_config["sigma"],
                    "--delta", "1e-5",
                    "--evaluation-every", "50",
                    "--evaluation-target", "test",
                    "--cutoff", "20",
                    "--report-strata",
                    "--popularity-strata", "training_mass_tertile",
                    "--clip-source", "frozen_TASK_GEOMETRY_BUDGET_ANALYSIS_20260808_V1",
                    "--study-phase", "post_result_analysis",
                    "--contribution-space", "direct_item_gradient",
                    "--contribution-rule", "clip",
                ]
                print(f"RUN {dataset} copy={copy} arm={arm}", flush=True)
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    main()
