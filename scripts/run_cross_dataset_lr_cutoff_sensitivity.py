"""Run the fixed cross-dataset learning-rate and cutoff sensitivity cells."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "cross_dataset_lr_cutoff_sensitivity_v1"
SEEDS = {1: 20260821, 2: 20260822, 3: 20260823}

DATASETS = {
    "ml1m": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "ML1M-TXT-FORMAT",
    "netflix5k5k": ROOT / "references" / "vendor" / "P-NSMF-upstream" / "data" / "Netflix5K5K-TXT-FORAMT",
}

NEW_CELLS = (
    ("ml1m", "high_lr7_stress", "7.0", "0.7"),
    ("netflix5k5k", "common_lr3", "3.0", "0.3"),
    ("netflix5k5k", "high_lr7_stress", "7.0", "0.7"),
)

ARMS = {
    "matched": "0.0",
    "user_dp": "3.0",
}


def main() -> None:
    runs = OUTPUT / "runs"
    factors = OUTPUT / "factors"
    runs.mkdir(parents=True, exist_ok=True)
    factors.mkdir(parents=True, exist_ok=True)
    for dataset, setting, user_lr, item_lr in NEW_CELLS:
        for copy, seed in SEEDS.items():
            for arm, sigma in ARMS.items():
                stem = f"{dataset}_{setting}_copy{copy}_{arm}_seed{seed}"
                result_path = runs / f"{stem}.json"
                factor_path = factors / f"{stem}.npz"
                if result_path.exists() and factor_path.exists():
                    print(f"SKIP complete {stem}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "scripts.run_dp_pnsmf_bgd",
                    "--data-dir",
                    str(DATASETS[dataset]),
                    "--dataset",
                    dataset,
                    "--copy",
                    str(copy),
                    "--output",
                    str(result_path),
                    "--factor-output",
                    str(factor_path),
                    "--seed",
                    str(seed),
                    "--rounds",
                    "200",
                    "--embedding-dim",
                    "20",
                    "--learning-rate",
                    user_lr,
                    "--item-learning-rate",
                    item_lr,
                    "--learning-rate-decay",
                    "0.999",
                    "--sampling-probability",
                    "0.1",
                    "--clip-norm",
                    "1.0",
                    "--clip-mode",
                    "fixed",
                    "--noise-multiplier",
                    sigma,
                    "--delta",
                    "1e-5",
                    "--evaluation-every",
                    "50",
                    "--evaluation-target",
                    "test",
                    "--cutoff",
                    "20",
                    "--clip-source",
                    "frozen_XDATA_LR_CUTOFF_SENSITIVITY_20260809_V1",
                    "--study-phase",
                    "post_result_analysis",
                    "--contribution-space",
                    "direct_item_gradient",
                    "--contribution-rule",
                    "clip",
                ]
                print(f"RUN {dataset} setting={setting} copy={copy} arm={arm}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
