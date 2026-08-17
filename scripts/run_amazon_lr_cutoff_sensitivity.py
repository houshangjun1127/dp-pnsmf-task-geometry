"""Run the fixed Amazon learning-rate and cutoff sensitivity matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT
    / "references"
    / "vendor"
    / "P-NSMF-upstream"
    / "data"
    / "Amazon_Kindle_Store-TXT-FORMAT"
)
OUTPUT = ROOT / "results" / "amazon_lr_cutoff_sensitivity_v1"

SETTINGS = {
    "equalized_both_lr3": {"user_lr": "3.0", "item_lr": "0.3"},
    "reduced_item_only": {"user_lr": "7.0", "item_lr": "0.3"},
}
ARMS = {
    "matched": {"sigma": "0.0"},
    "user_dp": {"sigma": "3.0"},
}
SEEDS = {1: 20260821, 2: 20260822, 3: 20260823}


def main() -> None:
    runs = OUTPUT / "runs"
    factors = OUTPUT / "factors"
    runs.mkdir(parents=True, exist_ok=True)
    factors.mkdir(parents=True, exist_ok=True)
    for setting, learning_rates in SETTINGS.items():
        for copy, seed in SEEDS.items():
            for arm, arm_config in ARMS.items():
                stem = f"amazon_kindle_{setting}_copy{copy}_{arm}_seed{seed}"
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
                    str(DATA_DIR),
                    "--dataset",
                    "amazon_kindle",
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
                    learning_rates["user_lr"],
                    "--item-learning-rate",
                    learning_rates["item_lr"],
                    "--learning-rate-decay",
                    "0.999",
                    "--sampling-probability",
                    "0.1",
                    "--clip-norm",
                    "1.0",
                    "--clip-mode",
                    "fixed",
                    "--noise-multiplier",
                    arm_config["sigma"],
                    "--delta",
                    "1e-5",
                    "--evaluation-every",
                    "50",
                    "--evaluation-target",
                    "test",
                    "--cutoff",
                    "20",
                    "--clip-source",
                    "frozen_AMAZON_LR_CUTOFF_SENSITIVITY_20260809_V1",
                    "--study-phase",
                    "post_result_analysis",
                    "--contribution-space",
                    "direct_item_gradient",
                    "--contribution-rule",
                    "clip",
                ]
                print(
                    f"RUN setting={setting} copy={copy} arm={arm}",
                    flush=True,
                )
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    main()
