"""Run the frozen confirmatory query-space feasibility experiment matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "confirmatory_query_feasibility_v1"

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
    "reference_full_nonprivate": {"q": "1.0", "clip": "1000000000", "sigma": "0.0"},
    "matched_clip_no_noise": {"q": "0.1", "clip": "1.0", "sigma": "0.0"},
    "user_dp_full_catalog": {"q": "0.1", "clip": "1.0", "sigma": "3.0"},
}

SEEDS = {1: 20260821, 2: 20260822, 3: 20260823}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_config in DATASETS.items():
        for copy, seed in SEEDS.items():
            for arm, arm_config in ARMS.items():
                destination = OUTPUT / f"{dataset}_copy{copy}_{arm}_seed{seed}.json"
                if destination.exists():
                    print(f"SKIP existing {destination.name}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "-m", "scripts.run_dp_pnsmf_bgd",
                    "--data-dir", str(dataset_config["data_dir"]),
                    "--dataset", dataset,
                    "--copy", str(copy),
                    "--output", str(destination),
                    "--seed", str(seed),
                    "--rounds", "200",
                    "--embedding-dim", "20",
                    "--item-learning-rate", dataset_config["item_lr"],
                    "--learning-rate-decay", "0.999",
                    "--sampling-probability", arm_config["q"],
                    "--clip-norm", arm_config["clip"],
                    "--clip-mode", "fixed",
                    "--noise-multiplier", arm_config["sigma"],
                    "--delta", "1e-5",
                    "--evaluation-every", "50",
                    "--evaluation-target", "test",
                    "--cutoff", "20",
                    "--report-strata",
                    "--popularity-strata", "training_mass_tertile",
                    "--clip-source", "frozen_public_protocol_constant_CONF_QSPACE_20260802_V1",
                    "--study-phase", "confirmatory",
                    "--contribution-space", "direct_item_gradient",
                    "--contribution-rule", "clip",
                ]
                print(f"RUN {dataset} copy={copy} arm={arm}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
