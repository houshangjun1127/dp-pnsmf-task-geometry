"""Execute and summarize the frozen DPALS-style stress-test protocol."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


TARGET_EPSILON = 2.193078238018666
TEST_SEEDS = {1: 20260911, 2: 20260912, 3: 20260913}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("tune", "test", "all"), default="all")
    return parser.parse_args()


def _run(command: list[str], output: Path) -> None:
    if output.is_file():
        print(f"reuse {output}")
        return
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
            f"DPALS child run failed ({completed.returncode}):\n{completed.stdout}"
        )
    print(f"completed {output}")


def _final_ndcg(path: Path) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))
    return float(result["history"][-1]["metrics"]["ndcg@20"])


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    tune_dir = args.result_dir / "validation_tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for k, rounds, radius, implicit in itertools.product(
        (20, 50), (5, 10), (0.5, 1.0), (0.01, 0.1)
    ):
        label = (
            f"k{k}_T{rounds}_Gu{str(radius).replace('.', 'p')}_"
            f"lambda0{str(implicit).replace('.', 'p')}"
        )
        output = tune_dir / f"{label}.json"
        command = [
            sys.executable,
            "-m",
            "scripts.run_dpals_baseline",
            "--data-dir",
            str(args.data_dir),
            "--copy",
            "1",
            "--output",
            str(output),
            "--seed",
            "20260901",
            "--public-cap-seed",
            "20260900",
            "--embedding-dim",
            "20",
            "--rounds",
            str(rounds),
            "--interaction-cap",
            str(k),
            "--user-clip-radius",
            str(radius),
            "--implicit-penalty",
            str(implicit),
            "--regularization",
            "0.01",
            "--noise-multiplier",
            "0",
            "--evaluation-every",
            "1",
            "--evaluation-target",
            "validation",
            "--cutoff",
            "20",
            "--study-phase",
            "validation_tuning",
        ]
        _run(command, output)
        candidates.append(
            {
                "label": label,
                "interaction_cap": k,
                "rounds": rounds,
                "user_clip_radius": radius,
                "implicit_penalty": implicit,
                "regularization": 0.01,
                "validation_ndcg@20": _final_ndcg(output),
                "result_file": str(output),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda value: (
            -value["validation_ndcg@20"],
            value["interaction_cap"] * value["rounds"],
            value["rounds"],
            value["label"],
        ),
    )
    selection = {
        "protocol_id": "REVIEW-REVISION-DPALS-ML1M-20260831-V2",
        "selection_data": "MovieLens-1M copy 1 validation only",
        "selection_arm": "matched DPALS-style no-noise",
        "selection_rule": "max NDCG@20; then smaller k*T; then smaller T; then lexical label",
        "selected": ranked[0],
        "candidates_ranked": ranked,
    }
    selection_path = args.result_dir / "validation_selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"selected {ranked[0]['label']} NDCG@20={ranked[0]['validation_ndcg@20']:.6f}")
    return selection


def _load_selection(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "validation_selection.json"
    if not path.is_file():
        raise FileNotFoundError("Run the tuning phase before the test phase.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_test(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection["selected"]
    test_dir = args.result_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    result_paths: dict[tuple[int, str], Path] = {}
    for copy, seed in TEST_SEEDS.items():
        for arm in ("matched_no_noise", "private"):
            output = test_dir / f"ml1m_copy{copy}_{arm}_seed{seed}.json"
            command = [
                sys.executable,
                "-m",
                "scripts.run_dpals_baseline",
                "--data-dir",
                str(args.data_dir),
                "--copy",
                str(copy),
                "--output",
                str(output),
                "--seed",
                str(seed),
                "--public-cap-seed",
                "20260900",
                "--embedding-dim",
                "20",
                "--rounds",
                str(selected["rounds"]),
                "--interaction-cap",
                str(selected["interaction_cap"]),
                "--user-clip-radius",
                str(selected["user_clip_radius"]),
                "--implicit-penalty",
                str(selected["implicit_penalty"]),
                "--regularization",
                str(selected["regularization"]),
                "--evaluation-every",
                "1",
                "--evaluation-target",
                "test",
                "--cutoff",
                "20",
                "--study-phase",
                "review_baseline",
            ]
            if arm == "private":
                command.extend(("--target-epsilon", str(TARGET_EPSILON)))
            else:
                command.extend(("--noise-multiplier", "0"))
            _run(command, output)
            result_paths[(copy, arm)] = output

    rows = []
    for copy in TEST_SEEDS:
        matched_path = result_paths[(copy, "matched_no_noise")]
        private_path = result_paths[(copy, "private")]
        matched = json.loads(matched_path.read_text(encoding="utf-8"))
        private = json.loads(private_path.read_text(encoding="utf-8"))
        matched_ndcg = float(matched["history"][-1]["metrics"]["ndcg@20"])
        private_ndcg = float(private["history"][-1]["metrics"]["ndcg@20"])
        rows.append(
            {
                "copy": copy,
                "seed": TEST_SEEDS[copy],
                "matched_ndcg@20": matched_ndcg,
                "private_ndcg@20": private_ndcg,
                "private_minus_matched": private_ndcg - matched_ndcg,
                "relative_loss_percent": 100.0 * (matched_ndcg - private_ndcg) / matched_ndcg,
                "epsilon": private["privacy_accounting"]["epsilon"],
                "noise_multiplier": private["privacy_accounting"]["noise_multiplier"],
                "matched_result": str(matched_path),
                "private_result": str(private_path),
            }
        )
    fields = (
        "matched_ndcg@20",
        "private_ndcg@20",
        "private_minus_matched",
        "relative_loss_percent",
    )
    aggregate = {
        field: {
            "mean": statistics.mean(row[field] for row in rows),
            "sample_sd": statistics.stdev(row[field] for row in rows),
            "values": [row[field] for row in rows],
        }
        for field in fields
    }
    summary = {
        "status": "reviewer_requested_dpals_style_baseline_summary",
        "protocol_id": "REVIEW-REVISION-DPALS-ML1M-20260831-V2",
        "selected_validation_configuration": selected,
        "privacy_target": {"epsilon": TARGET_EPSILON, "delta": 1e-5},
        "copy_results": rows,
        "descriptive_summary": aggregate,
        "inference": "descriptive across three fixed non-independent copies",
        "comparison_limit": "mechanism calibration, not a superiority test",
    }
    summary_path = args.result_dir / "dpals_test_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    selection = None
    if args.phase in ("tune", "all"):
        selection = run_tuning(args)
    if args.phase in ("test", "all"):
        if selection is None:
            selection = _load_selection(args.result_dir)
        summary = run_test(args, selection)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
