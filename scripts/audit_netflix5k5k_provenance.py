"""Audit the fixed Netflix-5K5K pair files without redistributing the data.

The audit reconstructs each copy's canonical positive-event set in memory,
checks the split invariants reported by Pan and Chen (IJCAI 2013), and writes
only counts and cryptographic hashes.  It deliberately does not claim that the
unpublished raw Netflix user/movie mapping or original random seeds have been
recovered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FILE_PATTERN = "NF5kUsers5kItemsHalfHalf-copy{copy}-{split}"
SOURCE_COMMIT = "b900e205d60f219ae3c2d78b2961a5886a50c0a7"
GBPR_URL = "https://www.ijcai.org/Proceedings/13/Papers/396.pdf"
PNSMF_URL = "https://github.com/PengQ94/P-NSMF"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_pair_sha256(keys: np.ndarray, *, num_items: int) -> str:
    """Hash sorted one-based ``user item\n`` records, independent of file order."""

    digest = hashlib.sha256()
    for key in np.sort(np.asarray(keys, dtype=np.int64)):
        user = int(key // num_items) + 1
        item = int(key % num_items) + 1
        digest.update(f"{user} {item}\n".encode("ascii"))
    return digest.hexdigest()


def _read_pair_keys(path: Path, *, num_users: int, num_items: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    pairs = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if pairs.shape[1] != 2:
        raise ValueError(f"Expected two integer columns in {path}.")
    if len(pairs) == 0:
        return np.empty(0, dtype=np.int64)
    users = pairs[:, 0]
    items = pairs[:, 1]
    if users.min() < 1 or users.max() > num_users:
        raise ValueError(f"User identifier outside 1..{num_users} in {path}.")
    if items.min() < 1 or items.max() > num_items:
        raise ValueError(f"Item identifier outside 1..{num_items} in {path}.")
    keys = (users - 1) * num_items + (items - 1)
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"Duplicate user-item pair in {path}.")
    return keys.astype(np.int64, copy=False)


def audit_netflix5k5k(
    data_dir: Path,
    *,
    num_users: int = 5000,
    num_items: int = 5000,
    copies: tuple[int, ...] = (1, 2, 3),
    expected_half_count: int = 77936,
    expected_positive_count: int = 155872,
) -> dict[str, Any]:
    """Return a serializable audit manifest for fixed Netflix-5K5K files."""

    copy_manifests: list[dict[str, Any]] = []
    canonical_unions: list[np.ndarray] = []
    for copy in copies:
        paths = {
            split: data_dir / FILE_PATTERN.format(copy=copy, split=split)
            for split in ("train", "valid", "test")
        }
        keys = {
            split: _read_pair_keys(
                path, num_users=num_users, num_items=num_items
            )
            for split, path in paths.items()
        }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = np.intersect1d(keys[left], keys[right], assume_unique=True)
            if overlap.size:
                raise ValueError(
                    f"Copy {copy} has {overlap.size} overlapping {left}/{right} pairs."
                )
        validation_users = keys["valid"] // num_items
        validation_counts = np.bincount(validation_users, minlength=num_users)
        if validation_counts.max(initial=0) > 1:
            raise ValueError(f"Copy {copy} validation contains >1 pair for a user.")
        train_validation = np.union1d(keys["train"], keys["valid"])
        union = np.union1d(train_validation, keys["test"])
        if len(train_validation) != expected_half_count:
            raise ValueError(
                f"Copy {copy} train+validation={len(train_validation)}, "
                f"expected {expected_half_count}."
            )
        if len(keys["test"]) != expected_half_count:
            raise ValueError(
                f"Copy {copy} test={len(keys['test'])}, expected {expected_half_count}."
            )
        if len(union) != expected_positive_count:
            raise ValueError(
                f"Copy {copy} union={len(union)}, expected {expected_positive_count}."
            )
        canonical_unions.append(union)
        union_users = union // num_items
        union_items = union % num_items
        copy_manifests.append(
            {
                "copy": copy,
                "counts": {
                    "train": int(len(keys["train"])),
                    "validation": int(len(keys["valid"])),
                    "train_plus_validation": int(len(train_validation)),
                    "test": int(len(keys["test"])),
                    "positive_union": int(len(union)),
                },
                "coverage": {
                    "users_in_positive_union": int(len(np.unique(union_users))),
                    "items_in_positive_union": int(len(np.unique(union_items))),
                    "users_with_validation_pair": int(np.count_nonzero(validation_counts)),
                    "maximum_validation_pairs_per_user": int(
                        validation_counts.max(initial=0)
                    ),
                },
                "canonical_positive_pair_sha256": _canonical_pair_sha256(
                    union, num_items=num_items
                ),
                "files": {
                    split: {
                        "name": path.name,
                        "records": int(len(keys[split])),
                        "sha256": _sha256_file(path),
                    }
                    for split, path in paths.items()
                },
                "invariants": {
                    "pairwise_disjoint_splits": True,
                    "validation_at_most_one_pair_per_user": True,
                    "train_plus_validation_equals_expected_half": True,
                    "test_equals_expected_half": True,
                    "union_equals_expected_positive_count": True,
                },
            }
        )

    reference_union = canonical_unions[0]
    same_union = all(
        np.array_equal(reference_union, candidate)
        for candidate in canonical_unions[1:]
    )
    if not same_union:
        raise ValueError("The three copies do not encode the same positive-event set.")

    return {
        "status": "verified_fixed_split_provenance_audit",
        "protocol_id": "REVIEW-REVISION-NETFLIX-PROVENANCE-20260831-V2",
        "dataset": {
            "name": "Netflix-5K5K",
            "declared_users": num_users,
            "declared_items": num_items,
            "ratings_before_positive_threshold_reported_by_source": 282474,
            "positive_rule_reported_by_source": "rating > 3",
            "positive_events_reconstructed_from_fixed_files": expected_positive_count,
            "identifier_semantics": "anonymous one-based integer user-item pairs",
        },
        "sources": {
            "construction_paper": GBPR_URL,
            "fixed_split_repository": PNSMF_URL,
            "audited_repository_commit": SOURCE_COMMIT,
        },
        "construction_reported_by_primary_source": {
            "sample_5000_users_and_5000_items_from_Netflix_pool": True,
            "retain_ratings_strictly_greater_than_3": True,
            "split_positive_pairs_half_train_half_test": True,
            "move_one_training_pair_per_eligible_user_to_validation": True,
            "repeat_split_three_times": True,
        },
        "copies": copy_manifests,
        "cross_copy": {
            "all_copy_unions_identical": same_union,
            "canonical_positive_pair_sha256": _canonical_pair_sha256(
                reference_union, num_items=num_items
            ),
        },
        "redistribution": {
            "pair_data_written_by_this_audit": False,
            "manifest_contains_only_counts_and_hashes": True,
        },
        "limitations": [
            "The raw Netflix Prize user/movie identifier mapping is not published with the fixed files.",
            "The original 5000-user/5000-item sampling seed is not reported.",
            "This verifies and reconstructs the anonymous positive-event object in memory; it is not a byte-identical rebuild from raw ratings.",
        ],
    }


def _markdown_report(manifest: dict[str, Any]) -> str:
    rows = []
    for copy in manifest["copies"]:
        counts = copy["counts"]
        rows.append(
            f"| {copy['copy']} | {counts['train']:,} | {counts['validation']:,} | "
            f"{counts['train_plus_validation']:,} | {counts['test']:,} | "
            f"{counts['positive_union']:,} | `{copy['canonical_positive_pair_sha256']}` |"
        )
    return "\n".join(
        [
            "# Netflix-5K5K fixed-split provenance audit",
            "",
            "The audit passed all frozen invariants. It reconstructed the anonymous positive-event set in memory and wrote no pair data.",
            "",
            "| Copy | Train | Validation | Train+validation | Test | Union | Canonical union SHA-256 |",
            "|---:|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            f"All copy unions identical: **{manifest['cross_copy']['all_copy_unions_identical']}**.",
            "",
            "Primary-source construction: randomly sample 5,000 users and 5,000 items; retain ratings >3 as positive events; split positives equally into training and test; move one training event per eligible user to validation; repeat three times.",
            "",
            "Limits: the original raw identifier mapping and sampling seed are not public. Therefore this is an exact audit of the distributed anonymous fixed data object, not a raw-data regeneration claim.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = audit_netflix5k5k(args.data_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown_report(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

