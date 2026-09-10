from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_netflix5k5k_provenance import audit_netflix5k5k


def _write_copy(root: Path, copy: int, train: str, valid: str, test: str) -> None:
    values = {"train": train, "valid": valid, "test": test}
    for split, text in values.items():
        path = root / f"NF5kUsers5kItemsHalfHalf-copy{copy}-{split}"
        path.write_text(text, encoding="ascii")


def test_provenance_audit_accepts_identical_unions_across_copies(tmp_path: Path) -> None:
    for copy in (1, 2, 3):
        _write_copy(tmp_path, copy, "1 1\n", "2 2\n", "1 3\n2 1\n")
    result = audit_netflix5k5k(
        tmp_path,
        num_users=2,
        num_items=3,
        expected_half_count=2,
        expected_positive_count=4,
    )
    assert result["cross_copy"]["all_copy_unions_identical"] is True
    assert result["copies"][0]["counts"]["positive_union"] == 4
    assert result["redistribution"]["pair_data_written_by_this_audit"] is False


def test_provenance_audit_rejects_split_overlap(tmp_path: Path) -> None:
    for copy in (1, 2, 3):
        _write_copy(tmp_path, copy, "1 1\n", "1 1\n", "1 3\n2 1\n")
    with pytest.raises(ValueError, match="overlapping train/valid"):
        audit_netflix5k5k(
            tmp_path,
            num_users=2,
            num_items=3,
            expected_half_count=2,
            expected_positive_count=4,
        )
