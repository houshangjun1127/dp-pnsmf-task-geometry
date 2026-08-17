from pathlib import Path

import numpy as np
import pytest

from src.data import (
    build_pnsmf_public_interaction_order,
    cap_pnsmf_user_interactions,
    get_pnsmf_fixed_dataset_spec,
    load_pnsmf_fixed_split,
    pnsmf_training_strata,
    rotating_pnsmf_user_interactions,
)


def _write_pairs(path: Path, rows: list[tuple[int, int]]) -> None:
    path.write_text("".join(f"{user} {item}\n" for user, item in rows), encoding="utf-8")


def test_fixed_loader_preserves_declared_catalog_and_converts_indices(tmp_path) -> None:
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    test = tmp_path / "test.txt"
    _write_pairs(train, [(1, 1), (1, 3), (2, 2)])
    _write_pairs(valid, [(1, 4), (3, 1)])
    _write_pairs(test, [(2, 5), (3, 4)])

    data = load_pnsmf_fixed_split(
        train, valid, test, num_users=3, num_items=6
    )
    assert data.num_items == 6
    assert data.train_count == 3
    assert data.validation_count == 2
    assert data.test_count == 2
    assert np.array_equal(data.train_by_user[0], np.array([0, 2]))
    assert np.array_equal(data.validation_by_user[2], np.array([0]))
    assert np.array_equal(data.test_by_user[1], np.array([4]))


def test_fixed_loader_rejects_cross_split_overlap(tmp_path) -> None:
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    test = tmp_path / "test.txt"
    _write_pairs(train, [(1, 1)])
    _write_pairs(valid, [(1, 1)])
    _write_pairs(test, [(1, 2)])
    with pytest.raises(ValueError, match="train/validation"):
        load_pnsmf_fixed_split(train, valid, test, num_users=1, num_items=2)


def test_fixed_loader_rejects_duplicate_pairs(tmp_path) -> None:
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    test = tmp_path / "test.txt"
    _write_pairs(train, [(1, 1), (1, 1)])
    _write_pairs(valid, [(1, 2)])
    _write_pairs(test, [(1, 3)])
    with pytest.raises(ValueError, match="Duplicate"):
        load_pnsmf_fixed_split(train, valid, test, num_users=1, num_items=3)


def test_training_strata_use_train_counts_and_keep_declared_items(tmp_path) -> None:
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    test = tmp_path / "test.txt"
    rows = [
        (user, item)
        for user in range(1, 11)
        for item in range(1, user + 1)
    ]
    _write_pairs(train, rows)
    _write_pairs(valid, [(user, 11) for user in range(1, 11)])
    _write_pairs(test, [(user, 12) for user in range(1, 11)])
    data = load_pnsmf_fixed_split(
        train, valid, test, num_users=10, num_items=15
    )
    activity, popularity, user_counts, item_counts = pnsmf_training_strata(data)
    assert set(activity) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert set(popularity).issubset({"P1", "P2", "P3", "P4", "P5"})
    assert np.array_equal(user_counts, np.arange(1, 11))
    assert np.all(item_counts[12:] == 0)
    assert len(popularity) == 15


def test_fixed_dataset_specs_cover_author_file_naming_and_hyperparameters() -> None:
    ml1m = get_pnsmf_fixed_dataset_spec("ml1m")
    assert ml1m.filenames(2) == (
        "ML1M-copy2-train",
        "ML1M-copy2-valid",
        "ML1M-copy2-test",
    )
    amazon = get_pnsmf_fixed_dataset_spec("amazon_kindle")
    assert amazon.num_users == 9862
    assert amazon.num_items == 11298
    assert amazon.paper_bgd_learning_rate == 7.0


def test_training_mass_strata_keep_cold_items_and_ties_together(tmp_path) -> None:
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    test = tmp_path / "test.txt"
    rows = [
        (user, item)
        for user in range(1, 11)
        for item in range(1, user + 1)
    ]
    _write_pairs(train, rows)
    _write_pairs(valid, [(user, 11) for user in range(1, 11)])
    _write_pairs(test, [(user, 12) for user in range(1, 11)])
    data = load_pnsmf_fixed_split(
        train, valid, test, num_users=10, num_items=15
    )
    _, popularity, _, item_counts = pnsmf_training_strata(
        data, popularity_scheme="training_mass_tertile"
    )
    assert np.all(popularity[item_counts == 0] == "M0_cold")
    for count in np.unique(item_counts[item_counts > 0]):
        assert len(set(popularity[item_counts == count])) == 1
    assert set(popularity).issubset(
        {"M0_cold", "M1_tail", "M2_mid", "M3_head"}
    )


def test_public_seed_interaction_cap_is_reproducible_and_bounded() -> None:
    users = np.repeat(np.arange(4, dtype=np.int64), [1, 3, 6, 8])
    items = np.arange(len(users), dtype=np.int64)
    first_users, first_items = cap_pnsmf_user_interactions(
        users,
        items,
        num_users=4,
        interaction_cap=3,
        public_seed=20260802,
    )
    second_users, second_items = cap_pnsmf_user_interactions(
        users,
        items,
        num_users=4,
        interaction_cap=3,
        public_seed=20260802,
    )

    assert np.array_equal(first_users, second_users)
    assert np.array_equal(first_items, second_items)
    assert np.all(np.bincount(first_users, minlength=4) <= 3)
    assert set(first_items).issubset(set(items))
    assert len(first_items) == 1 + 3 + 3 + 3


def test_rotating_cap_covers_large_user_history_across_rounds() -> None:
    users = np.repeat(np.arange(2, dtype=np.int64), [2, 7])
    items = np.arange(len(users), dtype=np.int64)
    order = build_pnsmf_public_interaction_order(
        users,
        num_users=2,
        public_seed=20260802,
    )
    covered = set()
    for round_index in range(7):
        selected_users, selected_items = rotating_pnsmf_user_interactions(
            users,
            items,
            order,
            interaction_cap=3,
            round_index=round_index,
        )
        assert np.all(np.bincount(selected_users, minlength=2) <= 3)
        covered.update(selected_items[selected_users == 1].tolist())

    assert covered == set(items[users == 1])
