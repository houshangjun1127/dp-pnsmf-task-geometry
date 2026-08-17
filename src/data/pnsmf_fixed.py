"""Load the fixed text splits distributed with the P-NSMF reference code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PNSMFFixedSplit:
    """One fixed P-NSMF train/validation/test copy with zero-based indices."""

    num_users: int
    num_items: int
    train_users: np.ndarray
    train_items: np.ndarray
    train_by_user: tuple[np.ndarray, ...]
    validation_by_user: tuple[np.ndarray, ...]
    test_by_user: tuple[np.ndarray, ...]
    train_count: int
    validation_count: int
    test_count: int


@dataclass(frozen=True)
class PNSMFFixedDatasetSpec:
    """Declared catalog and author-file naming for one P-NSMF dataset."""

    key: str
    display_name: str
    num_users: int
    num_items: int
    train_pattern: str
    validation_pattern: str
    test_pattern: str
    paper_omega: float
    paper_regularization: float
    paper_bgd_learning_rate: float

    def filenames(self, copy: int) -> tuple[str, str, str]:
        if copy not in (1, 2, 3):
            raise ValueError("copy must be one of 1, 2, or 3.")
        values = {"copy": copy}
        return (
            self.train_pattern.format(**values),
            self.validation_pattern.format(**values),
            self.test_pattern.format(**values),
        )


PNSMF_FIXED_DATASET_SPECS = {
    "ml1m": PNSMFFixedDatasetSpec(
        key="ml1m",
        display_name="MovieLens-1M",
        num_users=6040,
        num_items=3952,
        train_pattern="ML1M-copy{copy}-train",
        validation_pattern="ML1M-copy{copy}-valid",
        test_pattern="ML1M-copy{copy}-test",
        paper_omega=4.0,
        paper_regularization=0.01,
        paper_bgd_learning_rate=3.0,
    ),
    "netflix5k5k": PNSMFFixedDatasetSpec(
        key="netflix5k5k",
        display_name="Netflix-5K5K",
        num_users=5000,
        num_items=5000,
        train_pattern="NF5kUsers5kItemsHalfHalf-copy{copy}-train",
        validation_pattern="NF5kUsers5kItemsHalfHalf-copy{copy}-valid",
        test_pattern="NF5kUsers5kItemsHalfHalf-copy{copy}-test",
        paper_omega=4.0,
        paper_regularization=0.01,
        paper_bgd_learning_rate=2.4,
    ),
    "xing5k5k": PNSMFFixedDatasetSpec(
        key="xing5k5k",
        display_name="XING-5K5K",
        num_users=5000,
        num_items=5000,
        train_pattern="copy{copy}.train",
        validation_pattern="copy{copy}.valid",
        test_pattern="copy{copy}.test",
        paper_omega=6.0,
        paper_regularization=0.0,
        paper_bgd_learning_rate=8.2,
    ),
    "amazon_kindle": PNSMFFixedDatasetSpec(
        key="amazon_kindle",
        display_name="Amazon-Kindle-Store",
        num_users=9862,
        num_items=11298,
        train_pattern="Amazon_Kindle_Store-copy{copy}-train",
        validation_pattern="Amazon_Kindle_Store-copy{copy}-valid",
        test_pattern="Amazon_Kindle_Store-copy{copy}-test",
        paper_omega=6.0,
        paper_regularization=0.0,
        paper_bgd_learning_rate=7.0,
    ),
}


def get_pnsmf_fixed_dataset_spec(name: str) -> PNSMFFixedDatasetSpec:
    """Return a declared P-NSMF fixed-dataset specification."""

    try:
        return PNSMF_FIXED_DATASET_SPECS[name]
    except KeyError as error:
        choices = ", ".join(sorted(PNSMF_FIXED_DATASET_SPECS))
        raise ValueError(f"Unknown P-NSMF dataset {name!r}; choose from {choices}.") from error


def pnsmf_training_strata(
    data: PNSMFFixedSplit,
    *,
    quantiles: int = 5,
    popularity_scheme: str = "item_count_quintile",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign leakage-safe activity and popularity groups from training counts.

    User bins respect tied activity counts through ``qcut``. Item groups use
    average-rank percentiles so that equal-frequency items remain together;
    declared but unobserved catalog items are retained in the tail group.
    """

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")
    user_counts = np.bincount(data.train_users, minlength=data.num_users)
    item_counts = np.bincount(data.train_items, minlength=data.num_items)
    activity_labels = [f"Q{i}" for i in range(1, quantiles + 1)]
    activity = pd.qcut(
        pd.Series(user_counts),
        q=quantiles,
        labels=activity_labels,
        duplicates="drop",
    )
    if len(activity.cat.categories) != quantiles:
        raise ValueError(
            "Tied user activity counts collapsed quantile bins; revise the "
            "pre-registered binning rule."
        )
    if popularity_scheme == "item_count_quintile":
        popularity_labels = [f"P{i}" for i in range(1, quantiles + 1)]
        popularity_percentile = pd.Series(item_counts).rank(method="average", pct=True)
        popularity = pd.cut(
            popularity_percentile,
            bins=np.linspace(0.0, 1.0, quantiles + 1),
            labels=popularity_labels,
            include_lowest=True,
        ).astype("string").to_numpy(dtype=str)
    elif popularity_scheme == "training_mass_tertile":
        popularity = np.full(data.num_items, "M0_cold", dtype="<U8")
        positive_counts = np.unique(item_counts[item_counts > 0])
        total_mass = int(item_counts.sum())
        cumulative_mass = 0
        labels = ("M1_tail", "M2_mid", "M3_head")
        for count in positive_counts:
            selected = item_counts == count
            block_mass = int(count * selected.sum())
            midpoint_fraction = (
                cumulative_mass + 0.5 * block_mass
            ) / total_mass
            group_index = min(int(midpoint_fraction * 3), 2)
            popularity[selected] = labels[group_index]
            cumulative_mass += block_mass
    else:
        raise ValueError(
            "popularity_scheme must be item_count_quintile or "
            "training_mass_tertile."
        )
    return (
        activity.astype("string").to_numpy(dtype=str),
        popularity,
        user_counts.astype(np.int64, copy=False),
        item_counts.astype(np.int64, copy=False),
    )


def cap_pnsmf_user_interactions(
    train_users: np.ndarray,
    train_items: np.ndarray,
    *,
    num_users: int,
    interaction_cap: int,
    public_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select at most interaction_cap interactions per user.

    Selection is uniform without replacement within each user under a fixed
    public seed. The selected interactions remain private.
    """

    users = np.asarray(train_users, dtype=np.int64)
    items = np.asarray(train_items, dtype=np.int64)
    if users.ndim != 1 or items.ndim != 1 or users.shape != items.shape:
        raise ValueError("train_users and train_items must be aligned vectors.")
    if num_users <= 0 or interaction_cap <= 0:
        raise ValueError("num_users and interaction_cap must be positive.")
    if len(users) and (users.min() < 0 or users.max() >= num_users):
        raise ValueError("train_users contains an out-of-range user index.")

    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.searchsorted(sorted_users, np.arange(num_users + 1))
    selected_parts: list[np.ndarray] = []
    for user_index in range(num_users):
        positions = order[boundaries[user_index] : boundaries[user_index + 1]]
        if len(positions) > interaction_cap:
            generator = np.random.default_rng(
                np.random.SeedSequence([int(public_seed), user_index])
            )
            positions = np.sort(
                generator.choice(positions, size=interaction_cap, replace=False)
            )
        selected_parts.append(positions)
    selected = np.sort(np.concatenate(selected_parts))
    return users[selected], items[selected]


def build_pnsmf_public_interaction_order(
    train_users: np.ndarray,
    *,
    num_users: int,
    public_seed: int,
) -> tuple[np.ndarray, ...]:
    """Return one public-seed permutation of each user's interaction positions."""

    users = np.asarray(train_users, dtype=np.int64)
    if users.ndim != 1:
        raise ValueError("train_users must be a vector.")
    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if len(users) and (users.min() < 0 or users.max() >= num_users):
        raise ValueError("train_users contains an out-of-range user index.")
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.searchsorted(sorted_users, np.arange(num_users + 1))
    result = []
    for user_index in range(num_users):
        positions = order[boundaries[user_index] : boundaries[user_index + 1]].copy()
        generator = np.random.default_rng(
            np.random.SeedSequence([int(public_seed), user_index])
        )
        generator.shuffle(positions)
        result.append(positions)
    return tuple(result)


def rotating_pnsmf_user_interactions(
    train_users: np.ndarray,
    train_items: np.ndarray,
    public_order_by_user: tuple[np.ndarray, ...],
    *,
    interaction_cap: int,
    round_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a cyclic window of at most interaction_cap interactions per user."""

    users = np.asarray(train_users, dtype=np.int64)
    items = np.asarray(train_items, dtype=np.int64)
    if users.ndim != 1 or items.ndim != 1 or users.shape != items.shape:
        raise ValueError("train_users and train_items must be aligned vectors.")
    if interaction_cap <= 0 or round_index < 0:
        raise ValueError("interaction_cap must be positive and round_index non-negative.")
    selected_parts = []
    for positions in public_order_by_user:
        count = len(positions)
        if count <= interaction_cap:
            selected_parts.append(positions)
            continue
        start = (round_index * interaction_cap) % count
        offsets = (start + np.arange(interaction_cap)) % count
        selected_parts.append(positions[offsets])
    selected = np.sort(np.concatenate(selected_parts))
    return users[selected], items[selected]


def _read_pairs(path: Path, *, num_users: int, num_items: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    pairs = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if pairs.shape[1] != 2:
        raise ValueError(f"Expected two integer columns in {path}, got {pairs.shape[1]}.")
    pairs = pairs - 1
    if len(pairs) == 0:
        return pairs
    if pairs[:, 0].min() < 0 or pairs[:, 0].max() >= num_users:
        raise ValueError(f"User index outside [1, {num_users}] in {path}.")
    if pairs[:, 1].min() < 0 or pairs[:, 1].max() >= num_items:
        raise ValueError(f"Item index outside [1, {num_items}] in {path}.")
    keys = pairs[:, 0] * num_items + pairs[:, 1]
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"Duplicate user-item pairs found in {path}.")
    return pairs


def _group_by_user(pairs: np.ndarray, num_users: int) -> tuple[np.ndarray, ...]:
    order = np.argsort(pairs[:, 0], kind="stable")
    sorted_pairs = pairs[order]
    boundaries = np.searchsorted(sorted_pairs[:, 0], np.arange(num_users + 1))
    return tuple(
        np.sort(sorted_pairs[boundaries[u] : boundaries[u + 1], 1]).astype(
            np.int64, copy=False
        )
        for u in range(num_users)
    )


def _assert_disjoint(
    left: np.ndarray,
    right: np.ndarray,
    *,
    num_items: int,
    label: str,
) -> None:
    left_keys = left[:, 0] * num_items + left[:, 1]
    right_keys = right[:, 0] * num_items + right[:, 1]
    if np.intersect1d(left_keys, right_keys, assume_unique=True).size:
        raise ValueError(f"Overlapping user-item pairs found between {label} splits.")


def load_pnsmf_fixed_split(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    *,
    num_users: int,
    num_items: int,
) -> PNSMFFixedSplit:
    """Load and validate one author-provided P-NSMF data copy.

    The files use one-based user and item identifiers separated by whitespace.
    Unlike the project's generic data loader, this loader preserves the complete
    declared item catalog even when an item is absent from the training file.
    """

    if num_users <= 0 or num_items <= 0:
        raise ValueError("num_users and num_items must be positive.")
    train = _read_pairs(Path(train_path), num_users=num_users, num_items=num_items)
    validation = _read_pairs(
        Path(validation_path), num_users=num_users, num_items=num_items
    )
    test = _read_pairs(Path(test_path), num_users=num_users, num_items=num_items)
    _assert_disjoint(train, validation, num_items=num_items, label="train/validation")
    _assert_disjoint(train, test, num_items=num_items, label="train/test")
    _assert_disjoint(validation, test, num_items=num_items, label="validation/test")

    return PNSMFFixedSplit(
        num_users=num_users,
        num_items=num_items,
        train_users=train[:, 0].astype(np.int64, copy=False),
        train_items=train[:, 1].astype(np.int64, copy=False),
        train_by_user=_group_by_user(train, num_users),
        validation_by_user=_group_by_user(validation, num_users),
        test_by_user=_group_by_user(test, num_users),
        train_count=len(train),
        validation_count=len(validation),
        test_count=len(test),
    )
