import numpy as np
import pytest
import torch

from src.evaluation import evaluate_pnsmf_factors


def test_pnsmf_metrics_match_author_definitions() -> None:
    users = torch.tensor([[1.0]], dtype=torch.float64)
    items = torch.tensor([[5.0], [4.0], [3.0], [2.0]], dtype=torch.float64)
    result = evaluate_pnsmf_factors(
        users,
        items,
        targets_by_user=(np.array([1, 3], dtype=np.int64),),
        excluded_by_user=(np.array([0], dtype=np.int64),),
        cutoff=2,
    )
    expected_ndcg = 1.0 / (1.0 + 1.0 / np.log2(3.0))
    assert result["users"] == 1
    assert result["precision@2"] == pytest.approx(0.5)
    assert result["recall@2"] == pytest.approx(0.5)
    assert result["f1@2"] == pytest.approx(0.5)
    assert result["ndcg@2"] == pytest.approx(expected_ndcg)
    assert result["one_call@2"] == pytest.approx(1.0)


def test_fidelity_mode_masks_only_supplied_training_items() -> None:
    users = torch.tensor([[1.0]], dtype=torch.float64)
    items = torch.tensor([[5.0], [4.0], [3.0], [2.0]], dtype=torch.float64)
    targets = (np.array([3], dtype=np.int64),)
    author_protocol = evaluate_pnsmf_factors(
        users,
        items,
        targets,
        excluded_by_user=(np.array([0], dtype=np.int64),),
        cutoff=2,
    )
    train_plus_validation = evaluate_pnsmf_factors(
        users,
        items,
        targets,
        excluded_by_user=(np.array([0, 1], dtype=np.int64),),
        cutoff=2,
    )
    assert author_protocol["recall@2"] == 0.0
    assert train_plus_validation["recall@2"] == 1.0


def test_users_without_targets_are_excluded_from_macro_average() -> None:
    users = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    items = torch.tensor([[3.0], [2.0], [1.0]], dtype=torch.float64)
    result = evaluate_pnsmf_factors(
        users,
        items,
        targets_by_user=(np.array([1]), np.array([], dtype=np.int64)),
        excluded_by_user=(np.array([0]), np.array([0])),
        cutoff=1,
    )
    assert result["users"] == 1
    assert result["precision@1"] == 1.0
