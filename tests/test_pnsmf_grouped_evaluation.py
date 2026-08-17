import numpy as np
import torch

from src.evaluation import (
    evaluate_pnsmf_factors,
    evaluate_pnsmf_factors_by_strata,
)


def test_grouped_pnsmf_overall_matches_author_compatible_evaluation() -> None:
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    items = torch.tensor(
        [[0.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.8, 0.2]],
        dtype=torch.float64,
    )
    targets = (np.array([1]), np.array([2]))
    excluded = (np.array([0]), np.array([0]))
    expected = evaluate_pnsmf_factors(
        users, items, targets, excluded, cutoff=2
    )
    grouped = evaluate_pnsmf_factors_by_strata(
        users,
        items,
        targets,
        excluded,
        np.array(["Q1", "Q5"]),
        np.array(["P1", "P5", "P2", "P4"]),
        cutoff=2,
    )
    for key, value in expected.items():
        assert grouped["overall"][key] == value
    assert set(grouped["by_activity"]) == {"Q1", "Q5"}
    assert "Q1|P5" in grouped["by_activity_and_item_popularity"]
    assert "Q5|P2" in grouped["by_activity_and_item_popularity"]
