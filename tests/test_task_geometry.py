import numpy as np
import torch

from src.evaluation import paired_task_geometry_rows


def test_paired_task_geometry_detects_topk_replacement_and_target_rank_change() -> None:
    control_users = torch.tensor([[1.0]], dtype=torch.float64)
    control_items = torch.tensor([[5.0], [4.0], [3.0], [2.0]], dtype=torch.float64)
    private_users = torch.tensor([[1.0]], dtype=torch.float64)
    private_items = torch.tensor([[5.0], [1.0], [4.5], [2.0]], dtype=torch.float64)
    rows = paired_task_geometry_rows(
        control_users,
        control_items,
        private_users,
        private_items,
        [np.asarray([1], dtype=np.int64)],
        [np.asarray([], dtype=np.int64)],
        cutoff=2,
    )
    row = rows[0]
    assert row["margin_at_20"] == 1.0
    assert row["top20_intersection"] == 1
    assert row["top20_replacement_fraction"] == 0.5
    assert row["top20_jaccard"] == 1 / 3
    assert row["best_target_rank_control"] == 2
    assert row["best_target_rank_private"] == 4


def test_paired_task_geometry_excludes_training_items_from_candidate_metrics() -> None:
    users = torch.tensor([[1.0]], dtype=torch.float64)
    control_items = torch.tensor([[100.0], [4.0], [3.0], [2.0]], dtype=torch.float64)
    private_items = torch.tensor([[-100.0], [4.0], [3.0], [2.0]], dtype=torch.float64)
    row = paired_task_geometry_rows(
        users,
        control_items,
        users,
        private_items,
        [np.asarray([2], dtype=np.int64)],
        [np.asarray([0], dtype=np.int64)],
        cutoff=2,
    )[0]
    assert row["full_catalog_linf_perturbation"] == 0.0
    assert row["top20_replacement_fraction"] == 0.0
