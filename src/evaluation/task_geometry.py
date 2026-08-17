"""Descriptive task-geometry diagnostics for paired recommendation models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def paired_task_geometry_rows(
    control_user_factors: torch.Tensor,
    control_item_factors: torch.Tensor,
    private_user_factors: torch.Tensor,
    private_item_factors: torch.Tensor,
    targets_by_user: Sequence[np.ndarray],
    excluded_by_user: Sequence[np.ndarray],
    *,
    cutoff: int = 20,
    batch_size: int = 128,
) -> list[dict[str, float | int]]:
    """Return user-level paired geometry metrics without inferential pooling.

    The control arm defines the rank-``cutoff`` boundary and its score margin.
    The highest-scoring held-out target is used for the target-rank diagnostic.
    Excluded training items are omitted from all candidate-space quantities.
    """

    factors = (
        control_user_factors,
        control_item_factors,
        private_user_factors,
        private_item_factors,
    )
    if any(tensor.ndim != 2 for tensor in factors):
        raise ValueError("All factor matrices must be rank two.")
    if control_user_factors.shape != private_user_factors.shape:
        raise ValueError("Control and private user-factor shapes must match.")
    if control_item_factors.shape != private_item_factors.shape:
        raise ValueError("Control and private item-factor shapes must match.")
    if control_user_factors.shape[1] != control_item_factors.shape[1]:
        raise ValueError("Control embedding dimensions must match.")
    if private_user_factors.shape[1] != private_item_factors.shape[1]:
        raise ValueError("Private embedding dimensions must match.")
    devices = {tensor.device for tensor in factors}
    if len(devices) != 1:
        raise ValueError("All factor matrices must share a device.")
    num_users = control_user_factors.shape[0]
    num_items = control_item_factors.shape[0]
    if len(targets_by_user) != num_users or len(excluded_by_user) != num_users:
        raise ValueError("Targets and exclusions must cover every user.")
    if cutoff <= 0 or cutoff >= num_items:
        raise ValueError("cutoff must be positive and smaller than the catalog.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    evaluated_users = [u for u, targets in enumerate(targets_by_user) if len(targets)]
    rows: list[dict[str, float | int]] = []
    tiny = torch.finfo(control_user_factors.dtype).tiny

    with torch.no_grad():
        for start in range(0, len(evaluated_users), batch_size):
            batch_users = evaluated_users[start : start + batch_size]
            user_index = torch.as_tensor(
                batch_users, dtype=torch.long, device=control_user_factors.device
            )
            control_scores = (
                control_user_factors.index_select(0, user_index)
                @ control_item_factors.T
            )
            private_scores = (
                private_user_factors.index_select(0, user_index)
                @ private_item_factors.T
            )
            candidate_mask = torch.ones_like(control_scores, dtype=torch.bool)
            for row_index, user in enumerate(batch_users):
                excluded = torch.as_tensor(
                    excluded_by_user[user],
                    dtype=torch.long,
                    device=control_user_factors.device,
                )
                candidate_mask[row_index, excluded] = False
            control_rank_scores = control_scores.masked_fill(~candidate_mask, -torch.inf)
            private_rank_scores = private_scores.masked_fill(~candidate_mask, -torch.inf)
            control_values, control_top = torch.topk(
                control_rank_scores, k=cutoff + 1, dim=1, sorted=True
            )
            private_top = torch.topk(
                private_rank_scores, k=cutoff, dim=1, sorted=True
            ).indices
            score_change = (private_scores - control_scores).abs().masked_fill(
                ~candidate_mask, 0.0
            )
            linf = score_change.max(dim=1).values
            boundary_items = control_top[:, cutoff - 1 : cutoff + 1]
            boundary_perturbation = torch.gather(
                score_change, 1, boundary_items
            ).sum(dim=1)
            margin = control_values[:, cutoff - 1] - control_values[:, cutoff]

            for row_index, user in enumerate(batch_users):
                control_set = set(control_top[row_index, :cutoff].cpu().tolist())
                private_set = set(private_top[row_index].cpu().tolist())
                intersection = len(control_set & private_set)
                union = len(control_set | private_set)
                targets = torch.as_tensor(
                    targets_by_user[user],
                    dtype=torch.long,
                    device=control_user_factors.device,
                )
                control_target_score = control_rank_scores[row_index, targets].max()
                private_target_score = private_rank_scores[row_index, targets].max()
                control_target_rank = int(
                    (control_rank_scores[row_index] > control_target_score).sum().item()
                ) + 1
                private_target_rank = int(
                    (private_rank_scores[row_index] > private_target_score).sum().item()
                ) + 1
                local_perturbation = float(boundary_perturbation[row_index])
                max_perturbation = float(linf[row_index])
                margin_value = float(margin[row_index])
                rows.append(
                    {
                        "user": user,
                        "margin_at_20": margin_value,
                        "local_boundary_perturbation": local_perturbation,
                        "margin_to_local_perturbation": margin_value
                        / max(local_perturbation, float(tiny)),
                        "full_catalog_linf_perturbation": max_perturbation,
                        "conservative_stability_ratio": margin_value
                        / max(2.0 * max_perturbation, float(tiny)),
                        "top20_intersection": intersection,
                        "top20_jaccard": intersection / union,
                        "top20_replacement_fraction": 1.0 - intersection / cutoff,
                        "best_target_rank_control": control_target_rank,
                        "best_target_rank_private": private_target_rank,
                        "best_target_absolute_rank_change": abs(
                            private_target_rank - control_target_rank
                        ),
                        "best_target_hit_control": int(control_target_rank <= cutoff),
                        "best_target_hit_private": int(private_target_rank <= cutoff),
                    }
                )
    return rows
