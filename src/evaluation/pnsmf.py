"""Ranking evaluation compatible with the public P-NSMF implementation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def evaluate_pnsmf_factors(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    targets_by_user: Sequence[np.ndarray],
    excluded_by_user: Sequence[np.ndarray],
    *,
    cutoff: int = 5,
    batch_size: int = 256,
) -> dict[str, float | int]:
    """Evaluate factor matrices using the author's full-ranking convention.

    Only entries in ``excluded_by_user`` are masked.  For faithful final-test
    reproduction this must be the author's train file, not train plus validation.
    Users without target interactions are omitted from the macro average.
    """

    if user_factors.ndim != 2 or item_factors.ndim != 2:
        raise ValueError("User and item factors must be rank-two tensors.")
    if user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("User and item embedding dimensions must match.")
    num_users = user_factors.shape[0]
    num_items = item_factors.shape[0]
    if len(targets_by_user) != num_users or len(excluded_by_user) != num_users:
        raise ValueError("Target and exclusion sequences must cover every user.")
    if cutoff <= 0 or cutoff > num_items:
        raise ValueError("cutoff must be in [1, num_items].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if user_factors.device != item_factors.device:
        raise ValueError("User and item factors must be on the same device.")

    evaluated_users = [u for u, targets in enumerate(targets_by_user) if len(targets)]
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    ndcg_sum = 0.0
    one_call_sum = 0.0

    with torch.no_grad():
        for start in range(0, len(evaluated_users), batch_size):
            batch_users = evaluated_users[start : start + batch_size]
            user_index = torch.as_tensor(
                batch_users, dtype=torch.long, device=user_factors.device
            )
            scores = user_factors.index_select(0, user_index) @ item_factors.T
            for row, user in enumerate(batch_users):
                excluded = torch.as_tensor(
                    excluded_by_user[user],
                    dtype=torch.long,
                    device=user_factors.device,
                )
                scores[row, excluded] = -torch.inf
            recommendations = torch.topk(
                scores, k=cutoff, dim=1, largest=True, sorted=True
            ).indices.cpu().numpy()

            for row, user in enumerate(batch_users):
                targets = np.asarray(targets_by_user[user], dtype=np.int64)
                hits = np.isin(recommendations[row], targets, assume_unique=False)
                hit_count = int(hits.sum())
                precision = hit_count / cutoff
                recall = hit_count / len(targets)
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if precision + recall > 0.0
                    else 0.0
                )
                discounts = 1.0 / np.log2(np.arange(cutoff) + 2.0)
                dcg = float(discounts[hits].sum())
                ideal_count = min(len(targets), cutoff)
                idcg = float(discounts[:ideal_count].sum())

                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                ndcg_sum += dcg / idcg
                one_call_sum += float(hit_count > 0)

    count = len(evaluated_users)
    if count == 0:
        raise ValueError("At least one user must have a target interaction.")
    return {
        "users": count,
        f"precision@{cutoff}": precision_sum / count,
        f"recall@{cutoff}": recall_sum / count,
        f"f1@{cutoff}": f1_sum / count,
        f"ndcg@{cutoff}": ndcg_sum / count,
        f"one_call@{cutoff}": one_call_sum / count,
    }


def evaluate_pnsmf_factors_by_strata(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    targets_by_user: Sequence[np.ndarray],
    excluded_by_user: Sequence[np.ndarray],
    activity_groups: np.ndarray,
    item_popularity_groups: np.ndarray,
    *,
    cutoff: int = 5,
    batch_size: int = 256,
) -> dict[str, object]:
    """Report author-compatible ranking utility by activity and popularity.

    User-level metrics are macro averaged. Popularity and cross-stratum metrics
    are target-event averages, which is explicit because users can have
    different numbers of held-out interactions.
    """

    num_users = user_factors.shape[0]
    num_items = item_factors.shape[0]
    if len(activity_groups) != num_users:
        raise ValueError("activity_groups must align with users.")
    if len(item_popularity_groups) != num_items:
        raise ValueError("item_popularity_groups must align with items.")
    overall = evaluate_pnsmf_factors(
        user_factors,
        item_factors,
        targets_by_user,
        excluded_by_user,
        cutoff=cutoff,
        batch_size=batch_size,
    )
    evaluated_users = [u for u, targets in enumerate(targets_by_user) if len(targets)]
    user_rows: list[dict[str, float | str]] = []
    target_rows: list[dict[str, float | str]] = []
    recommended_items: set[int] = set()
    discounts = 1.0 / np.log2(np.arange(cutoff) + 2.0)

    with torch.no_grad():
        for start in range(0, len(evaluated_users), batch_size):
            batch_users = evaluated_users[start : start + batch_size]
            user_index = torch.as_tensor(
                batch_users, dtype=torch.long, device=user_factors.device
            )
            scores = user_factors.index_select(0, user_index) @ item_factors.T
            for row, user in enumerate(batch_users):
                excluded = torch.as_tensor(
                    excluded_by_user[user], dtype=torch.long, device=user_factors.device
                )
                scores[row, excluded] = -torch.inf
            recommendations = torch.topk(
                scores, k=cutoff, dim=1, largest=True, sorted=True
            ).indices.cpu().numpy()

            for row, user in enumerate(batch_users):
                ranked = recommendations[row]
                recommended_items.update(int(item) for item in ranked)
                targets = np.asarray(targets_by_user[user], dtype=np.int64)
                hits = np.isin(ranked, targets, assume_unique=False)
                hit_count = int(hits.sum())
                precision = hit_count / cutoff
                recall = hit_count / len(targets)
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if precision + recall > 0.0
                    else 0.0
                )
                idcg = float(discounts[: min(len(targets), cutoff)].sum())
                user_rows.append(
                    {
                        "activity": str(activity_groups[user]),
                        f"precision@{cutoff}": precision,
                        f"recall@{cutoff}": recall,
                        f"f1@{cutoff}": f1,
                        f"ndcg@{cutoff}": float(discounts[hits].sum()) / idcg,
                        f"one_call@{cutoff}": float(hit_count > 0),
                    }
                )
                positions = {int(item): position for position, item in enumerate(ranked)}
                for target in targets:
                    position = positions.get(int(target))
                    target_rows.append(
                        {
                            "activity": str(activity_groups[user]),
                            "popularity": str(item_popularity_groups[int(target)]),
                            f"target_hit@{cutoff}": float(position is not None),
                            f"target_discounted_hit@{cutoff}": (
                                float(1.0 / np.log2(position + 2))
                                if position is not None
                                else 0.0
                            ),
                        }
                    )

    user_metric_names = (
        f"precision@{cutoff}",
        f"recall@{cutoff}",
        f"f1@{cutoff}",
        f"ndcg@{cutoff}",
        f"one_call@{cutoff}",
    )

    def aggregate_users(selected: list[dict[str, float | str]]) -> dict[str, float | int]:
        return {
            "users": len(selected),
            **{
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in user_metric_names
            },
        }

    target_metric_names = (f"target_hit@{cutoff}", f"target_discounted_hit@{cutoff}")

    def aggregate_targets(selected: list[dict[str, float | str]]) -> dict[str, float | int]:
        return {
            "target_events": len(selected),
            **{
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in target_metric_names
            },
        }

    activity_values = sorted(set(str(row["activity"]) for row in user_rows))
    popularity_values = sorted(set(str(row["popularity"]) for row in target_rows))
    by_activity = {
        group: aggregate_users(
            [row for row in user_rows if row["activity"] == group]
        )
        for group in activity_values
    }
    by_popularity = {
        group: aggregate_targets(
            [row for row in target_rows if row["popularity"] == group]
        )
        for group in popularity_values
    }
    cross = {}
    for activity in activity_values:
        for popularity in popularity_values:
            selected = [
                row
                for row in target_rows
                if row["activity"] == activity and row["popularity"] == popularity
            ]
            if selected:
                cross[f"{activity}|{popularity}"] = aggregate_targets(selected)
    return {
        "overall": {
            **overall,
            f"unique_recommended_items@{cutoff}": len(recommended_items),
            f"catalog_coverage@{cutoff}": len(recommended_items) / num_items,
        },
        "by_activity": by_activity,
        "by_item_popularity": by_popularity,
        "by_activity_and_item_popularity": cross,
    }
