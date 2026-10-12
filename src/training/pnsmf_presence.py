"""Reference null extension of the existing direct-gradient round.

Test prototype, not a production private-release interface. Returned local
factors and diagnostics are curator/internal values, never the DP transcript.
No RNG draws precede the underlying Gaussian draw; public slots are never
compacted. An empty history and an absent slot are different states.
"""
from __future__ import annotations
import torch
from src.training.pnsmf_dp import run_uniform_dp_pnsmf_bgd_round


def run_presence_direct_round(
    user_factors: torch.Tensor,
    item_factors: torch.Tensor,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    selected_public_slots: torch.Tensor,
    presence_mask: torch.Tensor,
    **kwargs,
):
    """Keep fixed public N, skip absent local updates and contributions.

    Each selected_public_slot was sampled from the full public [N] universe.
    Presence is private curator input; active IDs/count are not public output.
    Existing rows of train_users for absent slots are ignored by selection.
    Their placeholder factors remain unchanged and are never contributed.
    """
    n = user_factors.shape[0]
    if presence_mask.shape != (n,) or presence_mask.dtype != torch.bool:
        raise ValueError('presence_mask must be bool with shape [public N].')
    if presence_mask.device != user_factors.device:
        raise ValueError('presence_mask and factors must share device.')
    if selected_public_slots.ndim != 1 or selected_public_slots.dtype != torch.long:
        raise ValueError('selected_public_slots must be one-dimensional LongTensor.')
    if selected_public_slots.device != user_factors.device:
        raise ValueError('selected_public_slots and factors must share device.')
    if len(selected_public_slots) and (
        int(selected_public_slots.min()) < 0 or int(selected_public_slots.max()) >= n
    ):
        raise ValueError('selected slot outside fixed public universe.')
    if len(torch.unique(selected_public_slots)) != len(selected_public_slots):
        raise ValueError('selected slots must be unique.')
    if kwargs.get('contribution_space', 'direct_item_gradient') != 'direct_item_gradient':
        raise ValueError('This closure prototype covers only direct_item_gradient.')
    if kwargs.get('contribution_rule', 'clip') != 'clip':
        raise ValueError('This closure prototype covers only ordinary clipping.')
    kwargs['contribution_space'] = 'direct_item_gradient'
    kwargs['contribution_rule'] = 'clip'
    active_selected = selected_public_slots[presence_mask[selected_public_slots]]
    # P retains all N rows. Thus original core computes q*N, not q*active_count.
    return run_uniform_dp_pnsmf_bgd_round(
        user_factors, item_factors, train_users, train_items, active_selected,
        **kwargs,
    )
