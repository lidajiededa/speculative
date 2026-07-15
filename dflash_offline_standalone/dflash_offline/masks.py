"""DFlash attention masks for dense attention and Flex Attention."""

from typing import Optional

import torch


def create_dflash_dense_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    context_length: int,
    block_size: int,
    context_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a boolean visibility mask shaped ``[B, 1, Q, KV]``.

    True means visible. A draft query can see context strictly before its
    anchor and every token in its own draft block. It cannot see its anchor in
    the context region or any other draft block.
    """
    if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
        raise ValueError("anchor_positions and block_keep_mask must both have shape [B, N]")

    batch_size, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    query_length = num_blocks * block_size
    kv_length = context_length + query_length

    query_block_ids = torch.arange(query_length, device=device) // block_size
    query_anchors = anchor_positions[:, query_block_ids]
    valid_queries = block_keep_mask[:, query_block_ids]
    kv_indices = torch.arange(kv_length, device=device).view(1, 1, kv_length)

    is_context = kv_indices < context_length
    context_visible = is_context & (kv_indices < query_anchors.unsqueeze(-1))
    if context_attention_mask is not None:
        if context_attention_mask.shape != (batch_size, context_length):
            raise ValueError(
                "context_attention_mask must have shape "
                f"[{batch_size}, {context_length}]"
            )
        padded_context_mask = torch.zeros(
            (batch_size, kv_length), dtype=torch.bool, device=device
        )
        padded_context_mask[:, :context_length] = context_attention_mask.to(torch.bool)
        context_visible &= padded_context_mask.unsqueeze(1)

    is_draft = kv_indices >= context_length
    kv_block_ids = (kv_indices - context_length).clamp_min(0) // block_size
    own_block_visible = is_draft & (kv_block_ids == query_block_ids.view(1, -1, 1))

    visible = (context_visible | own_block_visible) & valid_queries.unsqueeze(-1)
    return visible.unsqueeze(1)


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    context_length: int,
    block_size: int,
):
    """Return a PyTorch Flex Attention BlockMask."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError("flex_attention is unavailable in this PyTorch build") from exc

    def mask_mod(batch_idx, _head_idx, query_idx, kv_idx):
        query_block = query_idx // block_size
        anchor = anchor_positions[batch_idx, query_block]
        context_visible = (kv_idx < context_length) & (kv_idx < anchor)
        draft_visible = (kv_idx >= context_length) & (
            (kv_idx - context_length) // block_size == query_block
        )
        return (context_visible | draft_visible) & block_keep_mask[batch_idx, query_block]

    batch_size, num_blocks = anchor_positions.shape
    query_length = num_blocks * block_size
    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_length,
        KV_LEN=context_length + query_length,
        device=anchor_positions.device,
    )
