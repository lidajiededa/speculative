"""Pure tensor helpers for the DFlash offline objective."""

from typing import Optional, Tuple

import torch


def sample_anchor_positions(
    loss_mask: torch.Tensor,
    sequence_length: int,
    block_size: int,
    num_anchors: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Sample valid anchor positions independently for every batch item."""
    device = loss_mask.device
    batch_size = loss_mask.shape[0]
    max_anchor = max(sequence_length - block_size, 0)
    valid = loss_mask[:, : max_anchor + 1] > 0.5
    valid_counts = valid.sum(dim=1)
    max_valid = int(valid_counts.max().item())
    if max_valid <= 1:
        return None, None

    sampled_count = min(num_anchors, max_valid - 1)
    random_values = torch.rand(batch_size, max_anchor + 1, device=device)
    random_values.masked_fill_(~valid, 2.0)
    sorted_random_indices = random_values.argsort(dim=1)
    all_indices = torch.arange(max_anchor + 1, device=device).expand(batch_size, -1)
    masked_indices = torch.where(
        valid,
        all_indices,
        torch.full_like(all_indices, sequence_length + 1),
    )
    sampled_indices = torch.gather(masked_indices, 1, sorted_random_indices)
    sampled_indices = sampled_indices[:, :sampled_count].sort(dim=1).values

    keep_mask = torch.arange(sampled_count, device=device).unsqueeze(0) < (
        valid_counts.clamp(max=sampled_count).unsqueeze(1)
    )
    anchors = torch.where(keep_mask, sampled_indices, torch.zeros_like(sampled_indices))
    return anchors, keep_mask


def create_position_ids(
    anchor_positions: torch.Tensor,
    sequence_length: int,
    block_size: int,
) -> torch.Tensor:
    batch_size = anchor_positions.shape[0]
    device = anchor_positions.device
    context = torch.arange(sequence_length, device=device).expand(batch_size, -1)
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    draft = (anchor_positions.unsqueeze(-1) + offsets).flatten(1)
    return torch.cat((context, draft), dim=1)


def create_targets_and_weights(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
    loss_decay_gamma: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create flattened target IDs, decayed loss weights and binary eval weights."""
    sequence_length = input_ids.shape[1]
    device = input_ids.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    label_indices = anchor_positions.unsqueeze(-1) + offsets
    in_bounds = label_indices < sequence_length
    safe_indices = label_indices.clamp(max=sequence_length - 1)

    expanded_ids = input_ids.unsqueeze(1).expand(-1, anchor_positions.shape[1], -1)
    targets = torch.gather(expanded_ids, dim=2, index=safe_indices)
    weights = block_keep_mask.unsqueeze(-1).expand_as(targets).float()
    weights *= in_bounds.float()
    weights *= (offsets > 0).float()
    expanded_loss_mask = loss_mask.unsqueeze(1).expand_as(expanded_ids)
    weights *= torch.gather(expanded_loss_mask, dim=2, index=safe_indices)
    eval_weights = weights.flatten()

    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        decay = torch.exp(
            -(offsets - 1).clamp(min=0).float() / float(loss_decay_gamma)
        )
        weights = weights * decay
    return targets.flatten(), weights.flatten(), eval_weights
