import torch

from dflash_offline.objective import create_targets_and_weights, sample_anchor_positions


def test_targets_skip_anchor_and_apply_decay():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    anchors = torch.tensor([[1]])
    keep = torch.tensor([[True]])
    targets, weights, eval_weights = create_targets_and_weights(
        input_ids, loss_mask, anchors, keep, block_size=4, loss_decay_gamma=2.0
    )
    assert targets.tolist() == [11, 12, 13, 14]
    torch.testing.assert_close(eval_weights, torch.tensor([0.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(
        weights,
        torch.tensor([0.0, 1.0, torch.exp(torch.tensor(-0.5)), torch.exp(torch.tensor(-1.0))]),
    )


def test_anchor_sampling_keeps_only_each_rows_valid_positions():
    torch.manual_seed(7)
    loss_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.float32
    )
    anchors, keep = sample_anchor_positions(
        loss_mask, sequence_length=6, block_size=2, num_anchors=3
    )
    assert anchors[1, 0].item() == 2
    assert keep[1].tolist() == [True, False, False]
    assert anchors[1, 1:].tolist() == [0, 0]
