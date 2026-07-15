import torch

from dflash_offline.masks import create_dflash_dense_mask


def test_dense_mask_matches_scalar_rules():
    anchors = torch.tensor([[2, 5], [1, 4]])
    keep = torch.tensor([[True, True], [True, False]])
    context_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    block_size = 3
    context_length = 6
    actual = create_dflash_dense_mask(
        anchors, keep, context_length, block_size, context_mask
    )[:, 0]

    expected = torch.zeros_like(actual)
    for batch in range(2):
        for query in range(6):
            block = query // block_size
            for kv in range(12):
                context = (
                    kv < context_length
                    and kv < int(anchors[batch, block])
                    and bool(context_mask[batch, kv])
                )
                own_block = kv >= context_length and (
                    (kv - context_length) // block_size == block
                )
                expected[batch, query, kv] = bool(keep[batch, block]) and (
                    context or own_block
                )
    torch.testing.assert_close(actual, expected)
