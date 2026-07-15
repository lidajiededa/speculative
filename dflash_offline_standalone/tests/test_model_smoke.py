import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from dflash_offline.masks import create_dflash_dense_mask
from dflash_offline.modeling_qwen3_dflash import QwenDFlashDraftModel
from dflash_offline.objective import create_position_ids


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_tiny_model_forward_backward(backend):
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        layer_types=["full_attention", "full_attention"],
    )
    config.architectures = ["QwenDFlashDraftModel"]
    config.block_size = 4
    config.attention_backend = backend
    config.dflash_config = {"mask_token_id": 63, "target_layer_ids": [0, 1]}
    model = QwenDFlashDraftModel(config)

    anchors = torch.tensor([[2, 6]])
    keep = torch.tensor([[True, True]])
    sequence_length = 12
    mask = create_dflash_dense_mask(anchors, keep, sequence_length, 4)
    positions = create_position_ids(anchors, sequence_length, 4)
    output = model(
        position_ids=positions,
        attention_mask=mask,
        noise_embedding=torch.randn(1, 8, 32),
        target_hidden=torch.randn(1, sequence_length, 64),
    )
    assert output.shape == (1, 8, 32)
    output.square().mean().backward()
    assert model.fc.weight.grad is not None
