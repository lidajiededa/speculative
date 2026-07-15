"""Minimal Qwen3 DFlash draft model used by offline training.

Derived from Tencent AngelSlim's Apache-2.0-licensed Qwen DFlash model, with
the attention dispatch made explicit so dense masks work on Ascend NPU.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_length = query.size(-2)
    query = (query * cos[..., -query_length:, :]) + (
        rotate_half(query) * sin[..., -query_length:, :]
    )
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, length, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * repeats, length, head_dim)


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attention_mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        )
        return torch.matmul(probabilities, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        batch_size, query_length = hidden_states.shape[:2]
        context_length = target_hidden.shape[1]

        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch_size, query_length, -1, self.head_dim
            )
        ).transpose(1, 2)
        key = torch.cat((self.k_proj(target_hidden), self.k_proj(hidden_states)), dim=1)
        value = torch.cat((self.v_proj(target_hidden), self.v_proj(hidden_states)), dim=1)
        key = self.k_norm(
            key.view(batch_size, context_length + query_length, -1, self.head_dim)
        ).transpose(1, 2)
        value = value.view(
            batch_size, context_length + query_length, -1, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = _apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )

        key = _repeat_kv(key, self.num_kv_groups)
        value = _repeat_kv(value, self.num_kv_groups)
        backend = getattr(self.config, "attention_backend", "sdpa")
        if backend == "eager":
            output = self._eager_attention(query, key, value, attention_mask)
        elif backend == "sdpa":
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )
        elif backend == "flex_attention":
            try:
                from torch.nn.attention.flex_attention import flex_attention
            except ImportError as exc:
                raise RuntimeError("flex_attention is unavailable") from exc
            output = flex_attention(
                query,
                key,
                value,
                block_mask=attention_mask,
                scale=self.scaling,
            )
        else:
            raise ValueError(f"Unsupported attention backend: {backend}")

        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        return self.o_proj(output)


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class QwenDFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            Qwen3DFlashDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get("target_layer_ids", [])
        if not self.target_layer_ids:
            raise ValueError("dflash_config.target_layer_ids must not be empty")
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id")
        if self.mask_token_id is None:
            raise ValueError("dflash_config.mask_token_id must be set")
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del use_cache
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values,
                **kwargs,
            )
        return self.norm(hidden_states)
