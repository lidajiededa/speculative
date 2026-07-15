"""Load only the frozen target embedding table and language-model head."""

import gc
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig


class TargetEmbeddingsAndHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        embed_key: str = "model.embed_tokens.weight",
        lm_head_key: str = "lm_head.weight",
        trust_remote_code: bool = True,
    ) -> "TargetEmbeddingsAndHead":
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        instance = cls(config)
        instance._load_weights(
            model_path=model_path,
            embed_key=embed_key,
            lm_head_key=lm_head_key,
            tie_weights=bool(getattr(config, "tie_word_embeddings", False)),
        )
        instance.to(device=device, dtype=dtype)
        instance.eval().requires_grad_(False)
        return instance

    def _load_weights(
        self,
        model_path: str,
        embed_key: str,
        lm_head_key: str,
        tie_weights: bool,
    ) -> None:
        model_path = str(Path(model_path))
        index_files = sorted(glob.glob(os.path.join(model_path, "*.index.json")))
        key_to_file = {}
        if index_files:
            with open(index_files[0], "r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
            if embed_key not in weight_map:
                raise KeyError(f"Embedding key {embed_key!r} is absent from {index_files[0]}")
            key_to_file[embed_key] = weight_map[embed_key]
            if not tie_weights:
                if lm_head_key not in weight_map:
                    raise KeyError(f"LM-head key {lm_head_key!r} is absent from {index_files[0]}")
                key_to_file[lm_head_key] = weight_map[lm_head_key]
        else:
            candidates = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
            candidates += sorted(glob.glob(os.path.join(model_path, "*.bin")))
            if not candidates:
                raise FileNotFoundError(f"No safetensors or bin checkpoint found in {model_path}")
            filename = os.path.basename(candidates[0])
            key_to_file[embed_key] = filename
            if not tie_weights:
                key_to_file[lm_head_key] = filename

        file_to_keys = defaultdict(list)
        for key, filename in key_to_file.items():
            file_to_keys[os.path.join(model_path, filename)].append(key)

        loaded = set()
        for file_path, keys in file_to_keys.items():
            tensors = self._read_selected(file_path, keys)
            for key in keys:
                if key not in tensors:
                    raise KeyError(f"Weight {key!r} is absent from {file_path}")
                destination = (
                    self.embed_tokens.weight if key == embed_key else self.lm_head.weight
                )
                if tensors[key].shape != destination.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: {tuple(tensors[key].shape)} != "
                        f"{tuple(destination.shape)}"
                    )
                destination.data.copy_(tensors[key])
                loaded.add(key)
            del tensors
            gc.collect()

        if embed_key not in loaded:
            raise RuntimeError("Embedding weight was not loaded")
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight
        elif lm_head_key not in loaded:
            raise RuntimeError("LM-head weight was not loaded")

    @staticmethod
    def _read_selected(file_path: str, keys: list[str]) -> dict[str, torch.Tensor]:
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                return {key: handle.get_tensor(key) for key in keys if key in handle.keys()}
        try:
            state = torch.load(file_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(file_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return {key: state[key] for key in keys if key in state}
