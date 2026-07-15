"""Offline hidden-state checkpoint dataset and collator."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = ("input_ids", "hidden_states", "loss_mask")


def _ensure_batched(name: str, tensor: torch.Tensor) -> torch.Tensor:
    expected_ndim = 3 if name == "hidden_states" else 2
    if tensor.ndim == expected_ndim - 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != expected_ndim or tensor.shape[0] != 1:
        raise ValueError(
            f"{name} must have shape [1, S"
            + (", H]" if name == "hidden_states" else "]")
            + f"; got {tuple(tensor.shape)}"
        )
    return tensor


class OfflineDFlashDataset(Dataset):
    """Recursively load AngelSlim-compatible DFlash ``.ckpt`` samples."""

    def __init__(
        self,
        data_dir: str | Path,
        max_length: Optional[int] = None,
        cache_in_memory: bool = False,
        expected_hidden_size: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        self.ckpt_files = sorted(self.data_dir.rglob("*.ckpt"))
        if not self.ckpt_files:
            raise ValueError(f"No .ckpt files found recursively under {self.data_dir}")
        self.max_length = max_length
        self.expected_hidden_size = expected_hidden_size
        self.cached_data = None
        if cache_in_memory:
            self.cached_data = [self._load(path) for path in self.ckpt_files]

    def __len__(self) -> int:
        return len(self.ckpt_files)

    def _load(self, path: Path) -> Dict[str, torch.Tensor]:
        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc
        if not isinstance(sample, dict):
            raise TypeError(f"{path} must contain a dictionary")
        missing = [key for key in REQUIRED_KEYS if key not in sample]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")

        result: Dict[str, torch.Tensor] = {}
        for key in REQUIRED_KEYS:
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{path}: {key} is not a torch.Tensor")
            result[key] = _ensure_batched(key, value)

        attention_mask = sample.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(result["input_ids"])
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(f"{path}: attention_mask is not a torch.Tensor")
        result["attention_mask"] = _ensure_batched("attention_mask", attention_mask)

        sequence_lengths = {key: value.shape[1] for key, value in result.items()}
        if len(set(sequence_lengths.values())) != 1:
            raise ValueError(f"{path}: inconsistent sequence lengths: {sequence_lengths}")
        if self.expected_hidden_size is not None:
            actual = result["hidden_states"].shape[-1]
            if actual != self.expected_hidden_size:
                raise ValueError(
                    f"{path}: hidden_states[-1]={actual}, expected {self.expected_hidden_size}"
                )

        if self.max_length is not None:
            for key in result:
                result[key] = result[key][:, : self.max_length]
        result["input_ids"] = result["input_ids"].long()
        result["attention_mask"] = result["attention_mask"].long()
        result["loss_mask"] = result["loss_mask"].float()
        return result

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.cached_data is not None:
            return self.cached_data[index]
        return self._load(self.ckpt_files[index])


def _pad_sequence(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[1] == length:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[1] = length - tensor.shape[1]
    padding = torch.zeros(pad_shape, dtype=tensor.dtype)
    return torch.cat((tensor, padding), dim=1)


class DFlashDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        keys = ("input_ids", "attention_mask", "loss_mask", "hidden_states")
        return {
            key: torch.cat([_pad_sequence(item[key], max_length) for item in features], dim=0)
            for key in keys
        }
