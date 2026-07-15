#!/usr/bin/env python3
"""Validate offline DFlash checkpoint shapes before distributed training."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_path", required=True)
    parser.add_argument("--draft_config", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 validates every file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.draft_config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected_width = len(config["dflash_config"]["target_layer_ids"]) * config["hidden_size"]
    files = sorted(Path(args.hidden_path).rglob("*.ckpt"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No .ckpt files found")

    lengths = []
    loss_tokens = []
    for path in files:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("input_ids", "hidden_states", "loss_mask"):
            if key not in sample or not isinstance(sample[key], torch.Tensor):
                raise ValueError(f"{path}: missing tensor {key!r}")
        input_ids = sample["input_ids"]
        hidden_states = sample["hidden_states"]
        loss_mask = sample["loss_mask"]
        if input_ids.ndim != 2 or hidden_states.ndim != 3 or loss_mask.ndim != 2:
            raise ValueError(f"{path}: expected input/loss [1,S], hidden [1,S,H]")
        if input_ids.shape[0] != 1 or hidden_states.shape[0] != 1 or loss_mask.shape[0] != 1:
            raise ValueError(f"{path}: each file must contain exactly one sample")
        if not (input_ids.shape[1] == hidden_states.shape[1] == loss_mask.shape[1]):
            raise ValueError(f"{path}: sequence lengths are inconsistent")
        if hidden_states.shape[-1] != expected_width:
            raise ValueError(
                f"{path}: hidden width {hidden_states.shape[-1]} != expected {expected_width}"
            )
        lengths.append(input_ids.shape[1])
        loss_tokens.append(int((loss_mask > 0.5).sum()))

    print(f"validated_files={len(files)}")
    print(f"hidden_width={expected_width}")
    print(f"sequence_length_min/max={min(lengths)}/{max(lengths)}")
    print(f"loss_tokens_min/max={min(loss_tokens)}/{max(loss_tokens)}")


if __name__ == "__main__":
    main()
