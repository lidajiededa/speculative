#!/usr/bin/env python3
"""Standalone DFlash hidden-state generator for Qwen3-30B-A3B on NPU.

This script intentionally does not import AngelSlim. It reads JSONL chat data,
renders Qwen-style chat text, builds input/loss masks, runs a Hugging Face target
model with output_hidden_states=True, extracts selected target layers, and saves
one .ckpt file per sample.

Typical 2-NPU instance:

    ASCEND_RT_VISIBLE_DEVICES=0,1 python tools/standalone_generate_dflash_hidden_states_npu.py \
      --model-path /models/Qwen3-30B-A3B \
      --input-jsonl /data/qwen3_30b_train_shards/shard_00.jsonl \
      --output-dir /data/qwen3_30b_dflash_hidden_cache/instance_0 \
      --device-map auto \
      --attn-implementation eager \
      --target-layer-ids 1,12,23,34,45 \
      --model-max-length 4096
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, "
    "sexist, toxic, dangerous, or illegal content. Please ensure that "
    "your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. If you don't "
    "know the answer to a question, please don't share false information."
)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DFlash target hidden_states without AngelSlim dependencies."
    )
    parser.add_argument("--model-path", required=True, help="HF model path, e.g. Qwen3-30B-A3B")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL conversation file")
    parser.add_argument("--output-dir", required=True, help="Output directory for .ckpt files")
    parser.add_argument(
        "--target-layer-ids",
        default="1,12,23,34,45",
        help="Comma-separated decoder layer ids. Qwen3-30B-A3B default gives 2048*5 hidden dim.",
    )
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--sample-num", type=int, default=None)
    parser.add_argument("--min-loss-tokens", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0, help="Skip input lines before this index")
    parser.add_argument("--end-index", type=int, default=None, help="Stop before this input line index")

    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--device-map",
        default="auto",
        help='HF device_map. Use "auto" for 2-NPU model dispatch, or a single device like npu:0.',
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help='Optional max_memory, e.g. "0:60GiB,1:60GiB" for device_map=auto.',
    )
    parser.add_argument("--attn-implementation", default="eager", help="eager or sdpa on NPU")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--concat-device",
        default="cpu",
        choices=["cpu", "first_hidden_device"],
        help="Where to concatenate selected hidden layers before saving.",
    )

    parser.add_argument("--user-header", default="<|im_start|>user\n")
    parser.add_argument("--assistant-header", default="<|im_start|>assistant\n")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_target_layer_ids(text: str) -> List[int]:
    ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not ids:
        raise ValueError("--target-layer-ids must not be empty")
    if min(ids) < 0:
        raise ValueError("--target-layer-ids must be non-negative decoder layer ids")
    return ids


def parse_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_max_memory(text: Optional[str]) -> Optional[Dict[Any, str]]:
    if not text:
        return None
    result: Dict[Any, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        result[int(key) if key.isdigit() else key] = value
    return result


def is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def set_default_device() -> None:
    if is_npu_available():
        torch.npu.set_device(0)
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("text"):
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def normalize_turn(turn: Dict[str, Any]) -> Optional[Dict[str, str]]:
    raw_role = turn.get("role", turn.get("from", ""))
    role = ROLE_MAP.get(str(raw_role).lower())
    content = normalize_content(turn.get("content", turn.get("value", "")))
    if not role or not content:
        return None
    return {"role": role, "content": content}


def get_conversation(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("messages", "conversations", "conversation"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("record must contain a messages/conversations/conversation list")


def build_messages(record: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    source = [x for x in (normalize_turn(t) for t in get_conversation(record)) if x is not None]
    if not source:
        return []

    if source[0]["role"] == "system":
        messages = [source[0]]
        source = source[1:]
    else:
        messages = [{"role": "system", "content": system_prompt}]

    if source and source[0]["role"] != "user":
        source = source[1:]

    expected = ("user", "assistant")
    for idx, turn in enumerate(source):
        if turn["role"] != expected[idx % 2]:
            break
        messages.append(turn)

    return messages if len(messages) > 1 else []


def assistant_spans(rendered_chat: str, assistant_header: str, user_header: str) -> List[Tuple[int, int]]:
    pattern = re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + r"|$)"
    spans = []
    for match in re.finditer(pattern, rendered_chat, re.DOTALL):
        spans.append((match.start(1), match.end(1)))
    return spans


def loss_mask_from_offsets(
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    loss_mask = torch.zeros(len(offsets), dtype=torch.long)
    for idx, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        for span_start, span_end in spans:
            if token_start < span_end and token_end > span_start:
                loss_mask[idx] = 1
                break
    return loss_mask


def loss_mask_from_token_lengths(tokenizer, rendered_chat: str, spans: Sequence[Tuple[int, int]]) -> torch.Tensor:
    full_ids = tokenizer(rendered_chat, add_special_tokens=False).input_ids
    loss_mask = torch.zeros(len(full_ids), dtype=torch.long)
    for span_start, span_end in spans:
        prefix_len = len(tokenizer(rendered_chat[:span_start], add_special_tokens=False).input_ids)
        span_len = len(tokenizer(rendered_chat[span_start:span_end], add_special_tokens=False).input_ids)
        loss_mask[prefix_len : prefix_len + span_len] = 1
    return loss_mask


def tokenize_record(
    record: Dict[str, Any],
    tokenizer,
    model_max_length: int,
    user_header: str,
    assistant_header: str,
    system_prompt: str,
) -> Optional[Dict[str, torch.Tensor]]:
    messages = build_messages(record, system_prompt)
    if not messages:
        return None

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    spans = assistant_spans(rendered, assistant_header=assistant_header, user_header=user_header)
    if not spans:
        return None

    is_fast = bool(getattr(tokenizer, "is_fast", False))
    if is_fast:
        encoding = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            max_length=model_max_length,
            truncation=True,
            padding=False,
        )
        input_ids = torch.tensor(encoding.input_ids, dtype=torch.long)
        loss_mask = loss_mask_from_offsets(encoding.offset_mapping, spans)
    else:
        encoding = tokenizer(
            rendered,
            add_special_tokens=False,
            max_length=model_max_length,
            truncation=True,
            padding=False,
        )
        input_ids = torch.tensor(encoding.input_ids, dtype=torch.long)
        loss_mask = loss_mask_from_token_lengths(tokenizer, rendered, spans)[: input_ids.numel()]

    if input_ids.numel() == 0:
        return None
    if loss_mask.numel() < input_ids.numel():
        loss_mask = torch.nn.functional.pad(loss_mask, (0, input_ids.numel() - loss_mask.numel()))
    elif loss_mask.numel() > input_ids.numel():
        loss_mask = loss_mask[: input_ids.numel()]

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids[None, :],
        "attention_mask": attention_mask[None, :],
        "loss_mask": loss_mask[None, :],
    }


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def get_target_input_device(model) -> torch.device:
    embeddings = model.get_input_embeddings()
    return next(embeddings.parameters()).device


def run_base_model(model, input_ids: torch.Tensor):
    base_model = getattr(model, "model", None)
    if base_model is None:
        return model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return base_model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )


def extract_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    target_layer_ids: Sequence[int],
    concat_device: str,
) -> torch.Tensor:
    embed_offset = 1
    max_layer = len(hidden_states) - 2
    bad_layers = [x for x in target_layer_ids if x > max_layer]
    if bad_layers:
        raise ValueError(
            f"target_layer_ids {bad_layers} out of range; model has decoder layers 0..{max_layer}"
        )

    selected = [hidden_states[layer_id + embed_offset] for layer_id in target_layer_ids]
    if concat_device == "cpu":
        selected = [h.detach().to("cpu", dtype=torch.bfloat16) for h in selected]
    else:
        device = selected[0].device
        selected = [h.detach().to(device) for h in selected]
    return torch.cat(selected, dim=-1)


def save_ckpt(
    output_dir: Path,
    saved_idx: int,
    source_idx: int,
    shard_size: int,
    tensors: Dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    source_id: Optional[Any],
    overwrite: bool,
) -> Path:
    if shard_size > 0:
        save_dir = output_dir / f"shard_{saved_idx // shard_size:05d}"
    else:
        save_dir = output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f"sample_{saved_idx:08d}.ckpt"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    payload = {
        "input_ids": tensors["input_ids"].cpu(),
        "attention_mask": tensors["attention_mask"].cpu(),
        "loss_mask": tensors["loss_mask"].cpu(),
        "hidden_states": hidden_states.cpu().to(torch.bfloat16),
        "source_index": torch.tensor(source_idx, dtype=torch.long),
    }
    if source_id is not None:
        payload["source_id"] = str(source_id)

    torch.save(payload, path)
    return path


def maybe_empty_cache() -> None:
    if is_npu_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_device_map(model) -> None:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is None:
        print("[WARN] model has no hf_device_map; device_map may not have taken effect", flush=True)
        return
    print("[INFO] hf_device_map:", hf_device_map, flush=True)


def main() -> None:
    args = parse_args()
    set_default_device()

    target_layer_ids = parse_target_layer_ids(args.target_layer_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.torch_dtype)
    max_memory = parse_max_memory(args.max_memory)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    if max_memory:
        model_kwargs["max_memory"] = max_memory

    print(f"[INFO] loading tokenizer: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    print(f"[INFO] loading model: {args.model_path}", flush=True)
    print(f"[INFO] model kwargs: {model_kwargs}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print_device_map(model)
    input_device = get_target_input_device(model)
    print(f"[INFO] target input device: {input_device}", flush=True)
    print(f"[INFO] target layer ids: {target_layer_ids}", flush=True)

    saved = 0
    skipped = 0
    started = time.time()
    input_path = Path(args.input_jsonl)

    for source_idx, record in iter_jsonl(input_path):
        if source_idx < args.start_index:
            continue
        if args.end_index is not None and source_idx >= args.end_index:
            break
        if args.sample_num is not None and saved >= args.sample_num:
            break

        try:
            tensors = tokenize_record(
                record,
                tokenizer=tokenizer,
                model_max_length=args.model_max_length,
                user_header=args.user_header,
                assistant_header=args.assistant_header,
                system_prompt=args.system_prompt,
            )
            if tensors is None:
                skipped += 1
                continue
            if int(tensors["loss_mask"].sum().item()) < args.min_loss_tokens:
                skipped += 1
                continue

            input_ids = tensors["input_ids"].to(input_device)
            with torch.no_grad():
                outputs = run_base_model(model, input_ids=input_ids)
            hidden = extract_hidden_states(
                outputs.hidden_states,
                target_layer_ids=target_layer_ids,
                concat_device=args.concat_device,
            )

            path = save_ckpt(
                output_dir=output_dir,
                saved_idx=saved,
                source_idx=source_idx,
                shard_size=args.shard_size,
                tensors=tensors,
                hidden_states=hidden,
                source_id=record.get("id"),
                overwrite=args.overwrite,
            )
            saved += 1

            del outputs, hidden, input_ids
            maybe_empty_cache()

            if saved == 1 or saved % args.log_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[INFO] saved={saved} skipped={skipped} "
                    f"source_idx={source_idx} speed={saved / elapsed:.3f}/s last={path}",
                    flush=True,
                )

        except Exception as exc:
            skipped += 1
            print(f"[WARN] failed source_idx={source_idx}: {exc}", file=sys.stderr, flush=True)
            maybe_empty_cache()

    print(
        f"[INFO] done. saved={saved}, skipped={skipped}, output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
