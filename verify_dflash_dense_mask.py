#!/usr/bin/env python3
"""Verify dense DFlash attention mask correctness.

This script is intentionally standalone: it does not import AngelSlim trainer
modules or Transformers. It verifies that a dense additive Tensor mask for
eager/sdpa follows exactly the same DFlash visibility rules as the original
FlexAttention BlockMask predicate:

KV layout: [context tokens S | draft block tokens N * block_size]
Q layout:  [draft block tokens N * block_size]

Rules:
  1. Each draft block sees context tokens strictly before its anchor.
  2. Each draft block sees its own draft tokens.
  3. Different draft blocks are invisible to each other.
  4. Invalid blocks see nothing.

Usage:
  python tools/verify_dflash_dense_mask.py
  python tools/verify_dflash_dense_mask.py --device npu --cases 200
  python tools/verify_dflash_dense_mask.py --device cuda --dtype float16
"""

import argparse
import math
import random
from typing import Tuple

import torch


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if device_arg == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("Requested --device npu, but torch.npu is not available.")
        return torch.device("npu:0")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device("cpu")


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def create_dflash_dense_attention_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_idx = torch.arange(Q_LEN, device=device)
    kv_idx = torch.arange(KV_LEN, device=device)

    q_block_id = q_idx // block_size
    anchors_for_q = anchor_positions[:, q_block_id]
    valid_for_q = block_keep_mask[:, q_block_id]

    q_block_id_3d = q_block_id.view(1, Q_LEN, 1)
    kv_idx_3d = kv_idx.view(1, 1, KV_LEN)

    is_context = kv_idx_3d < S
    mask_context = is_context & (kv_idx_3d < anchors_for_q.unsqueeze(-1))

    is_draft = kv_idx_3d >= S
    kv_block_id = (kv_idx_3d - S) // block_size
    mask_draft = is_draft & (kv_block_id == q_block_id_3d)

    visible = (mask_context | mask_draft) & valid_for_q.unsqueeze(-1)
    visible = visible.unsqueeze(1)

    mask = torch.full(
        (B, 1, Q_LEN, KV_LEN),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return mask.masked_fill(visible, 0.0)


def reference_visible_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
) -> torch.Tensor:
    """Slow reference implementation using explicit Python loops."""
    anchors_cpu = anchor_positions.cpu()
    keep_cpu = block_keep_mask.cpu()
    B, N = anchors_cpu.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size
    visible = torch.zeros((B, 1, Q_LEN, KV_LEN), dtype=torch.bool)

    for b in range(B):
        for q_idx in range(Q_LEN):
            q_block_id = q_idx // block_size
            anchor_pos = int(anchors_cpu[b, q_block_id].item())
            is_valid_block = bool(keep_cpu[b, q_block_id].item())

            for kv_idx in range(KV_LEN):
                is_context = kv_idx < S
                mask_context = is_context and kv_idx < anchor_pos

                is_draft = kv_idx >= S
                kv_block_id = (kv_idx - S) // block_size if is_draft else -1
                mask_draft = is_draft and q_block_id == kv_block_id

                visible[b, 0, q_idx, kv_idx] = (
                    (mask_context or mask_draft) and is_valid_block
                )

    return visible


def assert_mask_matches_reference(
    dense_mask: torch.Tensor,
    ref_visible: torch.Tensor,
) -> None:
    dense_cpu = dense_mask.detach().cpu()
    ref_visible = ref_visible.cpu()

    dense_visible = dense_cpu == 0
    if not torch.equal(dense_visible, ref_visible):
        mismatch = torch.nonzero(dense_visible != ref_visible, as_tuple=False)[0]
        idx = tuple(int(x) for x in mismatch.tolist())
        raise AssertionError(
            f"Visibility mismatch at {idx}: "
            f"dense_visible={bool(dense_visible[idx])}, ref_visible={bool(ref_visible[idx])}"
        )

    masked_values = dense_cpu[~ref_visible]
    if masked_values.numel() > 0:
        expected = torch.finfo(dense_cpu.dtype).min
        if not torch.all(masked_values == expected):
            raise AssertionError("Masked positions are not all torch.finfo(dtype).min.")


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_mask: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores + dense_mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref_visible: torch.Tensor,
) -> torch.Tensor:
    """Reference attention that explicitly selects visible KV positions.

    Fully masked rows are left as zeros. The real DFlash loss later masks out
    invalid blocks, so only valid rows are compared by the caller.
    """
    B, H, Q_LEN, D = q.shape
    output = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    for b in range(B):
        for h in range(H):
            for qi in range(Q_LEN):
                visible_idx = torch.nonzero(ref_visible[b, 0, qi], as_tuple=False).view(-1)
                if visible_idx.numel() == 0:
                    continue
                qb = q[b, h, qi].float()
                kb = k[b, h, visible_idx].float()
                vb = v[b, h, visible_idx].float()
                scores = torch.matmul(kb, qb) * scale
                probs = torch.softmax(scores, dim=-1)
                output[b, h, qi] = torch.matmul(probs, vb).to(output.dtype)

    return output


def random_case(
    device: torch.device,
    max_b: int,
    max_s: int,
    max_n: int,
    max_block: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    B = random.randint(1, max_b)
    S = random.randint(2, max_s)
    N = random.randint(1, max_n)
    block_size = random.randint(1, max_block)

    anchors = []
    keep = []
    for _ in range(B):
        row_anchors = []
        row_keep = []
        for _ in range(N):
            valid = random.random() > 0.2
            row_keep.append(valid)
            row_anchors.append(random.randint(0, S - 1))
        anchors.append(row_anchors)
        keep.append(row_keep)

    anchor_positions = torch.tensor(anchors, dtype=torch.long, device=device)
    block_keep_mask = torch.tensor(keep, dtype=torch.bool, device=device)
    return anchor_positions, block_keep_mask, S, block_size


def run_one_case(
    case_id: int,
    device: torch.device,
    dtype: torch.dtype,
    max_b: int,
    max_s: int,
    max_n: int,
    max_block: int,
    check_attention: bool,
) -> None:
    anchor_positions, block_keep_mask, S, block_size = random_case(
        device=device,
        max_b=max_b,
        max_s=max_s,
        max_n=max_n,
        max_block=max_block,
    )
    dense_mask = create_dflash_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        S=S,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )
    ref_visible = reference_visible_mask(anchor_positions, block_keep_mask, S, block_size)

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    expected_shape = (B, 1, Q_LEN, KV_LEN)
    if tuple(dense_mask.shape) != expected_shape:
        raise AssertionError(
            f"Case {case_id}: shape mismatch, got {tuple(dense_mask.shape)}, "
            f"expected {expected_shape}"
        )
    if dense_mask.dtype != dtype:
        raise AssertionError(f"Case {case_id}: dtype mismatch: {dense_mask.dtype} != {dtype}")

    assert_mask_matches_reference(dense_mask, ref_visible)

    # Proves eager-style addition works. This is the exact class of operation
    # that failed when a BlockMask was passed to eager_attention_forward.
    scores = torch.zeros((B, 2, Q_LEN, KV_LEN), dtype=dtype, device=device)
    _ = scores + dense_mask

    if check_attention:
        # Keep dimensions small; this is a correctness check, not a benchmark.
        H, D = 2, 8
        q = torch.randn((B, H, Q_LEN, D), dtype=dtype, device=device)
        k = torch.randn((B, H, KV_LEN, D), dtype=dtype, device=device)
        v = torch.randn((B, H, KV_LEN, D), dtype=dtype, device=device)

        out_dense = dense_attention(q, k, v, dense_mask)
        out_ref = reference_attention(q.cpu(), k.cpu(), v.cpu(), ref_visible).to(device)

        valid_q = ref_visible.any(dim=-1).expand(B, H, Q_LEN)
        if valid_q.any():
            diff = (out_dense[valid_q] - out_ref[valid_q]).abs().max().float().item()
            tolerance = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
            if diff > tolerance:
                raise AssertionError(
                    f"Case {case_id}: attention output mismatch, max diff {diff} > {tolerance}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DFlash dense attention mask.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "npu"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--max-b", type=int, default=3)
    parser.add_argument("--max-s", type=int, default=24)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--max-block", type=int, default=8)
    parser.add_argument(
        "--skip-attention-check",
        action="store_true",
        help="Only compare mask visibility, skip manual attention output comparison.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        print("[WARN] float16 on CPU may be slow; continuing.")

    for case_id in range(args.cases):
        run_one_case(
            case_id=case_id,
            device=device,
            dtype=dtype,
            max_b=args.max_b,
            max_s=args.max_s,
            max_n=args.max_n,
            max_block=args.max_block,
            check_attention=not args.skip_attention_check,
        )

    print(
        "OK: dense DFlash mask matches reference rules and eager-style Tensor addition "
        f"for {args.cases} randomized cases on {device} with dtype={dtype}."
    )


if __name__ == "__main__":
    main()
