#!/usr/bin/env python3
"""Verify DFlash 2-D grid input-expand kernel on Ascend NPU.

This script is intentionally standalone:
it defines both the current single-grid kernel and the proposed 2-D grid
kernel, generates randomized test cases, runs both kernels on NPU, and compares
their outputs with a CPU reference implementation.

Run it inside an environment where torch-npu, vLLM, vLLM-Ascend and Triton for
Ascend are available.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass

import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "torch_npu is required. Please run this script in an Ascend NPU "
        "environment with torch-npu installed."
    ) from exc

from vllm.triton_utils import tl, triton


@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(
    next_token_ids_ptr,
    target_positions_ptr,
    out_input_ids_ptr,
    out_context_positions_ptr,
    out_query_positions_ptr,
    out_context_slot_mapping_ptr,
    out_query_slot_mapping_ptr,
    out_token_indices_ptr,
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    num_rejected_tokens_ptr,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    batch_size,
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    for req_idx in range(0, batch_size):
        ctx_start = tl.load(query_start_loc_ptr + req_idx)
        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
        num_ctx = ctx_end - ctx_start

        for j in range(0, num_ctx):
            ctx_pos_idx = ctx_start + j
            pos = tl.load(target_positions_ptr + ctx_pos_idx)
            tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

            block_num = pos // block_size
            block_id = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_num
            ).to(tl.int64)
            slot = block_id * block_size + (pos % block_size)
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

        if HAS_NUM_REJECTED:
            num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
            valid_ctx_end = ctx_end - num_rejected
        else:
            valid_ctx_end = ctx_end

        last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)

        for q_idx in range(0, num_query_per_req):
            query_pos = last_pos + 1 + q_idx
            query_out_idx = req_idx * num_query_per_req + q_idx

            tl.store(out_query_positions_ptr + query_out_idx, query_pos)

            block_num_q = query_pos // block_size
            block_id_q = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_num_q
            ).to(tl.int64)
            slot_q = block_id_q * block_size + (query_pos % block_size)
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

            if q_idx == 0:
                bonus_token = tl.load(next_token_ids_ptr + req_idx)
                tl.store(out_input_ids_ptr + query_out_idx, bonus_token)
            else:
                tl.store(
                    out_input_ids_ptr + query_out_idx,
                    parallel_drafting_token_id,
                )

                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)


@triton.jit
def copy_and_expand_dflash_inputs_kernel_2d(
    next_token_ids_ptr,
    target_positions_ptr,
    out_input_ids_ptr,
    out_context_positions_ptr,
    out_query_positions_ptr,
    out_context_slot_mapping_ptr,
    out_query_slot_mapping_ptr,
    out_token_indices_ptr,
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    num_rejected_tokens_ptr,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    req_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    total_tokens = num_ctx + num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = j < total_tokens
    is_ctx = j < num_ctx
    is_query = (~is_ctx) & in_bounds
    query_off = j - num_ctx

    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)

    if HAS_NUM_REJECTED:
        num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
    else:
        valid_ctx_end = ctx_end
    last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_pos = last_pos + 1 + query_off

    positions = tl.where(is_ctx, ctx_pos, query_pos)

    ctx_pos_out = ctx_start + j
    query_out = req_idx * num_query_per_req + query_off

    tl.store(out_context_positions_ptr + ctx_pos_out, ctx_pos, mask=is_ctx)
    tl.store(out_query_positions_ptr + query_out, query_pos, mask=is_query)

    block_num = positions // block_size
    block_num = tl.minimum(block_num, block_table_stride - 1)
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_num,
        mask=in_bounds,
        other=0,
    ).to(tl.int64)
    slot = block_id * block_size + (positions % block_size)

    tl.store(out_context_slot_mapping_ptr + ctx_pos_out, slot, mask=is_ctx)
    tl.store(out_query_slot_mapping_ptr + query_out, slot, mask=is_query)

    bonus_token = tl.load(next_token_ids_ptr + req_idx)
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)

    is_sample = is_query & (query_off > 0)
    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
    tl.store(out_token_indices_ptr + sample_out_idx, query_out, mask=is_sample)


@dataclass
class Case:
    batch_size: int
    num_speculative_tokens: int
    block_size: int
    has_num_rejected: bool
    block_size_tl: int
    next_token_ids_cpu: torch.Tensor
    target_positions_cpu: torch.Tensor
    block_table_cpu: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    num_rejected_tokens_cpu: torch.Tensor


def fill_block_table(batch_size: int, max_blocks: int) -> torch.Tensor:
    block_table = torch.empty((batch_size, max_blocks), dtype=torch.int32)
    for req_idx in range(batch_size):
        # Keep fake block ids unique per request without making slot_mapping
        # overflow int32 when large batch sizes are benchmarked.
        start = req_idx * max_blocks
        block_table[req_idx] = torch.arange(
            start,
            start + max_blocks,
            dtype=torch.int32,
        )
    return block_table


def make_case(
    rng: random.Random,
    *,
    batch_size: int,
    num_speculative_tokens: int,
    block_size: int,
    has_num_rejected: bool,
    block_size_tl: int,
    max_prompt_len: int,
    max_position_base: int,
) -> Case:
    ctx_lens = [
        rng.randint(1, max_prompt_len)
        for _ in range(batch_size)
    ]
    query_start_loc = [0]
    for length in ctx_lens:
        query_start_loc.append(query_start_loc[-1] + length)

    target_positions: list[int] = []
    max_future_pos = 0
    for length in ctx_lens:
        base = rng.randint(0, max_position_base)
        positions = list(range(base, base + length))
        target_positions.extend(positions)
        max_future_pos = max(
            max_future_pos,
            positions[-1] + 1 + num_speculative_tokens,
        )

    max_blocks = max_future_pos // block_size + 3
    block_table = fill_block_table(batch_size, max_blocks)

    if has_num_rejected:
        rejected = [
            rng.randint(0, min(num_speculative_tokens, ctx_lens[i] - 1))
            for i in range(batch_size)
        ]
    else:
        rejected = [0] * batch_size

    return Case(
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        block_size=block_size,
        has_num_rejected=has_num_rejected,
        block_size_tl=block_size_tl,
        next_token_ids_cpu=torch.tensor(
            [rng.randint(1, 200000) for _ in range(batch_size)],
            dtype=torch.int32,
        ),
        target_positions_cpu=torch.tensor(target_positions, dtype=torch.int32),
        block_table_cpu=block_table,
        query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
        num_rejected_tokens_cpu=torch.tensor(rejected, dtype=torch.int32),
    )


def make_uniform_case(
    *,
    batch_size: int,
    prompt_len: int,
    num_speculative_tokens: int,
    block_size: int,
    has_num_rejected: bool,
    block_size_tl: int,
    max_position_base: int,
) -> Case:
    query_start_loc = [i * prompt_len for i in range(batch_size + 1)]

    target_positions: list[int] = []
    max_future_pos = 0
    for req_idx in range(batch_size):
        # Keep bases deterministic but separated enough to stress block-table
        # lookup without creating overlapping request slots.
        base = (req_idx * 997) % max_position_base
        positions = list(range(base, base + prompt_len))
        target_positions.extend(positions)
        max_future_pos = max(
            max_future_pos,
            positions[-1] + 1 + num_speculative_tokens,
        )

    max_blocks = max_future_pos // block_size + 3
    block_table = fill_block_table(batch_size, max_blocks)

    if has_num_rejected:
        rejected = [
            min(num_speculative_tokens, max(prompt_len - 1, 0))
            for _ in range(batch_size)
        ]
    else:
        rejected = [0] * batch_size

    return Case(
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        block_size=block_size,
        has_num_rejected=has_num_rejected,
        block_size_tl=block_size_tl,
        next_token_ids_cpu=torch.arange(
            1,
            batch_size + 1,
            dtype=torch.int32,
        ) + 1000,
        target_positions_cpu=torch.tensor(target_positions, dtype=torch.int32),
        block_table_cpu=block_table,
        query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
        num_rejected_tokens_cpu=torch.tensor(rejected, dtype=torch.int32),
    )


def cpu_reference(case: Case, parallel_drafting_token_id: int) -> dict[str, torch.Tensor]:
    batch_size = case.batch_size
    k = case.num_speculative_tokens
    num_query_per_req = k + 1
    num_context = int(case.target_positions_cpu.numel())
    num_query_total = batch_size * num_query_per_req

    out = {
        "input_ids": torch.full((num_query_total,), -1, dtype=torch.int32),
        "context_positions": torch.full((num_context,), -1, dtype=torch.int32),
        "query_positions": torch.full((num_query_total,), -1, dtype=torch.int32),
        "context_slot_mapping": torch.full((num_context,), -1, dtype=torch.int32),
        "query_slot_mapping": torch.full((num_query_total,), -1, dtype=torch.int32),
        "token_indices": torch.full((batch_size * k,), -1, dtype=torch.int32),
    }

    for req_idx in range(batch_size):
        ctx_start = int(case.query_start_loc_cpu[req_idx])
        ctx_end = int(case.query_start_loc_cpu[req_idx + 1])
        num_ctx = ctx_end - ctx_start

        for j in range(num_ctx):
            ctx_pos_idx = ctx_start + j
            pos = int(case.target_positions_cpu[ctx_pos_idx])
            out["context_positions"][ctx_pos_idx] = pos
            block_num = pos // case.block_size
            block_id = int(case.block_table_cpu[req_idx, block_num])
            slot = block_id * case.block_size + (pos % case.block_size)
            out["context_slot_mapping"][ctx_pos_idx] = slot

        if case.has_num_rejected:
            valid_ctx_end = ctx_end - int(case.num_rejected_tokens_cpu[req_idx])
        else:
            valid_ctx_end = ctx_end
        last_pos = int(case.target_positions_cpu[valid_ctx_end - 1])

        for q_idx in range(num_query_per_req):
            query_out_idx = req_idx * num_query_per_req + q_idx
            query_pos = last_pos + 1 + q_idx
            out["query_positions"][query_out_idx] = query_pos
            block_num_q = query_pos // case.block_size
            block_id_q = int(case.block_table_cpu[req_idx, block_num_q])
            slot_q = block_id_q * case.block_size + (query_pos % case.block_size)
            out["query_slot_mapping"][query_out_idx] = slot_q
            if q_idx == 0:
                out["input_ids"][query_out_idx] = case.next_token_ids_cpu[req_idx]
            else:
                out["input_ids"][query_out_idx] = parallel_drafting_token_id
                sample_out_idx = req_idx * k + (q_idx - 1)
                out["token_indices"][sample_out_idx] = query_out_idx

    return out


@dataclass
class DeviceCase:
    next_token_ids: torch.Tensor
    target_positions: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_rejected_tokens: torch.Tensor


def to_device_case(case: Case, device: torch.device) -> DeviceCase:
    return DeviceCase(
        next_token_ids=case.next_token_ids_cpu.to(device),
        target_positions=case.target_positions_cpu.to(device),
        block_table=case.block_table_cpu.to(device),
        query_start_loc=case.query_start_loc_cpu.to(device),
        num_rejected_tokens=case.num_rejected_tokens_cpu.to(device),
    )


def allocate_outputs(case: Case, device: torch.device) -> dict[str, torch.Tensor]:
    k = case.num_speculative_tokens
    num_query_per_req = k + 1
    num_context = int(case.target_positions_cpu.numel())
    num_query_total = case.batch_size * num_query_per_req
    return {
        "input_ids": torch.full((num_query_total,), -1, dtype=torch.int32, device=device),
        "context_positions": torch.full((num_context,), -1, dtype=torch.int32, device=device),
        "query_positions": torch.full((num_query_total,), -1, dtype=torch.int32, device=device),
        "context_slot_mapping": torch.full((num_context,), -1, dtype=torch.int32, device=device),
        "query_slot_mapping": torch.full((num_query_total,), -1, dtype=torch.int32, device=device),
        "token_indices": torch.full((case.batch_size * k,), -1, dtype=torch.int32, device=device),
    }


def launch_old_kernel(
    case: Case,
    device_case: DeviceCase,
    out: dict[str, torch.Tensor],
    parallel_drafting_token_id: int,
) -> None:
    copy_and_expand_dflash_inputs_kernel_single_grid[1,](
        next_token_ids_ptr=device_case.next_token_ids,
        target_positions_ptr=device_case.target_positions,
        out_input_ids_ptr=out["input_ids"],
        out_context_positions_ptr=out["context_positions"],
        out_query_positions_ptr=out["query_positions"],
        out_context_slot_mapping_ptr=out["context_slot_mapping"],
        out_query_slot_mapping_ptr=out["query_slot_mapping"],
        out_token_indices_ptr=out["token_indices"],
        block_table_ptr=device_case.block_table,
        block_table_stride=device_case.block_table.stride(0),
        query_start_loc_ptr=device_case.query_start_loc,
        num_rejected_tokens_ptr=(
            device_case.num_rejected_tokens if case.has_num_rejected else 0
        ),
        parallel_drafting_token_id=parallel_drafting_token_id,
        block_size=case.block_size,
        num_query_per_req=case.num_speculative_tokens + 1,
        num_speculative_tokens=case.num_speculative_tokens,
        total_input_tokens=case.target_positions_cpu.numel(),
        batch_size=case.batch_size,
        HAS_NUM_REJECTED=case.has_num_rejected,
    )


def launch_new_kernel(
    case: Case,
    device_case: DeviceCase,
    out: dict[str, torch.Tensor],
    parallel_drafting_token_id: int,
) -> None:
    max_ctx_per_req = max(
        int(case.query_start_loc_cpu[i + 1] - case.query_start_loc_cpu[i])
        for i in range(case.batch_size)
    )
    max_tokens_per_req = max_ctx_per_req + case.num_speculative_tokens + 1
    num_blocks = math.ceil(max_tokens_per_req / case.block_size_tl)
    grid = (case.batch_size, num_blocks)

    copy_and_expand_dflash_inputs_kernel_2d[grid](
        next_token_ids_ptr=device_case.next_token_ids,
        target_positions_ptr=device_case.target_positions,
        out_input_ids_ptr=out["input_ids"],
        out_context_positions_ptr=out["context_positions"],
        out_query_positions_ptr=out["query_positions"],
        out_context_slot_mapping_ptr=out["context_slot_mapping"],
        out_query_slot_mapping_ptr=out["query_slot_mapping"],
        out_token_indices_ptr=out["token_indices"],
        block_table_ptr=device_case.block_table,
        block_table_stride=device_case.block_table.stride(0),
        query_start_loc_ptr=device_case.query_start_loc,
        num_rejected_tokens_ptr=(
            device_case.num_rejected_tokens if case.has_num_rejected else 0
        ),
        parallel_drafting_token_id=parallel_drafting_token_id,
        block_size=case.block_size,
        num_query_per_req=case.num_speculative_tokens + 1,
        num_speculative_tokens=case.num_speculative_tokens,
        total_input_tokens=case.target_positions_cpu.numel(),
        BLOCK_SIZE=case.block_size_tl,
        HAS_NUM_REJECTED=case.has_num_rejected,
    )


def run_old_kernel(case: Case, device: torch.device, parallel_drafting_token_id: int) -> dict[str, torch.Tensor]:
    out = allocate_outputs(case, device)
    device_case = to_device_case(case, device)
    launch_old_kernel(case, device_case, out, parallel_drafting_token_id)
    torch.npu.synchronize()
    return out


def run_new_kernel(case: Case, device: torch.device, parallel_drafting_token_id: int) -> dict[str, torch.Tensor]:
    out = allocate_outputs(case, device)
    device_case = to_device_case(case, device)
    launch_new_kernel(case, device_case, out, parallel_drafting_token_id)
    torch.npu.synchronize()
    return out


def assert_outputs(
    case: Case,
    expected: dict[str, torch.Tensor],
    old_out: dict[str, torch.Tensor],
    new_out: dict[str, torch.Tensor],
) -> None:
    for name, expected_cpu in expected.items():
        old_cpu = old_out[name].cpu()
        new_cpu = new_out[name].cpu()
        if not torch.equal(old_cpu, expected_cpu):
            mismatch = torch.nonzero(old_cpu != expected_cpu).flatten()[0].item()
            raise AssertionError(
                f"old kernel mismatch: {name}, index={mismatch}, "
                f"expected={expected_cpu[mismatch].item()}, "
                f"actual={old_cpu[mismatch].item()}, case={case}"
            )
        if not torch.equal(new_cpu, expected_cpu):
            mismatch = torch.nonzero(new_cpu != expected_cpu).flatten()[0].item()
            raise AssertionError(
                f"new kernel mismatch: {name}, index={mismatch}, "
                f"expected={expected_cpu[mismatch].item()}, "
                f"actual={new_cpu[mismatch].item()}, case={case}"
            )
        if not torch.equal(new_cpu, old_cpu):
            mismatch = torch.nonzero(new_cpu != old_cpu).flatten()[0].item()
            raise AssertionError(
                f"old/new mismatch: {name}, index={mismatch}, "
                f"old={old_cpu[mismatch].item()}, new={new_cpu[mismatch].item()}, "
                f"case={case}"
            )


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def time_kernel_ms(
    launch_fn,
    case: Case,
    device: torch.device,
    parallel_drafting_token_id: int,
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    device_case = to_device_case(case, device)
    out = allocate_outputs(case, device)

    for _ in range(warmup):
        launch_fn(case, device_case, out, parallel_drafting_token_id)
    torch.npu.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        launch_fn(case, device_case, out, parallel_drafting_token_id)
        end.record()
        torch.npu.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def summarize_times(values: list[float]) -> dict[str, float]:
    return {
        "avg": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def run_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    batch_sizes = parse_int_list(args.bench_batch_sizes)
    prompt_lens = parse_int_list(args.bench_prompt_lens)
    spec_tokens_list = parse_int_list(args.bench_spec_tokens)
    kv_block_sizes = parse_int_list(args.bench_kv_block_sizes)
    triton_block_sizes = parse_int_list(args.bench_triton_block_sizes)

    print("\nBenchmark: kernel-only elapsed time in milliseconds")
    print(
        "batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,"
        "old_avg,old_p50,old_p90,old_p99,"
        "new_avg,new_p50,new_p90,new_p99,speedup_avg"
    )

    for batch_size in batch_sizes:
        for prompt_len in prompt_lens:
            for k in spec_tokens_list:
                for kv_block_size in kv_block_sizes:
                    for block_size_tl in triton_block_sizes:
                        for has_num_rejected in (False, True):
                            case = make_uniform_case(
                                batch_size=batch_size,
                                prompt_len=prompt_len,
                                num_speculative_tokens=k,
                                block_size=kv_block_size,
                                has_num_rejected=has_num_rejected,
                                block_size_tl=block_size_tl,
                                max_position_base=args.bench_max_position_base,
                            )
                            # Benchmark case correctness before timing, so a
                            # fast but wrong kernel does not produce numbers.
                            expected = cpu_reference(
                                case,
                                args.parallel_drafting_token_id,
                            )
                            old_out = run_old_kernel(
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            new_out = run_new_kernel(
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            assert_outputs(case, expected, old_out, new_out)

                            old_times = time_kernel_ms(
                                launch_old_kernel,
                                case,
                                device,
                                args.parallel_drafting_token_id,
                                warmup=args.bench_warmup,
                                iters=args.bench_iters,
                            )
                            new_times = time_kernel_ms(
                                launch_new_kernel,
                                case,
                                device,
                                args.parallel_drafting_token_id,
                                warmup=args.bench_warmup,
                                iters=args.bench_iters,
                            )
                            old_s = summarize_times(old_times)
                            new_s = summarize_times(new_times)
                            speedup = (
                                old_s["avg"] / new_s["avg"]
                                if new_s["avg"] > 0
                                else float("inf")
                            )
                            print(
                                f"{batch_size},{prompt_len},{k},"
                                f"{kv_block_size},{block_size_tl},"
                                f"{int(has_num_rejected)},"
                                f"{old_s['avg']:.4f},{old_s['p50']:.4f},"
                                f"{old_s['p90']:.4f},{old_s['p99']:.4f},"
                                f"{new_s['avg']:.4f},{new_s['p50']:.4f},"
                                f"{new_s['p90']:.4f},{new_s['p99']:.4f},"
                                f"{speedup:.2f}"
                            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--cases-per-combo", type=int, default=5)
    parser.add_argument("--batch-sizes", default="1,2,8,32,64,128")
    parser.add_argument("--spec-tokens", default="1,2,4,8")
    parser.add_argument("--kv-block-sizes", default="16,32,128")
    parser.add_argument("--triton-block-sizes", default="16,64,128,256")
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-position-base", type=int, default=8192)
    parser.add_argument("--parallel-drafting-token-id", type=int, default=999999)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also run kernel-only timing comparison after precision checks.",
    )
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--bench-batch-sizes", default="32,64,128")
    parser.add_argument("--bench-prompt-lens", default="256,1024,4096")
    parser.add_argument("--bench-spec-tokens", default="4,8")
    parser.add_argument("--bench-kv-block-sizes", default="128")
    parser.add_argument("--bench-triton-block-sizes", default="64,128,256")
    parser.add_argument("--bench-max-position-base", type=int, default=65536)
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available. Please check torch_npu setup.")

    device = torch.device(args.device)
    torch.npu.set_device(device)

    rng = random.Random(args.seed)
    batch_sizes = parse_int_list(args.batch_sizes)
    spec_tokens_list = parse_int_list(args.spec_tokens)
    kv_block_sizes = parse_int_list(args.kv_block_sizes)
    triton_block_sizes = parse_int_list(args.triton_block_sizes)

    total = 0
    for batch_size in batch_sizes:
        for num_speculative_tokens in spec_tokens_list:
            for kv_block_size in kv_block_sizes:
                for has_num_rejected in (False, True):
                    for block_size_tl in triton_block_sizes:
                        for _ in range(args.cases_per_combo):
                            case = make_case(
                                rng,
                                batch_size=batch_size,
                                num_speculative_tokens=num_speculative_tokens,
                                block_size=kv_block_size,
                                has_num_rejected=has_num_rejected,
                                block_size_tl=block_size_tl,
                                max_prompt_len=args.max_prompt_len,
                                max_position_base=args.max_position_base,
                            )
                            expected = cpu_reference(
                                case,
                                args.parallel_drafting_token_id,
                            )
                            old_out = run_old_kernel(
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            new_out = run_new_kernel(
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            assert_outputs(case, expected, old_out, new_out)
                            total += 1
                            if total % 50 == 0:
                                print(f"verified cases={total}")

    print(f"PASS NPU DFlash input-expand precision cases={total}")

    if args.benchmark:
        run_benchmark(args, device)


if __name__ == "__main__":
    main()
