#!/usr/bin/env python3
"""CPU semantic verification for DFlash 2-D grid input-expand.

This script does not require Ascend NPU, torch-npu, Triton, vLLM, or
vLLM-Ascend. It compares a Python translation of the current single-grid
kernel with a Python translation of the proposed 2-D grid kernel.

It is not a performance proxy for NPU kernels. The optional benchmark only
measures Python-loop cost and is intended to make the verification flow easy
to run locally.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from dataclasses import dataclass


@dataclass
class Case:
    batch_size: int
    num_speculative_tokens: int
    block_size: int
    has_num_rejected: bool
    block_size_tl: int
    next_token_ids: list[int]
    target_positions: list[int]
    block_table: list[list[int]]
    query_start_loc: list[int]
    num_rejected_tokens: list[int]


@dataclass
class Outputs:
    input_ids: list[int]
    context_positions: list[int]
    query_positions: list[int]
    context_slot_mapping: list[int]
    query_slot_mapping: list[int]
    token_indices: list[int]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def fill_block_table(batch_size: int, max_blocks: int) -> list[list[int]]:
    return [
        [req_idx * max_blocks + block_idx for block_idx in range(max_blocks)]
        for req_idx in range(batch_size)
    ]


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
    ctx_lens = [rng.randint(1, max_prompt_len) for _ in range(batch_size)]
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
        next_token_ids=[
            rng.randint(1, 200000)
            for _ in range(batch_size)
        ],
        target_positions=target_positions,
        block_table=block_table,
        query_start_loc=query_start_loc,
        num_rejected_tokens=rejected,
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
        next_token_ids=[1000 + i for i in range(batch_size)],
        target_positions=target_positions,
        block_table=block_table,
        query_start_loc=query_start_loc,
        num_rejected_tokens=rejected,
    )


def empty_outputs(case: Case) -> Outputs:
    k = case.num_speculative_tokens
    num_query_per_req = k + 1
    num_context = len(case.target_positions)
    num_query_total = case.batch_size * num_query_per_req
    sentinel = -1
    return Outputs(
        input_ids=[sentinel] * num_query_total,
        context_positions=[sentinel] * num_context,
        query_positions=[sentinel] * num_query_total,
        context_slot_mapping=[sentinel] * num_context,
        query_slot_mapping=[sentinel] * num_query_total,
        token_indices=[sentinel] * (case.batch_size * k),
    )


def old_single_grid_cpu(case: Case, parallel_drafting_token_id: int) -> Outputs:
    out = empty_outputs(case)
    k = case.num_speculative_tokens
    num_query_per_req = k + 1

    for req_idx in range(case.batch_size):
        ctx_start = case.query_start_loc[req_idx]
        ctx_end = case.query_start_loc[req_idx + 1]
        num_ctx = ctx_end - ctx_start
        assert num_ctx > 0

        for j in range(num_ctx):
            ctx_pos_idx = ctx_start + j
            pos = case.target_positions[ctx_pos_idx]
            out.context_positions[ctx_pos_idx] = pos

            block_num = pos // case.block_size
            block_id = case.block_table[req_idx][block_num]
            slot = block_id * case.block_size + (pos % case.block_size)
            out.context_slot_mapping[ctx_pos_idx] = slot

        if case.has_num_rejected:
            valid_ctx_end = ctx_end - case.num_rejected_tokens[req_idx]
        else:
            valid_ctx_end = ctx_end
        assert valid_ctx_end > ctx_start
        last_pos = case.target_positions[valid_ctx_end - 1]

        for q_idx in range(num_query_per_req):
            query_pos = last_pos + 1 + q_idx
            query_out_idx = req_idx * num_query_per_req + q_idx
            out.query_positions[query_out_idx] = query_pos

            block_num_q = query_pos // case.block_size
            block_id_q = case.block_table[req_idx][block_num_q]
            slot_q = block_id_q * case.block_size + (query_pos % case.block_size)
            out.query_slot_mapping[query_out_idx] = slot_q

            if q_idx == 0:
                out.input_ids[query_out_idx] = case.next_token_ids[req_idx]
            else:
                out.input_ids[query_out_idx] = parallel_drafting_token_id
                sample_out_idx = req_idx * k + (q_idx - 1)
                out.token_indices[sample_out_idx] = query_out_idx

    return out


def new_2d_grid_cpu(case: Case, parallel_drafting_token_id: int) -> Outputs:
    out = empty_outputs(case)
    k = case.num_speculative_tokens
    num_query_per_req = k + 1
    block_table_stride = len(case.block_table[0])

    max_ctx_per_req = max(
        case.query_start_loc[i + 1] - case.query_start_loc[i]
        for i in range(case.batch_size)
    )
    max_tokens_per_req = max_ctx_per_req + num_query_per_req
    num_blocks = math.ceil(max_tokens_per_req / case.block_size_tl)
    total_input_tokens = len(case.target_positions)

    for req_idx in range(case.batch_size):
        ctx_start = case.query_start_loc[req_idx]
        ctx_end = case.query_start_loc[req_idx + 1]
        num_ctx = ctx_end - ctx_start
        total_tokens = num_ctx + num_query_per_req
        assert num_ctx > 0

        if case.has_num_rejected:
            valid_ctx_end = ctx_end - case.num_rejected_tokens[req_idx]
        else:
            valid_ctx_end = ctx_end
        assert valid_ctx_end > ctx_start
        last_pos = case.target_positions[valid_ctx_end - 1]

        for block_idx in range(num_blocks):
            for lane in range(case.block_size_tl):
                j = block_idx * case.block_size_tl + lane
                in_bounds = j < total_tokens
                is_ctx = j < num_ctx
                is_query = (not is_ctx) and in_bounds
                query_off = j - num_ctx

                if is_ctx:
                    ctx_pos_idx = min(ctx_start + j, total_input_tokens - 1)
                    ctx_pos = case.target_positions[ctx_pos_idx]
                    positions = ctx_pos
                    ctx_pos_out = ctx_start + j
                    out.context_positions[ctx_pos_out] = ctx_pos
                elif is_query:
                    query_pos = last_pos + 1 + query_off
                    positions = query_pos
                    query_out = req_idx * num_query_per_req + query_off
                    out.query_positions[query_out] = query_pos
                else:
                    continue

                block_num = positions // case.block_size
                block_num = min(block_num, block_table_stride - 1)
                block_id = case.block_table[req_idx][block_num]
                slot = block_id * case.block_size + (positions % case.block_size)

                if is_ctx:
                    out.context_slot_mapping[ctx_pos_out] = slot
                else:
                    out.query_slot_mapping[query_out] = slot
                    if query_off == 0:
                        out.input_ids[query_out] = case.next_token_ids[req_idx]
                    else:
                        out.input_ids[query_out] = parallel_drafting_token_id
                        sample_out_idx = req_idx * k + (query_off - 1)
                        out.token_indices[sample_out_idx] = query_out

    return out


def output_items(out: Outputs):
    return {
        "input_ids": out.input_ids,
        "context_positions": out.context_positions,
        "query_positions": out.query_positions,
        "context_slot_mapping": out.context_slot_mapping,
        "query_slot_mapping": out.query_slot_mapping,
        "token_indices": out.token_indices,
    }.items()


def assert_outputs_equal(case: Case, old: Outputs, new: Outputs) -> None:
    for name, old_value in output_items(old):
        new_value = dict(output_items(new))[name]
        if old_value != new_value:
            for idx, (lhs, rhs) in enumerate(zip(old_value, new_value)):
                if lhs != rhs:
                    raise AssertionError(
                        f"mismatch {name}: index={idx}, old={lhs}, "
                        f"new={rhs}, case={case}"
                    )
            raise AssertionError(
                f"mismatch {name}: different lengths, "
                f"old={len(old_value)}, new={len(new_value)}, case={case}"
            )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def summarize_times(values: list[float]) -> dict[str, float]:
    return {
        "avg": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def time_fn_ms(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return times


def run_precision(args: argparse.Namespace) -> None:
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
                            old = old_single_grid_cpu(
                                case,
                                args.parallel_drafting_token_id,
                            )
                            new = new_2d_grid_cpu(
                                case,
                                args.parallel_drafting_token_id,
                            )
                            assert_outputs_equal(case, old, new)
                            total += 1
                            if args.progress_every and total % args.progress_every == 0:
                                print(f"verified cases={total}")

    print(f"PASS CPU DFlash input-expand semantic cases={total}")


def run_benchmark(args: argparse.Namespace) -> None:
    batch_sizes = parse_int_list(args.bench_batch_sizes)
    prompt_lens = parse_int_list(args.bench_prompt_lens)
    spec_tokens_list = parse_int_list(args.bench_spec_tokens)
    kv_block_sizes = parse_int_list(args.bench_kv_block_sizes)
    triton_block_sizes = parse_int_list(args.bench_triton_block_sizes)

    print("\nCPU benchmark: Python-loop elapsed time in milliseconds")
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
                            old = old_single_grid_cpu(
                                case,
                                args.parallel_drafting_token_id,
                            )
                            new = new_2d_grid_cpu(
                                case,
                                args.parallel_drafting_token_id,
                            )
                            assert_outputs_equal(case, old, new)

                            old_times = time_fn_ms(
                                lambda: old_single_grid_cpu(
                                    case,
                                    args.parallel_drafting_token_id,
                                ),
                                warmup=args.bench_warmup,
                                iters=args.bench_iters,
                            )
                            new_times = time_fn_ms(
                                lambda: new_2d_grid_cpu(
                                    case,
                                    args.parallel_drafting_token_id,
                                ),
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
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--cases-per-combo", type=int, default=10)
    parser.add_argument("--batch-sizes", default="1,2,8,32,64,128")
    parser.add_argument("--spec-tokens", default="1,2,4,8")
    parser.add_argument("--kv-block-sizes", default="16,32,128")
    parser.add_argument("--triton-block-sizes", default="1,7,16,64,128,256")
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-position-base", type=int, default=8192)
    parser.add_argument("--parallel-drafting-token-id", type=int, default=999999)
    parser.add_argument("--progress-every", type=int, default=0)

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run CPU Python-loop timing comparison after semantic checks.",
    )
    parser.add_argument("--bench-warmup", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument("--bench-batch-sizes", default="32,64,128")
    parser.add_argument("--bench-prompt-lens", default="256,1024")
    parser.add_argument("--bench-spec-tokens", default="4,8")
    parser.add_argument("--bench-kv-block-sizes", default="128")
    parser.add_argument("--bench-triton-block-sizes", default="64,128,256")
    parser.add_argument("--bench-max-position-base", type=int, default=65536)

    args = parser.parse_args()

    run_precision(args)
    if args.benchmark:
        run_benchmark(args)


if __name__ == "__main__":
    main()
