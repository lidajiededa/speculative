#!/usr/bin/env python3
"""CUDA semantic and timing check for DFlash 2-D grid input-expand.

This script compares two CUDA micro kernels:

* single-grid: one CUDA thread serially walks all requests and context tokens;
* 2-D grid: one CUDA block handles one request/tile, and threads in that block
  handle token lanes in parallel.

It is a GPU hardware-concurrency simulation for the input-expand step only.
It does not measure full vLLM speculative decoding latency.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
#include <torch/extension.h>

void launch_single_grid_cuda(
    torch::Tensor next_token_ids,
    torch::Tensor target_positions,
    torch::Tensor out_input_ids,
    torch::Tensor out_context_positions,
    torch::Tensor out_query_positions,
    torch::Tensor out_context_slot_mapping,
    torch::Tensor out_query_slot_mapping,
    torch::Tensor out_token_indices,
    torch::Tensor block_table,
    torch::Tensor query_start_loc,
    torch::Tensor num_rejected_tokens,
    int block_table_stride,
    int parallel_drafting_token_id,
    int block_size,
    int num_query_per_req,
    int num_speculative_tokens,
    int batch_size,
    bool has_num_rejected);

void launch_2d_grid_cuda(
    torch::Tensor next_token_ids,
    torch::Tensor target_positions,
    torch::Tensor out_input_ids,
    torch::Tensor out_context_positions,
    torch::Tensor out_query_positions,
    torch::Tensor out_context_slot_mapping,
    torch::Tensor out_query_slot_mapping,
    torch::Tensor out_token_indices,
    torch::Tensor block_table,
    torch::Tensor query_start_loc,
    torch::Tensor num_rejected_tokens,
    int block_table_stride,
    int parallel_drafting_token_id,
    int block_size,
    int num_query_per_req,
    int num_speculative_tokens,
    int total_input_tokens,
    int batch_size,
    int num_blocks,
    int block_size_tl,
    bool has_num_rejected);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)

__global__ void single_grid_kernel(
    const int* __restrict__ next_token_ids,
    const int* __restrict__ target_positions,
    int* __restrict__ out_input_ids,
    int* __restrict__ out_context_positions,
    int* __restrict__ out_query_positions,
    int* __restrict__ out_context_slot_mapping,
    int* __restrict__ out_query_slot_mapping,
    int* __restrict__ out_token_indices,
    const int* __restrict__ block_table,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ num_rejected_tokens,
    int block_table_stride,
    int parallel_drafting_token_id,
    int block_size,
    int num_query_per_req,
    int num_speculative_tokens,
    int batch_size,
    bool has_num_rejected) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
    int ctx_start = query_start_loc[req_idx];
    int ctx_end = query_start_loc[req_idx + 1];
    int num_ctx = ctx_end - ctx_start;

    for (int j = 0; j < num_ctx; ++j) {
      int ctx_pos_idx = ctx_start + j;
      int pos = target_positions[ctx_pos_idx];
      out_context_positions[ctx_pos_idx] = pos;

      int block_num = pos / block_size;
      if (block_num >= block_table_stride) {
        block_num = block_table_stride - 1;
      }
      int block_id = block_table[req_idx * block_table_stride + block_num];
      int slot = block_id * block_size + (pos % block_size);
      out_context_slot_mapping[ctx_pos_idx] = slot;
    }

    int valid_ctx_end = ctx_end;
    if (has_num_rejected) {
      valid_ctx_end = ctx_end - num_rejected_tokens[req_idx];
    }
    int last_pos = target_positions[valid_ctx_end - 1];

    for (int q_idx = 0; q_idx < num_query_per_req; ++q_idx) {
      int query_pos = last_pos + 1 + q_idx;
      int query_out_idx = req_idx * num_query_per_req + q_idx;
      out_query_positions[query_out_idx] = query_pos;

      int block_num_q = query_pos / block_size;
      if (block_num_q >= block_table_stride) {
        block_num_q = block_table_stride - 1;
      }
      int block_id_q = block_table[req_idx * block_table_stride + block_num_q];
      int slot_q = block_id_q * block_size + (query_pos % block_size);
      out_query_slot_mapping[query_out_idx] = slot_q;

      if (q_idx == 0) {
        out_input_ids[query_out_idx] = next_token_ids[req_idx];
      } else {
        out_input_ids[query_out_idx] = parallel_drafting_token_id;
        int sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1);
        out_token_indices[sample_out_idx] = query_out_idx;
      }
    }
  }
}

__global__ void two_d_grid_kernel(
    const int* __restrict__ next_token_ids,
    const int* __restrict__ target_positions,
    int* __restrict__ out_input_ids,
    int* __restrict__ out_context_positions,
    int* __restrict__ out_query_positions,
    int* __restrict__ out_context_slot_mapping,
    int* __restrict__ out_query_slot_mapping,
    int* __restrict__ out_token_indices,
    const int* __restrict__ block_table,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ num_rejected_tokens,
    int block_table_stride,
    int parallel_drafting_token_id,
    int block_size,
    int num_query_per_req,
    int num_speculative_tokens,
    int total_input_tokens,
    bool has_num_rejected) {
  int req_idx = blockIdx.x;
  int block_idx = blockIdx.y;
  int lane = threadIdx.x;
  int block_size_tl = blockDim.x;

  int ctx_start = query_start_loc[req_idx];
  int ctx_end = query_start_loc[req_idx + 1];
  int num_ctx = ctx_end - ctx_start;
  int total_tokens = num_ctx + num_query_per_req;
  int j = block_idx * block_size_tl + lane;
  if (j >= total_tokens) {
    return;
  }

  int valid_ctx_end = ctx_end;
  if (has_num_rejected) {
    valid_ctx_end = ctx_end - num_rejected_tokens[req_idx];
  }
  int last_pos = target_positions[valid_ctx_end - 1];

  bool is_ctx = j < num_ctx;
  int positions;
  int query_off = j - num_ctx;
  int ctx_pos_out = ctx_start + j;
  int query_out = req_idx * num_query_per_req + query_off;

  if (is_ctx) {
    int ctx_pos_idx = ctx_start + j;
    if (ctx_pos_idx >= total_input_tokens) {
      ctx_pos_idx = total_input_tokens - 1;
    }
    positions = target_positions[ctx_pos_idx];
    out_context_positions[ctx_pos_out] = positions;
  } else {
    positions = last_pos + 1 + query_off;
    out_query_positions[query_out] = positions;
  }

  int block_num = positions / block_size;
  if (block_num >= block_table_stride) {
    block_num = block_table_stride - 1;
  }
  int block_id = block_table[req_idx * block_table_stride + block_num];
  int slot = block_id * block_size + (positions % block_size);

  if (is_ctx) {
    out_context_slot_mapping[ctx_pos_out] = slot;
  } else {
    out_query_slot_mapping[query_out] = slot;
    if (query_off == 0) {
      out_input_ids[query_out] = next_token_ids[req_idx];
    } else {
      out_input_ids[query_out] = parallel_drafting_token_id;
      int sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1);
      out_token_indices[sample_out_idx] = query_out;
    }
  }
}

static void check_common_inputs(
    torch::Tensor next_token_ids,
    torch::Tensor target_positions,
    torch::Tensor out_input_ids,
    torch::Tensor out_context_positions,
    torch::Tensor out_query_positions,
    torch::Tensor out_context_slot_mapping,
    torch::Tensor out_query_slot_mapping,
    torch::Tensor out_token_indices,
    torch::Tensor block_table,
    torch::Tensor query_start_loc,
    torch::Tensor num_rejected_tokens) {
  CHECK_INPUT(next_token_ids);
  CHECK_INPUT(target_positions);
  CHECK_INPUT(out_input_ids);
  CHECK_INPUT(out_context_positions);
  CHECK_INPUT(out_query_positions);
  CHECK_INPUT(out_context_slot_mapping);
  CHECK_INPUT(out_query_slot_mapping);
  CHECK_INPUT(out_token_indices);
  CHECK_INPUT(block_table);
  CHECK_INPUT(query_start_loc);
  CHECK_INPUT(num_rejected_tokens);
}

void launch_single_grid_cuda(
    torch::Tensor next_token_ids,
    torch::Tensor target_positions,
    torch::Tensor out_input_ids,
    torch::Tensor out_context_positions,
    torch::Tensor out_query_positions,
    torch::Tensor out_context_slot_mapping,
    torch::Tensor out_query_slot_mapping,
    torch::Tensor out_token_indices,
    torch::Tensor block_table,
    torch::Tensor query_start_loc,
    torch::Tensor num_rejected_tokens,
    int block_table_stride,
    int parallel_drafting_token_id,
    int block_size,
    int num_query_per_req,
    int num_speculative_tokens,
    int batch_size,
    bool has_num_rejected) {
  check_common_inputs(
      next_token_ids,
      target_positions,
      out_input_ids,
      out_context_positions,
      out_query_positions,
      out_context_slot_mapping,
      out_query_slot_mapping,
      out_token_indices,
      block_table,
      query_start_loc,
      num_rejected_tokens);

  single_grid_kernel<<<1, 1>>>(
      next_token_ids.data_ptr<int>(),
      target_positions.data_ptr<int>(),
      out_input_ids.data_ptr<int>(),
      out_context_positions.data_ptr<int>(),
      out_query_positions.data_ptr<int>(),
      out_context_slot_mapping.data_ptr<int>(),
      out_query_slot_mapping.data_ptr<int>(),
      out_token_indices.data_ptr<int>(),
      block_table.data_ptr<int>(),
      query_start_loc.data_ptr<int>(),
      num_rejected_tokens.data_ptr<int>(),
      block_table_stride,
      parallel_drafting_token_id,
      block_size,
      num_query_per_req,
      num_speculative_tokens,
      batch_size,
      has_num_rejected);
}

void launch_2d_grid_cuda(
    torch::Tensor next_token_ids,
    torch::Tensor target_positions,
    torch::Tensor out_input_ids,
    torch::Tensor out_context_positions,
    torch::Tensor out_query_positions,
    torch::Tensor out_context_slot_mapping,
    torch::Tensor out_query_slot_mapping,
    torch::Tensor out_token_indices,
    torch::Tensor block_table,
    torch::Tensor query_start_loc,
    torch::Tensor num_rejected_tokens,
    int block_table_stride,
    int parallel_drafting_token_id,
    int block_size,
    int num_query_per_req,
    int num_speculative_tokens,
    int total_input_tokens,
    int batch_size,
    int num_blocks,
    int block_size_tl,
    bool has_num_rejected) {
  check_common_inputs(
      next_token_ids,
      target_positions,
      out_input_ids,
      out_context_positions,
      out_query_positions,
      out_context_slot_mapping,
      out_query_slot_mapping,
      out_token_indices,
      block_table,
      query_start_loc,
      num_rejected_tokens);
  TORCH_CHECK(block_size_tl > 0 && block_size_tl <= 1024,
              "block_size_tl must be in [1, 1024]");

  dim3 grid(batch_size, num_blocks);
  two_d_grid_kernel<<<grid, block_size_tl>>>(
      next_token_ids.data_ptr<int>(),
      target_positions.data_ptr<int>(),
      out_input_ids.data_ptr<int>(),
      out_context_positions.data_ptr<int>(),
      out_query_positions.data_ptr<int>(),
      out_context_slot_mapping.data_ptr<int>(),
      out_query_slot_mapping.data_ptr<int>(),
      out_token_indices.data_ptr<int>(),
      block_table.data_ptr<int>(),
      query_start_loc.data_ptr<int>(),
      num_rejected_tokens.data_ptr<int>(),
      block_table_stride,
      parallel_drafting_token_id,
      block_size,
      num_query_per_req,
      num_speculative_tokens,
      total_input_tokens,
      has_num_rejected);
}
"""


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


@dataclass
class DeviceCase:
    next_token_ids: torch.Tensor
    target_positions: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_rejected_tokens: torch.Tensor


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def load_cuda_module(verbose: bool):
    return load_inline(
        name="dflash_2d_grid_cuda_ext_v1",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["launch_single_grid_cuda", "launch_2d_grid_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )


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
            block_num = min(pos // case.block_size, case.block_table_cpu.shape[1] - 1)
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

            block_num_q = min(
                query_pos // case.block_size,
                case.block_table_cpu.shape[1] - 1,
            )
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


def to_device_case(case: Case, device: torch.device) -> DeviceCase:
    return DeviceCase(
        next_token_ids=case.next_token_ids_cpu.contiguous().to(device),
        target_positions=case.target_positions_cpu.contiguous().to(device),
        block_table=case.block_table_cpu.contiguous().to(device),
        query_start_loc=case.query_start_loc_cpu.contiguous().to(device),
        num_rejected_tokens=case.num_rejected_tokens_cpu.contiguous().to(device),
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


def num_blocks_for_case(case: Case) -> int:
    max_ctx_per_req = int(
        (case.query_start_loc_cpu[1:] - case.query_start_loc_cpu[:-1]).max().item()
    )
    max_tokens_per_req = max_ctx_per_req + case.num_speculative_tokens + 1
    return math.ceil(max_tokens_per_req / case.block_size_tl)


def launch_old_kernel(
    ext,
    case: Case,
    device_case: DeviceCase,
    out: dict[str, torch.Tensor],
    parallel_drafting_token_id: int,
) -> None:
    ext.launch_single_grid_cuda(
        device_case.next_token_ids,
        device_case.target_positions,
        out["input_ids"],
        out["context_positions"],
        out["query_positions"],
        out["context_slot_mapping"],
        out["query_slot_mapping"],
        out["token_indices"],
        device_case.block_table,
        device_case.query_start_loc,
        device_case.num_rejected_tokens,
        int(case.block_table_cpu.shape[1]),
        int(parallel_drafting_token_id),
        int(case.block_size),
        int(case.num_speculative_tokens + 1),
        int(case.num_speculative_tokens),
        int(case.batch_size),
        bool(case.has_num_rejected),
    )


def launch_new_kernel(
    ext,
    case: Case,
    device_case: DeviceCase,
    out: dict[str, torch.Tensor],
    parallel_drafting_token_id: int,
) -> None:
    ext.launch_2d_grid_cuda(
        device_case.next_token_ids,
        device_case.target_positions,
        out["input_ids"],
        out["context_positions"],
        out["query_positions"],
        out["context_slot_mapping"],
        out["query_slot_mapping"],
        out["token_indices"],
        device_case.block_table,
        device_case.query_start_loc,
        device_case.num_rejected_tokens,
        int(case.block_table_cpu.shape[1]),
        int(parallel_drafting_token_id),
        int(case.block_size),
        int(case.num_speculative_tokens + 1),
        int(case.num_speculative_tokens),
        int(case.target_positions_cpu.numel()),
        int(case.batch_size),
        int(num_blocks_for_case(case)),
        int(case.block_size_tl),
        bool(case.has_num_rejected),
    )


def run_kernel(
    launch_fn,
    ext,
    case: Case,
    device: torch.device,
    parallel_drafting_token_id: int,
) -> dict[str, torch.Tensor]:
    device_case = to_device_case(case, device)
    out = allocate_outputs(case, device)
    launch_fn(ext, case, device_case, out, parallel_drafting_token_id)
    torch.cuda.synchronize(device)
    return {name: value.cpu() for name, value in out.items()}


def assert_outputs(
    case: Case,
    expected: dict[str, torch.Tensor],
    old_out: dict[str, torch.Tensor],
    new_out: dict[str, torch.Tensor],
) -> None:
    for name, expected_tensor in expected.items():
        for label, actual in (("single_grid", old_out[name]), ("2d_grid", new_out[name])):
            if not torch.equal(expected_tensor, actual):
                mismatch = (expected_tensor != actual).nonzero(as_tuple=False).flatten()
                idx = int(mismatch[0]) if int(mismatch.numel()) else -1
                raise AssertionError(
                    f"{label} output mismatch for {name} at index={idx}; "
                    f"expected={int(expected_tensor[idx]) if idx >= 0 else 'n/a'} "
                    f"actual={int(actual[idx]) if idx >= 0 else 'n/a'}; "
                    f"batch={case.batch_size} k={case.num_speculative_tokens} "
                    f"kv_block={case.block_size} block_size_tl={case.block_size_tl} "
                    f"rejected={case.has_num_rejected}"
                )


def time_kernel_ms(
    launch_fn,
    ext,
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
        launch_fn(ext, case, device_case, out, parallel_drafting_token_id)
    torch.cuda.synchronize(device)

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        launch_fn(ext, case, device_case, out, parallel_drafting_token_id)
        end.record()
        torch.cuda.synchronize(device)
        times.append(float(start.elapsed_time(end)))
    return times


def summarize_times(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pct(percentile: float) -> float:
        if not ordered:
            return float("nan")
        idx = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
        return ordered[idx]

    return {
        "avg": statistics.mean(values),
        "p50": statistics.median(values),
        "p90": pct(0.90),
        "p99": pct(0.99),
    }


def run_precision(args: argparse.Namespace, ext, device: torch.device) -> int:
    rng = random.Random(args.seed)
    total = 0
    for batch_size in parse_int_list(args.batch_sizes):
        for num_speculative_tokens in parse_int_list(args.spec_tokens):
            for kv_block_size in parse_int_list(args.kv_block_sizes):
                for has_num_rejected in (False, True):
                    for block_size_tl in parse_int_list(args.triton_block_sizes):
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
                            old_out = run_kernel(
                                launch_old_kernel,
                                ext,
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            new_out = run_kernel(
                                launch_new_kernel,
                                ext,
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            assert_outputs(case, expected, old_out, new_out)
                            total += 1
                            if args.progress_every and total % args.progress_every == 0:
                                print(f"verified cases={total}")
    return total


def run_benchmark(args: argparse.Namespace, ext, device: torch.device) -> None:
    print("\nCUDA benchmark: kernel-only elapsed time in milliseconds")
    print(
        "batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,"
        "old_avg,old_p50,old_p90,old_p99,"
        "new_avg,new_p50,new_p90,new_p99,speedup_avg"
    )

    for batch_size in parse_int_list(args.bench_batch_sizes):
        for prompt_len in parse_int_list(args.bench_prompt_lens):
            for k in parse_int_list(args.bench_spec_tokens):
                for kv_block_size in parse_int_list(args.bench_kv_block_sizes):
                    for block_size_tl in parse_int_list(args.bench_triton_block_sizes):
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
                            expected = cpu_reference(
                                case,
                                args.parallel_drafting_token_id,
                            )
                            old_out = run_kernel(
                                launch_old_kernel,
                                ext,
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            new_out = run_kernel(
                                launch_new_kernel,
                                ext,
                                case,
                                device,
                                args.parallel_drafting_token_id,
                            )
                            assert_outputs(case, expected, old_out, new_out)

                            old_times = time_kernel_ms(
                                launch_old_kernel,
                                ext,
                                case,
                                device,
                                args.parallel_drafting_token_id,
                                warmup=args.bench_warmup,
                                iters=args.bench_iters,
                            )
                            new_times = time_kernel_ms(
                                launch_new_kernel,
                                ext,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--cases-per-combo", type=int, default=5)
    parser.add_argument("--batch-sizes", default="1,2,8,32,64,128")
    parser.add_argument("--spec-tokens", default="1,2,4,8")
    parser.add_argument("--kv-block-sizes", default="16,32,128")
    parser.add_argument("--triton-block-sizes", default="16,64,128,256")
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-position-base", type=int, default=8192)
    parser.add_argument("--parallel-drafting-token-id", type=int, default=999999)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--verbose-build", action="store_true")

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also run CUDA kernel-only timing comparison after precision checks.",
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

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is available. Please check CUDA/PyTorch setup.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ext = load_cuda_module(args.verbose_build)

    total = run_precision(args, ext, device)
    print(f"PASS CUDA DFlash input-expand precision cases={total}")

    if args.benchmark:
        run_benchmark(args, ext, device)


if __name__ == "__main__":
    main()
