# vLLM-Ascend DFlash Input Expand 二维 Grid 优化说明

本文只说明 DFlash input expand kernel 从 single-grid 改成二维 grid 的优化点。

目标是把当前 NPU 上接近串行的 DFlash 输入准备 kernel，改成按 request 和 token block 并行执行，降低高并发、长输入场景下的 TTFT/TPOT 开销。

## 1. 结论

需要修改两个文件：

```text
vllm_ascend/ops/triton/spec_decode/utils.py
vllm_ascend/spec_decode/dflash_proposer.py
```

修改内容：

| 文件 | 当前行附近 | 修改内容 |
| --- | ---: | --- |
| `ops/triton/spec_decode/utils.py` | 68-136 | 保留旧 single-grid kernel，新增二维 grid kernel `copy_and_expand_dflash_inputs_kernel` |
| `spec_decode/dflash_proposer.py` | 1-12 | 增加 `os`、`triton` import，并同时 import 新旧 kernel |
| `spec_decode/dflash_proposer.py` | 95-120 | 用环境变量选择新二维 kernel 或旧 single-grid kernel |

建议保留旧 kernel 做 A/B 和快速回退，不要直接删除。

## 2. 当前问题

当前 Ascend DFlash 在 `set_inputs_first_pass()` 中调用：

```python
copy_and_expand_dflash_inputs_kernel_single_grid[1,](
    ...
    batch_size=batch_size,
    HAS_NUM_REJECTED=has_num_rejected,
)
```

对应 kernel 位于：

```text
vllm_ascend/ops/triton/spec_decode/utils.py
```

当前约 68-136 行：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(
    ...
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    for req_idx in range(0, batch_size):
        ctx_start = tl.load(query_start_loc_ptr + req_idx)
        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
        num_ctx = ctx_end - ctx_start

        for j in range(0, num_ctx):
            ...

        ...

        for q_idx in range(0, num_query_per_req):
            ...
```

问题在于：

- launch grid 是 `[1,]`，只有一个 Triton program。
- `for req_idx in range(0, batch_size)` 串行扫所有请求。
- `for j in range(0, num_ctx)` 串行扫每个请求的 context tokens。
- 高并发和长输入时，这个 kernel 会被 batch size 和 context length 放大。

它要做的事情本身很适合并行：

```text
每个 request 的每段 token block 独立计算 positions / slot_mapping / input_ids
```

所以可以改成二维 grid：

```text
axis 0: req_idx
axis 1: block_idx inside this request
```

## 3. 上游 vLLM 的实现逻辑

上游 vLLM v0.20.2 已经是二维 grid。

调用侧：

```text
vllm/v1/spec_decode/dflash.py
```

上游约 109-116 行：

```python
max_ctx_per_req = cad.max_query_len
max_tokens_per_req = max_ctx_per_req + num_query_per_req
BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
grid = (batch_size, num_blocks)

has_num_rejected = num_rejected_tokens_gpu is not None
copy_and_expand_dflash_inputs_kernel[grid](
    ...
    BLOCK_SIZE=BLOCK_SIZE,
    HAS_NUM_REJECTED=has_num_rejected,
)
```

kernel 侧：

```text
vllm/v1/spec_decode/utils.py
```

上游约 498-507 行：

```python
req_idx = tl.program_id(axis=0)
block_idx = tl.program_id(axis=1)

ctx_start = tl.load(query_start_loc_ptr + req_idx)
ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
num_ctx = ctx_end - ctx_start
total_tokens = num_ctx + num_query_per_req

j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

也就是说，上游把 “batch 维” 和 “token block 维” 都并行化了。

## 4. 修改点 1：新增二维 grid kernel

文件：

```text
vllm_ascend/ops/triton/spec_decode/utils.py
```

保留原来的：

```python
copy_and_expand_dflash_inputs_kernel_single_grid
```

在它后面新增：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total]
    out_context_positions_ptr,  # [num_context]
    out_query_positions_ptr,  # [num_query_total]
    out_context_slot_mapping_ptr,  # [num_context]
    out_query_slot_mapping_ptr,  # [num_query_total]
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens]
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    num_rejected_tokens_ptr,  # [num_reqs] or null when not padded
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    # axis 0: request id
    # axis 1: token block inside this request
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

    # Context positions come from target model positions.
    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(
        target_positions_ptr + ctx_pos_idx,
        mask=is_ctx,
        other=0,
    )

    # Query positions start from the last valid context position + 1.
    # In padded speculative rounds, rejected tokens should not extend context.
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

    # Map absolute positions to KV-cache slots through block table.
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

    # Query token 0 is bonus token; query tokens 1..K are DFlash mask tokens.
    bonus_token = tl.load(next_token_ids_ptr + req_idx)
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)

    # Sample only mask tokens, not the bonus token.
    is_sample = is_query & (query_off > 0)
    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
    tl.store(
        out_token_indices_ptr + sample_out_idx,
        query_out,
        mask=is_sample,
    )
```

### 修改前后对比

修改前：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(...):
    for req_idx in range(0, batch_size):
        ...
        for j in range(0, num_ctx):
            ...
        ...
        for q_idx in range(0, num_query_per_req):
            ...
```

修改后：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel(...):
    req_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ...
```

并行方式从：

```text
1 个 program 串行处理全部请求和 token
```

变成：

```text
batch_size * num_blocks 个 program 并行处理
```

## 5. 修改点 2：修改 `dflash_proposer.py` import

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

当前约 1-12 行：

```python
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_inputs_kernel_single_grid
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
```

建议改成：

```python
import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    copy_and_expand_dflash_inputs_kernel_single_grid,
)
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
```

为什么：

- `os` 用于环境变量开关。
- `triton` 用于 `next_power_of_2` 和 `cdiv`。
- 同时 import 新旧 kernel，方便 A/B 和回退。

## 6. 修改点 3：修改调用侧

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

当前位置：`set_inputs_first_pass()` 中，当前约 93-120 行：

```python
has_num_rejected = num_rejected_tokens_gpu is not None

copy_and_expand_dflash_inputs_kernel_single_grid[1,](
    # Inputs
    next_token_ids_ptr=next_token_ids,
    target_positions_ptr=target_positions,
    # Outputs
    out_input_ids_ptr=self.input_ids,
    out_context_positions_ptr=self._context_positions_buffer,
    out_query_positions_ptr=self.positions,
    out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
    out_query_slot_mapping_ptr=self._slot_mapping_buffer,
    out_token_indices_ptr=token_indices_to_sample,
    # Block table
    block_table_ptr=cad.block_table_tensor,
    block_table_stride=cad.block_table_tensor.stride(0),
    # Metadata
    query_start_loc_ptr=cad.query_start_loc,
    num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
    # Scalars
    parallel_drafting_token_id=self.parallel_drafting_token_id,
    block_size=self.kernel_block_size,
    num_query_per_req=num_query_per_req,
    num_speculative_tokens=self.num_speculative_tokens,
    total_input_tokens=num_context,
    batch_size=batch_size,
    HAS_NUM_REJECTED=has_num_rejected,
)
```

建议替换为：

```python
has_num_rejected = num_rejected_tokens_gpu is not None

use_parallel_expand = os.getenv(
    "VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND",
    "1",
) == "1"

if use_parallel_expand:
    # Parallelize over request and token block. This avoids the single-grid
    # kernel serializing all requests and context tokens in high-concurrency
    # prefill batches.
    max_ctx_per_req = cad.max_query_len
    max_tokens_per_req = max_ctx_per_req + num_query_per_req
    block_size_tl = min(
        128,
        triton.next_power_of_2(max_tokens_per_req),
    )
    num_blocks = triton.cdiv(max_tokens_per_req, block_size_tl)
    grid = (batch_size, num_blocks)

    copy_and_expand_dflash_inputs_kernel[grid](
        # Inputs
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        # Outputs
        out_input_ids_ptr=self.input_ids,
        out_context_positions_ptr=self._context_positions_buffer,
        out_query_positions_ptr=self.positions,
        out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
        out_query_slot_mapping_ptr=self._slot_mapping_buffer,
        out_token_indices_ptr=token_indices_to_sample,
        # Block table
        block_table_ptr=cad.block_table_tensor,
        block_table_stride=cad.block_table_tensor.stride(0),
        # Metadata
        query_start_loc_ptr=cad.query_start_loc,
        num_rejected_tokens_ptr=(
            num_rejected_tokens_gpu if has_num_rejected else 0
        ),
        # Scalars
        parallel_drafting_token_id=self.parallel_drafting_token_id,
        block_size=self.kernel_block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=self.num_speculative_tokens,
        total_input_tokens=num_context,
        BLOCK_SIZE=block_size_tl,
        HAS_NUM_REJECTED=has_num_rejected,
    )
else:
    copy_and_expand_dflash_inputs_kernel_single_grid[1,](
        # Inputs
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        # Outputs
        out_input_ids_ptr=self.input_ids,
        out_context_positions_ptr=self._context_positions_buffer,
        out_query_positions_ptr=self.positions,
        out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
        out_query_slot_mapping_ptr=self._slot_mapping_buffer,
        out_token_indices_ptr=token_indices_to_sample,
        # Block table
        block_table_ptr=cad.block_table_tensor,
        block_table_stride=cad.block_table_tensor.stride(0),
        # Metadata
        query_start_loc_ptr=cad.query_start_loc,
        num_rejected_tokens_ptr=(
            num_rejected_tokens_gpu if has_num_rejected else 0
        ),
        # Scalars
        parallel_drafting_token_id=self.parallel_drafting_token_id,
        block_size=self.kernel_block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=self.num_speculative_tokens,
        total_input_tokens=num_context,
        batch_size=batch_size,
        HAS_NUM_REJECTED=has_num_rejected,
    )
```

### 修改前后对比

修改前：

```python
copy_and_expand_dflash_inputs_kernel_single_grid[1,](
    ...
    batch_size=batch_size,
    HAS_NUM_REJECTED=has_num_rejected,
)
```

修改后：

```python
max_ctx_per_req = cad.max_query_len
max_tokens_per_req = max_ctx_per_req + num_query_per_req
block_size_tl = min(128, triton.next_power_of_2(max_tokens_per_req))
num_blocks = triton.cdiv(max_tokens_per_req, block_size_tl)
grid = (batch_size, num_blocks)

copy_and_expand_dflash_inputs_kernel[grid](
    ...
    BLOCK_SIZE=block_size_tl,
    HAS_NUM_REJECTED=has_num_rejected,
)
```

## 7. 为什么 `BLOCK_SIZE` 建议先用 128

上游 GPU 使用：

```python
BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
```

Ascend NPU 建议先保守使用：

```python
block_size_tl = min(128, triton.next_power_of_2(max_tokens_per_req))
```

原因：

- NPU Triton kernel 对过大的 block size 可能更敏感。
- `BLOCK_SIZE=128` 通常更稳，先验证正确性和收益。
- 后续可以 profile `64 / 128 / 256`，选择最优值。

建议环境变量扩展：

```python
block_size_tl = int(os.getenv(
    "VLLM_ASCEND_DFLASH_INPUT_EXPAND_BLOCK_SIZE",
    "128",
))
block_size_tl = min(
    block_size_tl,
    triton.next_power_of_2(max_tokens_per_req),
)
```

这个扩展不是必须，第一版可以先固定 128。

## 8. 输出语义必须保持一致

新旧 kernel 的输出必须完全一致：

| 输出 | shape | 语义 |
| --- | --- | --- |
| `out_input_ids` | `[batch_size * (K + 1)]` | 每个请求 `[bonus_token, mask, mask, ...]` |
| `out_context_positions` | `[num_context]` | target context positions |
| `out_query_positions` | `[batch_size * (K + 1)]` | bonus/mask query positions |
| `out_context_slot_mapping` | `[num_context]` | context KV cache slot |
| `out_query_slot_mapping` | `[batch_size * (K + 1)]` | query KV cache slot |
| `out_token_indices` | `[batch_size * K]` | mask token hidden states 的采样位置 |

重点边界：

- `HAS_NUM_REJECTED=True` 时，`last_pos` 必须基于 `valid_ctx_end = ctx_end - num_rejected`。
- `query_off == 0` 是 bonus token，不写 `out_token_indices`。
- `query_off > 0` 是 speculative mask token，需要写 `out_token_indices`。
- `block_num` 要 clamp 到 `block_table_stride - 1`，避免位置接近上限时越界。
- `ctx_pos_idx` 要用 mask，避免 query lanes 读取越界 context positions。

## 9. 图模式影响

这步会影响 DFlash drafter graph capture 到的内容，但不应该破坏图模式。

原因：

- kernel 调用发生在 `set_inputs_first_pass()` 中，在进入 `_run_merged_draft()` 之前。
- 它不是 graph 内部的动态分支。
- 它只是准备 DFlash draft forward 的 `input_ids / positions / slot_mapping / token_indices_to_sample`。
- 输出 buffer 仍然是原来的预分配 buffer：

```python
self.input_ids
self._context_positions_buffer
self.positions
self._context_slot_mapping_buffer
self._slot_mapping_buffer
token_indices_to_sample
```

注意：

- 修改后需要重启进程，让新代码生效。
- 如果旧 graph 已经 capture，热改不会修改旧 graph 行为。
- 建议配合 `VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND=0/1` 做 A/B。

## 10. 启动和回退

默认开启新 kernel：

```bash
export VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND=1
```

回退旧 kernel：

```bash
export VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND=0
```

建议第一轮验证先用：

```bash
export VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND=1
```

如果出现输出不一致、NPU Triton 编译问题或性能反向，立即切回：

```bash
export VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND=0
```

## 11. 正确性验证建议

建议先写一个小测试，让新旧 kernel 输入相同，比较所有输出。

覆盖场景：

```text
batch_size = 1 / 8 / 32 / 128
num_speculative_tokens = 1 / 4 / 8
prompt lengths = uniform / non-uniform
HAS_NUM_REJECTED = False / True
block_size = 当前 kernel_block_size
```

比较输出：

```python
torch.testing.assert_close(out_input_ids_old, out_input_ids_new)
torch.testing.assert_close(out_context_positions_old, out_context_positions_new)
torch.testing.assert_close(out_query_positions_old, out_query_positions_new)
torch.testing.assert_close(out_context_slot_mapping_old, out_context_slot_mapping_new)
torch.testing.assert_close(out_query_slot_mapping_old, out_query_slot_mapping_new)
torch.testing.assert_close(out_token_indices_old, out_token_indices_new)
```

如果 `HAS_NUM_REJECTED=True`，要特别检查：

```text
last_pos = target_positions[ctx_end - num_rejected - 1]
query_pos = last_pos + 1 + query_off
```

## 12. 性能验证建议

建议重点看高并发/长 prompt：

```text
batch_size = 32 / 64 / 128 / 256
prompt_len = 1024 / 4096 / 8192
num_speculative_tokens = 4 / 8
```

记录：

```text
DFlash input expand kernel time
TTFT p50/p90/p99
TPOT / ITL p50/p90/p99
overall tokens/s
```

预期：

- 小 batch、短 prompt：收益可能很小。
- 高并发、长 prompt：input expand kernel 可能有 3-20x kernel 级别收益。
- 端到端收益取决于 input expand 在总耗时中的占比。

### 12.1 NPU Triton 多核缺陷风险

vLLM-Ascend DFlash 引入时的 commit 说明里提到：

```text
The NPU Triton multi-core is faulty.
Currently, only use a single core to process all reqs, which needs to be improved.
```

这也是当前 vllm-ascend 使用 `copy_and_expand_dflash_inputs_kernel_single_grid[1,]` 的核心原因：它不是不知道上游 vLLM 已经有 2-D grid kernel，而是为了绕开当时 NPU Triton 多 program / 多核路径的问题，把所有 request 放进一个 Triton program 内串行处理。

因此，在 NPU 上把 DFlash input expand 改成二维 grid 会有影响：

| 风险 | 说明 |
| --- | --- |
| 正确性风险 | `grid=(batch_size, num_blocks)` 会产生大量 Triton program，可能重新触发当时提到的 NPU Triton 多核缺陷 |
| 稳定性风险 | 如果缺陷仍存在，可能表现为输出不一致、偶发错误、kernel hang、编译失败或不同 batch 下结果不稳定 |
| 性能风险 | 即使输出正确，如果 NPU Triton runtime 对多 program 调度存在退化，也可能没有 CUDA Triton 那样的明显收益 |
| 图模式风险 | 2-D grid 的 `num_blocks` 和 `BLOCK_SIZE` 需要保持 shape 稳定；否则可能影响 ACL graph capture / replay |

所以 2-D grid 在 NPU 上不能直接无条件替换 single-grid。建议实现为可选开关：

```text
默认：继续使用 single-grid
实验：通过环境变量或配置打开 2-D grid
失败：精度或稳定性不过时回退 single-grid
```

推荐门禁：

1. 先跑 `smoke` 精度验证。
2. 再跑 `full` 随机精度验证。
3. 再跑 `long` 覆盖大 batch / 长 prompt。
4. 最后跑 `bench` 看 kernel-only 性能。
5. 如果要进真实推理链路，还需要分别验证 eager、ACL graph capture、ACL graph replay。

NPU 2-D grid 只有在下面条件都满足时才建议进入生产路径：

- `verify_dflash_2d_grid_npu.py` 的 single-grid 和 2-D grid 输出完全一致。
- 多次重复运行结果稳定。
- 大 batch / 长 prompt 下 `speedup_avg > 1` 且波动可接受。
- 图模式下 capture/replay 不失败。
- 真实 DFlash 推理 TTFT/TPOT 有收益，而不只是 micro benchmark 有收益。

## 13. 风险点

| 风险 | 说明 | 处理 |
| --- | --- | --- |
| 输出不一致 | mask 或 index 边界写错 | 用新旧 kernel 对比测试 |
| `BLOCK_SIZE` 不合适 | NPU 上过大可能性能反向 | 先用 128，再 profile 64/256 |
| graph 已 capture 旧逻辑 | 热改不生效 | 重启服务 |
| shape 特殊场景 | 非均匀 prompt、rejected tokens | 覆盖测试 |
| block table 越界 | position 接近上限 | 保留 `tl.minimum(block_num, block_table_stride - 1)` |

## 14. 推荐落地顺序

1. 在 `utils.py` 新增二维 kernel，保留旧 kernel。
2. 在 `dflash_proposer.py` 增加 import。
3. 在调用侧加 `VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND` 开关。
4. 做新旧 kernel 输出一致性测试。
5. 小 batch 跑通。
6. 高并发长 prompt 压测。
7. 根据 profile 选择 `BLOCK_SIZE=64/128/256`。
8. 稳定后再考虑默认开启。

## 15. 无 NPU 环境下的初步验证结果

当前 workspace 没有 NPU 环境，因此先做了 CPU 语义模型验证：

- 把旧 `copy_and_expand_dflash_inputs_kernel_single_grid` 翻译成 Python 版本。
- 把新增二维 `copy_and_expand_dflash_inputs_kernel` 翻译成 Python 版本。
- 随机构造相同输入，比较两个版本的所有输出。

覆盖组合：

```text
batch_size = 1 / 2 / 8 / 32 / 64 / 128
num_speculative_tokens = 1 / 2 / 4 / 8
block_size = 16 / 32 / 128
HAS_NUM_REJECTED = False / True
BLOCK_SIZE = 1 / 7 / 16 / 64 / 128 / 256
每组随机 10 个 prompt length / position / block_table case
```

验证输出：

```text
PASS randomized semantic equivalence cases=8640, failures=0
```

比较字段：

```text
out_input_ids
out_context_positions
out_query_positions
out_context_slot_mapping
out_query_slot_mapping
out_token_indices
```

额外检查了一个非均匀 prompt + rejected tokens case：

```text
sample query_start_loc: [0, 58, 71, 212, 338]
sample num_rejected: [0, 1, 1, 4]
sample out_input_ids:
[154795, 999999, 999999, 999999, 999999,
 110605, 999999, 999999, 999999, 999999,
 8332, 999999, 999999, 999999, 999999,
 7812, 999999, 999999, 999999, 999999]
sample out_token_indices:
[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
```

这个验证能说明：

- 二维 grid 的 `req_idx / block_idx / j` 索引拆分和旧循环语义一致。
- context/query 两类输出 buffer 的写入位置一致。
- bonus token 和 mask token 的 `input_ids` 语义一致。
- `out_token_indices` 跳过 bonus token 的逻辑一致。
- `HAS_NUM_REJECTED=True` 时，`last_pos` 的计算语义一致。

限制：

- 这不是 NPU Triton 编译验证。
- 没有验证 NPU kernel launch、Triton lowering、性能和 graph capture 行为。
- CPU 验证中输入保证 block table 覆盖 position 范围；真实部署仍需验证极限长度和 block table 边界。

下一步仍需在 NPU 环境做：

```text
1. 新旧 kernel 输出一致性测试
2. NPU Triton 编译/运行测试
3. graph mode capture/replay 测试
4. 高并发 TTFT/TPOT profile
```

## 16. NPU 精度验证脚本

已提供一个可直接运行的 NPU 精度验证脚本：

```text
verify_dflash_2d_grid_npu.py
run_verify_dflash_2d_grid_npu.sh
```

这个脚本会：

- 在脚本内定义旧 single-grid kernel。
- 在脚本内定义新增二维 grid kernel。
- 随机构造 DFlash input expand 输入。
- 跑 CPU reference。
- 跑旧 NPU kernel。
- 跑新二维 NPU kernel。
- 比较三者输出是否完全一致。

比较字段：

```text
input_ids
context_positions
query_positions
context_slot_mapping
query_slot_mapping
token_indices
```

### 16.1 环境要求

需要在 Ascend NPU 环境中运行，并且当前 Python 环境能 import：

```python
import torch
import torch_npu
from vllm.triton_utils import tl, triton
```

建议在 vLLM-Ascend 容器或已安装 vLLM-Ascend 的环境中运行。

如果直接在源码目录运行，需要确保 `PYTHONPATH` 包含 vLLM 和 vLLM-Ascend：

```bash
export PYTHONPATH=/path/to/vllm-ascend:/path/to/vllm:${PYTHONPATH:-}
```

### 16.1.1 WSL/Linux 启动脚本

如果在 WSL 里访问当前 Windows workspace，路径通常是：

```bash
cd /mnt/d/workspace/speculative
```

如果 workspace 已经复制到 Linux 文件系统，进入实际目录即可：

```bash
cd /path/to/speculative
```

第一次可以给脚本加执行权限：

```bash
chmod +x run_verify_dflash_2d_grid_cpu.sh
chmod +x run_verify_dflash_2d_grid_npu.sh
chmod +x run_verify_dflash_2d_grid_cuda.sh
chmod +x run_verify_dflash_2d_grid_triton_cuda.sh
```

也可以不加执行权限，直接用 `bash` 启动：

```bash
bash run_verify_dflash_2d_grid_npu.sh smoke
```

NPU 启动脚本会自动设置：

```bash
VLLM_ASCEND_PATH=${PWD}/vllm-ascend-0.20.2rc1
VLLM_PATH=${PWD}/vllm-v0.20.2
PYTHONPATH=${VLLM_ASCEND_PATH}:${VLLM_PATH}:${PYTHONPATH:-}
```

如果你的 vLLM / vLLM-Ascend 源码路径不同，可以手动覆盖：

```bash
VLLM_ASCEND_PATH=/path/to/vllm-ascend \
VLLM_PATH=/path/to/vllm \
bash run_verify_dflash_2d_grid_npu.sh smoke
```

常用环境变量：

```bash
PYTHON_BIN=python3
DEVICE=npu:0
```

### 16.1.2 精度测试 / 性能测试总览

这几个启动脚本的 mode 约定是一致的：

| mode | 用途 | 是否计时 | 适合场景 |
| --- | --- | --- | --- |
| `smoke` | 小规模精度验证 | 否 | 先确认环境、编译和 kernel 输出正确 |
| `full` | 默认参数组合精度验证 | 否 | 较完整的随机 case 覆盖 |
| `long` | 大 batch / 长 prompt 精度验证 | 否 | 接近高并发长输入场景，但不计时 |
| `bench` | 精度验证通过后追加 kernel-only 计时 | 是 | 对比 single-grid 与 2-D grid 性能 |

精度测试只看输出是否和 CPU reference 一致。性能测试的 `bench` 会先跑一组小规模精度验证，然后每个 benchmark case 也会再次确认 single-grid、2-D grid 和 CPU reference 输出一致，最后才统计耗时。

推荐按下面顺序跑。

NPU 精度测试：

```bash
cd /mnt/d/workspace/speculative
bash run_verify_dflash_2d_grid_npu.sh smoke
bash run_verify_dflash_2d_grid_npu.sh full
```

NPU 大 batch / 长 prompt 精度测试：

```bash
BATCH_SIZES=128,256 \
MAX_PROMPT_LEN=4096 \
SPEC_TOKENS=15 \
KV_BLOCK_SIZES=128 \
TRITON_BLOCK_SIZES=128 \
bash run_verify_dflash_2d_grid_npu.sh long
```

NPU 性能测试：

```bash
BENCH_WARMUP=5 \
BENCH_ITERS=20 \
BENCH_BATCH_SIZES=128,256 \
BENCH_PROMPT_LENS=4096 \
BENCH_SPEC_TOKENS=15 \
BENCH_KV_BLOCK_SIZES=128 \
BENCH_TRITON_BLOCK_SIZES=128 \
bash run_verify_dflash_2d_grid_npu.sh bench
```

CUDA Triton 精度测试，用于在 GPU 上对比“Ascend single-grid 风格”和“上游 vLLM 2-D grid 风格”：

```bash
bash run_verify_dflash_2d_grid_triton_cuda.sh smoke
```

CUDA Triton 性能测试：

```bash
BENCH_WARMUP=5 \
BENCH_ITERS=20 \
BENCH_BATCH_SIZES=256 \
BENCH_PROMPT_LENS=4096 \
BENCH_SPEC_TOKENS=15 \
BENCH_KV_BLOCK_SIZES=128 \
BENCH_TRITON_BLOCK_SIZES=128 \
bash run_verify_dflash_2d_grid_triton_cuda.sh bench
```

CPU 精度测试只用于验证索引语义，不用于性能结论：

```bash
bash run_verify_dflash_2d_grid_cpu.sh smoke
bash run_verify_dflash_2d_grid_cpu.sh full
```

输出字段里的 `speedup_avg` 都是：

```text
speedup_avg = single_avg / grid2d_avg
```

或在旧字段名里：

```text
speedup_avg = old_avg / new_avg
```

因此 `speedup_avg > 1` 表示 2-D grid 更快。所有 benchmark 都只统计 input expand kernel 自身耗时，不代表端到端 TTFT/TPOT。

### 16.1.3 可选参数说明

启动脚本本身只接收一个位置参数：

```bash
bash run_verify_dflash_2d_grid_triton_cuda.sh smoke
bash run_verify_dflash_2d_grid_triton_cuda.sh full
bash run_verify_dflash_2d_grid_triton_cuda.sh long
bash run_verify_dflash_2d_grid_triton_cuda.sh bench
```

`smoke/full/long/bench` 的含义见上一节。除 mode 以外，启动脚本主要通过环境变量传参。

通用环境变量：

| 环境变量 | 适用脚本 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `PYTHON_BIN` | 全部 | `python3` | 指定 Python 可执行文件 |
| `DEVICE` | NPU/CUDA/CUDA Triton | NPU 为 `npu:0`，CUDA 为 `cuda:0` | 指定运行设备 |
| `VLLM_ASCEND_PATH` | NPU | `${PWD}/vllm-ascend-0.20.2rc1` | vLLM-Ascend 源码路径 |
| `VLLM_PATH` | NPU | `${PWD}/vllm-v0.20.2` | 上游 vLLM 源码路径 |
| `TRITON_CACHE_DIR` | CUDA Triton | Triton 默认 cache | 指定 Triton 编译缓存目录 |
| `TORCH_CUDA_ARCH_LIST` | CUDA C++ extension | PyTorch 自动推断 | CUDA 13 / RTX 50 系列建议设为 `12.0` |
| `MAX_JOBS` | CUDA C++ extension | 构建工具默认值 | 限制 extension 编译并发 |

`long` 模式环境变量：

| 环境变量 | 默认值 | 对应 Python 参数 | 说明 |
| --- | --- | --- | --- |
| `CASES_PER_COMBO` | `3` | `--cases-per-combo` | 每个参数组合随机 case 数 |
| `BATCH_SIZES` | `64,128,256` | `--batch-sizes` | batch size 列表，逗号分隔 |
| `SPEC_TOKENS` | `4,8` | `--spec-tokens` | speculative token 数列表 |
| `KV_BLOCK_SIZES` | `128` | `--kv-block-sizes` | KV cache block size 列表 |
| `TRITON_BLOCK_SIZES` | `64,128,256` | `--triton-block-sizes` | 2-D grid 的 token tile size 列表 |
| `MAX_PROMPT_LEN` | `4096` | `--max-prompt-len` | 随机 prompt length 上限 |
| `MAX_POSITION_BASE` | `65536` | `--max-position-base` | 随机 absolute position base 上限 |

`bench` 模式环境变量：

| 环境变量 | NPU/CUDA Triton 默认值 | CPU 默认值 | 对应 Python 参数 | 说明 |
| --- | --- | --- | --- | --- |
| `BENCH_WARMUP` | `10` | `3` | `--bench-warmup` | 计时前 warmup 次数 |
| `BENCH_ITERS` | `50` | `10` | `--bench-iters` | 计时迭代次数 |
| `BENCH_BATCH_SIZES` | `32,64,128` | `32,64,128` | `--bench-batch-sizes` | benchmark batch size 列表 |
| `BENCH_PROMPT_LENS` | `256,1024,4096` | `256,1024` | `--bench-prompt-lens` | 每个 request 的固定 context length 列表 |
| `BENCH_SPEC_TOKENS` | `4,8` | `4,8` | `--bench-spec-tokens` | benchmark speculative token 数列表 |
| `BENCH_KV_BLOCK_SIZES` | `128` | `128` | `--bench-kv-block-sizes` | benchmark KV cache block size 列表 |
| `BENCH_TRITON_BLOCK_SIZES` | `64,128,256` | `64,128,256` | `--bench-triton-block-sizes` | benchmark 2-D grid token tile size 列表 |

如果不通过启动脚本，而是直接运行 Python，可以使用下面这些参数。NPU、CUDA Triton、CUDA C++ extension 的参数基本一致；CPU 脚本没有 `--device`。

精度验证参数：

| Python 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | NPU 为 `npu:0`，CUDA 为 `cuda:0` | 运行设备 |
| `--seed` | `20260607` | 随机种子 |
| `--cases-per-combo` | NPU/CUDA 为 `5`，CPU 为 `10` | 每个参数组合随机 case 数 |
| `--batch-sizes` | `1,2,8,32,64,128` | batch size 列表 |
| `--spec-tokens` | `1,2,4,8` | speculative token 数列表 |
| `--kv-block-sizes` | `16,32,128` | KV cache block size 列表 |
| `--triton-block-sizes` | NPU/CUDA 为 `16,64,128,256`，CPU 为 `1,7,16,64,128,256` | 2-D grid token tile size 列表 |
| `--max-prompt-len` | `256` | 随机 prompt length 上限 |
| `--max-position-base` | `8192` | 随机 absolute position base 上限 |
| `--parallel-drafting-token-id` | `999999` | DFlash mask token / parallel drafting token 的占位 ID |
| `--progress-every` | `0` | 每验证多少个 case 打印一次进度；`0` 表示不打印 |

性能测试参数：

| Python 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--benchmark` | 关闭 | 开启 kernel-only 耗时统计 |
| `--bench-warmup` | NPU/CUDA 为 `10`，CPU 为 `3` | benchmark warmup 次数 |
| `--bench-iters` | NPU/CUDA 为 `50`，CPU 为 `10` | benchmark 计时迭代次数 |
| `--bench-batch-sizes` | `32,64,128` | benchmark batch size 列表 |
| `--bench-prompt-lens` | NPU/CUDA 为 `256,1024,4096`，CPU 为 `256,1024` | benchmark prompt length 列表 |
| `--bench-spec-tokens` | `4,8` | benchmark speculative token 数列表 |
| `--bench-kv-block-sizes` | `128` | benchmark KV cache block size 列表 |
| `--bench-triton-block-sizes` | `64,128,256` | benchmark 2-D grid token tile size 列表 |
| `--bench-max-position-base` | `65536` | benchmark absolute position base 上限 |

注意：启动脚本的 `bench` mode 会先固定跑一组小规模精度验证参数，然后再使用 `BENCH_*` 或 `--bench-*` 对 benchmark case 计时。因此 `BENCH_BATCH_SIZES` 等参数只控制性能测试部分，不会改变前置 smoke 精度验证部分。

### 16.2 快速 smoke test

先跑一个小规模验证：

```bash
cd /mnt/d/workspace/speculative

bash run_verify_dflash_2d_grid_npu.sh smoke
```

等价的直接 Python 命令是：

```bash
python3 verify_dflash_2d_grid_npu.py \
  --device npu:0 \
  --cases-per-combo 1 \
  --batch-sizes 1,8 \
  --spec-tokens 1,4 \
  --kv-block-sizes 16,128 \
  --triton-block-sizes 64,128 \
  --max-prompt-len 64
```

成功时会输出类似：

```text
PASS NPU DFlash input-expand precision cases=32
```

### 16.3 完整随机验证

跑默认组合：

```bash
cd /mnt/d/workspace/speculative

bash run_verify_dflash_2d_grid_npu.sh full
```

默认覆盖：

```text
batch_size = 1,2,8,32,64,128
num_speculative_tokens = 1,2,4,8
kv block_size = 16,32,128
HAS_NUM_REJECTED = False / True
Triton BLOCK_SIZE = 16,64,128,256
每组随机 5 个 case
```

默认总 case 数：

```text
6 * 4 * 3 * 2 * 4 * 5 = 2880
```

成功时会输出：

```text
PASS NPU DFlash input-expand precision cases=2880
```

### 16.4 更大 batch / 更长 prompt 验证

如果想更接近高并发长输入：

```bash
bash run_verify_dflash_2d_grid_npu.sh long
```

注意：

- 这个命令会分配更大的 `target_positions`、`block_table` 和输出 buffer。
- 如果显存紧张，先降低 `--batch-sizes` 或 `--max-prompt-len`。

### 16.5 精度验证 + 耗时对比

脚本支持在精度验证通过后继续做 kernel-only 耗时对比：

```bash
bash run_verify_dflash_2d_grid_npu.sh bench
```

说明：

- `--benchmark` 会在精度验证完成后运行耗时对比。
- 计时使用 `torch.npu.Event(enable_timing=True)`。
- 计时只包 kernel launch，不包含 CPU 构造输入、H2D 拷贝和输出分配。
- 每个 benchmark case 会先再次验证 CPU reference、旧 kernel、新 kernel 三者输出一致，再计时。

输出格式是 CSV 风格：

```text
Benchmark: kernel-only elapsed time in milliseconds
batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,old_avg,old_p50,old_p90,old_p99,new_avg,new_p50,new_p90,new_p99,speedup_avg
64,1024,4,128,128,0,12.3456,12.1000,12.9000,13.4000,1.2345,1.2000,1.3000,1.5000,10.00
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `batch` | benchmark batch size |
| `prompt_len` | 每个请求的 context length |
| `k` | `num_speculative_tokens` |
| `kv_block` | KV cache block size |
| `BLOCK_SIZE` | 二维 Triton kernel 的 token block size |
| `rejected` | 是否模拟 `HAS_NUM_REJECTED` |
| `old_*` | 旧 single-grid kernel 耗时，单位 ms |
| `new_*` | 新二维 grid kernel 耗时，单位 ms |
| `speedup_avg` | `old_avg / new_avg` |

如果只想快速看一个场景：

```bash
BENCH_WARMUP=5 \
BENCH_ITERS=20 \
BENCH_BATCH_SIZES=128 \
BENCH_PROMPT_LENS=4096 \
BENCH_SPEC_TOKENS=4 \
BENCH_KV_BLOCK_SIZES=128 \
BENCH_TRITON_BLOCK_SIZES=128 \
bash run_verify_dflash_2d_grid_npu.sh bench
```

注意：

- benchmark 输出的是 input expand kernel 自身耗时，不是端到端 TTFT/TPOT。
- 第一次运行可能包含 Triton 编译开销，脚本使用 warmup 尽量避开，但建议同一命令跑两次看稳定值。
- 如果旧 single-grid kernel 在大 batch/长 prompt 下非常慢，可以先降低 `--bench-iters`。

### 16.6 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `npu:0` | NPU 设备 |
| `--seed` | `20260607` | 随机种子 |
| `--cases-per-combo` | `5` | 每个参数组合随机 case 数 |
| `--batch-sizes` | `1,2,8,32,64,128` | batch size 列表 |
| `--spec-tokens` | `1,2,4,8` | `num_speculative_tokens` 列表 |
| `--kv-block-sizes` | `16,32,128` | KV cache block size 列表 |
| `--triton-block-sizes` | `16,64,128,256` | 二维 kernel 的 `BLOCK_SIZE` 列表 |
| `--max-prompt-len` | `256` | 随机 prompt length 上限 |
| `--max-position-base` | `8192` | 随机 absolute position base 上限 |
| `--parallel-drafting-token-id` | `999999` | DFlash mask token id |
| `--benchmark` | 关闭 | 精度验证后追加 kernel 耗时对比 |
| `--bench-warmup` | `10` | benchmark warmup 次数 |
| `--bench-iters` | `50` | benchmark 计时迭代次数 |
| `--bench-batch-sizes` | `32,64,128` | benchmark batch size 列表 |
| `--bench-prompt-lens` | `256,1024,4096` | benchmark prompt length 列表 |
| `--bench-spec-tokens` | `4,8` | benchmark speculative token 数列表 |
| `--bench-kv-block-sizes` | `128` | benchmark KV block size 列表 |
| `--bench-triton-block-sizes` | `64,128,256` | benchmark 二维 kernel `BLOCK_SIZE` 列表 |
| `--bench-max-position-base` | `65536` | benchmark absolute position base 上限 |

### 16.7 失败时如何看

脚本失败时会抛出 `AssertionError`，包含：

```text
old kernel mismatch
new kernel mismatch
old/new mismatch
```

含义：

| 错误 | 说明 |
| --- | --- |
| `old kernel mismatch` | 旧 kernel 和 CPU reference 不一致，说明测试输入或旧 kernel 运行本身有问题 |
| `new kernel mismatch` | 新二维 kernel 和 CPU reference 不一致，优先检查 mask/index 边界 |
| `old/new mismatch` | 新旧 NPU kernel 结果不一致 |

常见排查方向：

- `HAS_NUM_REJECTED=True` 时 `valid_ctx_end` 是否正确。
- `query_off == 0` 是否只写 bonus token。
- `query_off > 0` 是否正确写 `token_indices`。
- `block_num` 是否越界。
- `BLOCK_SIZE` 是否触发 NPU Triton 编译问题。

## 17. CPU 语义验证脚本

如果当前机器没有 Ascend NPU，也可以先运行 CPU 语义验证脚本：

```text
verify_dflash_2d_grid_cpu.py
run_verify_dflash_2d_grid_cpu.sh
```

这个脚本不依赖：

```text
torch_npu
Triton
vLLM
vLLM-Ascend
```

它只用 Python 标准库，把旧 single-grid kernel 和新二维 grid kernel 都翻译成 Python 循环，然后比较两者输出。

### 17.1 默认验证

在当前 workspace 直接运行：

```bash
cd /mnt/d/workspace/speculative

bash run_verify_dflash_2d_grid_cpu.sh full
```

当前本机已验证通过：

```text
PASS CPU DFlash input-expand semantic cases=8640
```

默认覆盖：

```text
batch_size = 1 / 2 / 8 / 32 / 64 / 128
num_speculative_tokens = 1 / 2 / 4 / 8
kv block_size = 16 / 32 / 128
HAS_NUM_REJECTED = False / True
Python BLOCK_SIZE = 1 / 7 / 16 / 64 / 128 / 256
每组随机 10 个 case
```

### 17.2 快速验证

如果只想快速跑通：

```bash
bash run_verify_dflash_2d_grid_cpu.sh smoke
```

等价的直接 Python 命令：

```bash
python3 verify_dflash_2d_grid_cpu.py \
  --cases-per-combo 1 \
  --batch-sizes 1,8 \
  --spec-tokens 1,4 \
  --kv-block-sizes 16,128 \
  --triton-block-sizes 64,128 \
  --max-prompt-len 64
```

### 17.3 CPU 耗时对比

CPU 脚本也支持 `--benchmark`：

```bash
bash run_verify_dflash_2d_grid_cpu.sh bench
```

如果想跑更小的 smoke benchmark：

```bash
BENCH_WARMUP=1 \
BENCH_ITERS=2 \
BENCH_BATCH_SIZES=8 \
BENCH_PROMPT_LENS=16 \
BENCH_SPEC_TOKENS=2 \
BENCH_KV_BLOCK_SIZES=128 \
BENCH_TRITON_BLOCK_SIZES=128 \
bash run_verify_dflash_2d_grid_cpu.sh bench
```

本机 smoke test 输出：

```text
PASS CPU DFlash input-expand semantic cases=2

CPU benchmark: Python-loop elapsed time in milliseconds
batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,old_avg,old_p50,old_p90,old_p99,new_avg,new_p50,new_p90,new_p99,speedup_avg
8,16,2,128,128,0,0.0173,0.0173,0.0174,0.0174,0.0823,0.0823,0.0825,0.0825,0.21
8,16,2,128,128,1,0.0161,0.0161,0.0162,0.0162,0.0854,0.0854,0.0878,0.0878,0.19
```

注意：

- CPU benchmark 只是 Python 循环耗时，不代表 NPU kernel 性能。
- CPU 版二维 grid 会因为模拟每个 lane 的循环，可能比旧单循环更慢，这是正常的。
- `speedup_avg = old_avg / new_avg`；大于 `1` 表示新实现更快，小于 `1` 表示新实现更慢。因此 CPU 上 `0.32` 表示 Python 模拟版新路径约为旧路径的 `3.1x` 耗时。
- CPU 脚本的价值是验证索引和输出语义，不是验证并行加速比。

### 17.4 CPU 脚本参数

主要参数和 NPU 脚本保持一致：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed` | `20260607` | 随机种子 |
| `--cases-per-combo` | `10` | 每个参数组合随机 case 数 |
| `--batch-sizes` | `1,2,8,32,64,128` | batch size 列表 |
| `--spec-tokens` | `1,2,4,8` | `num_speculative_tokens` 列表 |
| `--kv-block-sizes` | `16,32,128` | KV cache block size 列表 |
| `--triton-block-sizes` | `1,7,16,64,128,256` | 模拟二维 kernel 的 `BLOCK_SIZE` 列表 |
| `--max-prompt-len` | `256` | 随机 prompt length 上限 |
| `--max-position-base` | `8192` | 随机 absolute position base 上限 |
| `--benchmark` | 关闭 | 追加 CPU Python-loop 耗时对比 |

## 18. CUDA 环境模拟 single-grid vs 2-D grid

如果机器有 CUDA 环境，可以用 CUDA 自定义 kernel 做更接近硬件并发的 micro benchmark：

```text
verify_dflash_2d_grid_cuda.py
run_verify_dflash_2d_grid_cuda.sh
```

CUDA 版会编译两个 kernel：

- `single_grid_kernel`：`<<<1, 1>>>`，一个 CUDA thread 串行遍历所有 request/context/query，模拟当前 single-grid 串行路径。
- `two_d_grid_kernel`：`<<<dim3(batch, num_blocks), BLOCK_SIZE>>>`，每个 request/token tile 一个 CUDA block，block 内线程并行处理 token lane，模拟二维 grid 拆分。

这个脚本仍然是 input-expand micro benchmark，不是完整 vLLM 推理性能测试；但它比 CPU Python-loop 更能体现二维 grid 把工作拆给硬件并行单元后的趋势。

### 18.1 环境要求

需要：

- Linux/WSL CUDA 环境。
- 可用的 `nvcc`。
- PyTorch CUDA 版。
- `torch.utils.cpp_extension` 能正常编译 CUDA extension。

首次运行会编译 extension，通常会慢一些；编译产物会缓存到 PyTorch extension cache，例如：

```text
~/.cache/torch_extensions
```

如果你的 GPU 架构没有被 PyTorch 自动识别，或者 CUDA 13 编译时报
`Unsupported gpu architecture 'compute_70'`，可以显式设置目标架构：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

常见取值：

| GPU | `TORCH_CUDA_ARCH_LIST` |
| --- | --- |
| RTX 50 系列 / Blackwell | `12.0` |
| RTX 4090 / Ada | `8.9` |
| A100 / Ampere | `8.0` |

也可以限制编译并发：

```bash
export MAX_JOBS=4
```

### 18.2 快速验证

```bash
cd /mnt/d/workspace/speculative
bash run_verify_dflash_2d_grid_cuda.sh smoke
```

等价的直接 Python 命令：

```bash
python3 verify_dflash_2d_grid_cuda.py \
  --device cuda:0 \
  --cases-per-combo 1 \
  --batch-sizes 1,8 \
  --spec-tokens 1,4 \
  --kv-block-sizes 16,128 \
  --triton-block-sizes 64,128 \
  --max-prompt-len 64
```

期望输出类似：

```text
PASS CUDA DFlash input-expand precision cases=32
```

### 18.3 大 batch/长 prompt benchmark

可以直接跑默认 benchmark：

```bash
bash run_verify_dflash_2d_grid_cuda.sh bench
```

也可以跑和前面 CPU 例子相同的参数：

```bash
BENCH_WARMUP=1 \
BENCH_ITERS=2 \
BENCH_BATCH_SIZES=128 \
BENCH_PROMPT_LENS=4096 \
BENCH_SPEC_TOKENS=15 \
BENCH_KV_BLOCK_SIZES=128 \
BENCH_TRITON_BLOCK_SIZES=128 \
bash run_verify_dflash_2d_grid_cuda.sh bench
```

输出格式：

```text
CUDA benchmark: kernel-only elapsed time in milliseconds
batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,old_avg,old_p50,old_p90,old_p99,new_avg,new_p50,new_p90,new_p99,speedup_avg
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `old_*` | `single_grid_kernel` 的 kernel-only 耗时 |
| `new_*` | `two_d_grid_kernel` 的 kernel-only 耗时 |
| `speedup_avg` | `old_avg / new_avg` |

解读：

- `speedup_avg > 1`：二维 grid 更快。
- `speedup_avg < 1`：二维 grid 更慢。
- CUDA 版如果在 high batch/long prompt 下出现明显 `speedup_avg > 1`，说明该输入展开逻辑确实具备硬件并行拆分收益。

### 18.4 和 CPU/NPU 结果的关系

三类脚本的定位不同：

| 脚本 | 用途 | 性能可信度 |
| --- | --- | --- |
| `verify_dflash_2d_grid_cpu.py` | 纯 Python 语义验证 | 不用于性能判断 |
| `verify_dflash_2d_grid_cuda.py` | CUDA hardware-concurrency micro benchmark | 可观察 GPU 并行趋势 |
| `verify_dflash_2d_grid_npu.py` | Ascend Triton/NPU kernel 验证 | 最接近目标优化效果 |

CUDA benchmark 能回答“二维 grid 拆分在 GPU 并行硬件上是否有收益”；NPU 上是否也有同等收益，还要看 Ascend Triton 编译、调度、访存和 ACL graph 条件。

### 18.5 关键代码

CUDA 旧路径：

```cpp
single_grid_kernel<<<1, 1>>>(...)
```

对应脚本中的：

```cpp
for (int req_idx = 0; req_idx < batch_size; ++req_idx) {
  for (int j = 0; j < num_ctx; ++j) {
    ...
  }
}
```

CUDA 二维路径：

```cpp
dim3 grid(batch_size, num_blocks);
two_d_grid_kernel<<<grid, block_size_tl>>>(...)
```

对应脚本中的：

```cpp
int req_idx = blockIdx.x;
int block_idx = blockIdx.y;
int lane = threadIdx.x;
int j = block_idx * blockDim.x + lane;
```

这里的 `req_idx/block_idx/lane` 对应前面 Ascend Triton 方案里的：

```python
req_idx = tl.program_id(axis=0)
block_idx = tl.program_id(axis=1)
j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

## 19. CUDA Triton 版 single-grid vs 上游 2-D grid

如果希望对比更贴近 vLLM/Triton 代码路径，可以使用：

```text
verify_dflash_2d_grid_triton_cuda.py
run_verify_dflash_2d_grid_triton_cuda.sh
```

这个脚本和上一节 CUDA C++ extension 版不同：

- `copy_and_expand_dflash_inputs_kernel_2d` 直接按上游 vLLM `vllm/v1/spec_decode/utils.py` 的 2-D Triton kernel 结构实现。
- `copy_and_expand_dflash_inputs_kernel_single_grid` 按 vllm-ascend `vllm_ascend/ops/triton/spec_decode/utils.py` 的 single-grid kernel 结构实现。
- 两者都通过 CUDA Triton JIT 在 GPU 上运行，避免 C++ extension 对 CUDA dev header 的额外依赖。

### 19.1 快速验证

```bash
cd /mnt/d/workspace/speculative
bash run_verify_dflash_2d_grid_triton_cuda.sh smoke
```

期望输出：

```text
PASS CUDA Triton DFlash input-expand precision cases=32
```

### 19.2 benchmark

默认 benchmark：

```bash
bash run_verify_dflash_2d_grid_triton_cuda.sh bench
```

复现 high batch/long prompt case：

```bash
BENCH_WARMUP=5 \
BENCH_ITERS=20 \
BENCH_BATCH_SIZES=256 \
BENCH_PROMPT_LENS=4096 \
BENCH_SPEC_TOKENS=15 \
BENCH_KV_BLOCK_SIZES=128 \
BENCH_TRITON_BLOCK_SIZES=128 \
bash run_verify_dflash_2d_grid_triton_cuda.sh bench
```

输出格式：

```text
CUDA Triton benchmark: kernel-only elapsed time in milliseconds
batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,single_avg,single_p50,single_p90,single_p99,grid2d_avg,grid2d_p50,grid2d_p90,grid2d_p99,speedup_avg
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `single_*` | vllm-ascend single-grid 风格 Triton kernel 耗时 |
| `grid2d_*` | 上游 vLLM 2-D grid 风格 Triton kernel 耗时 |
| `speedup_avg` | `single_avg / grid2d_avg` |

### 19.3 本机 Docker 验证结果

在 `optimistic_galileo` 容器中，环境为：

```text
torch 2.10.0+cu130
triton 3.6.0
GPU NVIDIA GeForce RTX 5070
```

smoke test：

```text
PASS CUDA Triton DFlash input-expand precision cases=32
```

`batch=128, prompt_len=4096, k=15, kv_block=128, BLOCK_SIZE=128, warmup=5, iters=20`：

```text
batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,single_avg,grid2d_avg,speedup_avg
128,4096,15,128,128,0,45.4395,0.0582,780.47
128,4096,15,128,128,1,44.9154,0.0510,881.39
```

`batch=256, prompt_len=4096, k=15, kv_block=128, BLOCK_SIZE=128, warmup=5, iters=20`：

```text
batch,prompt_len,k,kv_block,BLOCK_SIZE,rejected,single_avg,grid2d_avg,speedup_avg
256,4096,15,128,128,0,84.9225,0.0424,2003.80
256,4096,15,128,128,1,86.0082,0.0693,1241.69
```

这个结果说明：在 CUDA Triton 上，当前 single-grid 结构确实会把 `batch * prompt_len` 的 input expand 工作压到一个 program 的串行循环里；上游 2-D grid 把 request 和 token tile 拆开后，可以显著提高硬件并行度。

### 19.4 大 batch 验证数据溢出说明

旧验证脚本里为了区分不同 request 的 fake `block_id`，曾使用：

```python
block_id = req_idx * 100000 + block_idx
```

当 `batch=256, kv_block=128` 时，最大 `slot_mapping` 约为：

```text
255 * 100000 * 128 = 3,264,000,000
```

这超过了 `int32` 上限 `2,147,483,647`，因此 CPU reference 会先报：

```text
RuntimeError: value cannot be converted to type int without overflow
```

这不是 Triton 2-D kernel 的正确性问题，而是 synthetic block table 的造数方式过大。现在验证脚本统一改为紧凑 fake block id：

```python
block_id = req_idx * max_blocks + block_idx
```

这样仍然能保证不同 request 的 block id 不重叠，同时不会在大 batch benchmark 下撑爆 `int32 slot_mapping`。
