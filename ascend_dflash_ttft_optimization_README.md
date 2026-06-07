# vLLM-Ascend DFlash 首 Token 时延优化方向

本文整理 `vllm-ascend v0.20.2rc1` 在 NPU 上启用 DFlash speculative decoding 后，高并发首 token 时延明显变慢的代码原因和优化方向。

相关源码基于当前 workspace：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1
D:\workspace\speculative\vllm-v0.20.2
```

## 1. 结论

DFlash 会降低后续 decode 轮数，但在首 token 返回前会额外执行 drafter first pass。高并发时，首 token 时延不仅包含 target prefill，还包含：

- target hidden states 提取和可能的 aux hidden states 拼接。
- DFlash input/position/slot mapping 构造。
- target hidden states 到 DFlash hidden buffer 的拷贝。
- DFlash context K/V 预计算并写入 draft KV cache。
- draft model 对 `batch_size * (K + 1)` query tokens 的 forward。
- draft token ids 拷贝和 KV connector finalize。

因此高并发下 TTFT 变慢是符合代码链路的；如果变慢非常多，优先怀疑 input expand kernel 串行化、DFlash context KV 预计算、hidden states 拷贝、graph padding 和首轮同步执行 draft。

## 2. GPU vs NPU 差异对比

GPU 上启用 DFlash 也会在首 token 前多做 drafter first pass，但通常不会慢非常多。主要原因不是 GPU 少做了这条链路，而是上游 GPU 实现更并行、更 fused、少拷贝；而当前 vllm-ascend NPU 实现有几处高并发下会线性放大的适配代码。

### 2.1 input expand kernel：GPU 二维并行，NPU 单 grid 串行

上游 GPU 路径：

```text
vllm/v1/spec_decode/dflash.py
vllm/v1/spec_decode/utils.py
```

调用侧按 request 和 block 启动二维 grid：

```python
max_ctx_per_req = cad.max_query_len
max_tokens_per_req = max_ctx_per_req + num_query_per_req
BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
grid = (batch_size, num_blocks)
copy_and_expand_dflash_inputs_kernel[grid](...)
```

kernel 内部：

```python
req_idx = tl.program_id(axis=0)
block_idx = tl.program_id(axis=1)
j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

NPU 路径：

```text
vllm_ascend/spec_decode/dflash_proposer.py
vllm_ascend/ops/triton/spec_decode/utils.py
```

调用侧当前只有一个 program：

```python
copy_and_expand_dflash_inputs_kernel_single_grid[1,](...)
```

kernel 内部串行遍历：

```python
for req_idx in range(0, batch_size):
    for j in range(0, num_ctx):
        ...
```

影响：

- GPU 高并发下按 request/block 并行处理 input/position/slot mapping。
- NPU 当前实现会把 batch 和 context token 串在一个 Triton program 里处理。
- 这是 NPU 高并发 TTFT 比 GPU 恶化更多的首要嫌疑点。

### 2.2 target hidden states：GPU 引用，NPU 拷贝

上游 GPU：

```python
self._dflash_hidden_states = target_hidden_states
```

NPU：

```python
self._dflash_hidden_states[:num_context] = target_hidden_states
```

影响：

- GPU 避免了大 tensor copy。
- NPU 为了预分配 buffer / graph shape 稳定做了显式 copy。
- 高并发和长 prompt 时，`target_hidden_states` 很大，这个 copy 会直接进入 TTFT。

### 2.3 context KV precompute：GPU fused ops，NPU module/patch 路径

上游 GPU `qwen3_dflash.py` 使用：

```python
ops.rms_norm(...)
ops.rotary_embedding(...)
```

这些是 vLLM CUDA custom ops，RoPE 还是原地修改输入。

NPU patch 使用：

```python
self.hidden_norm(context_states)
k_norm_layer(all_k[i])
self.layers[0].self_attn.rotary_emb(...)
```

影响：

- GPU RMSNorm/RoPE 更 fused。
- NPU 走 module forward 和 Python per-layer loop，kernel launch 和调度更重。
- 当前 NPU patch 还有 `tmpv = all_k_flat.clone()` 的大拷贝，以及 RoPE 返回值未接收的问题。

### 2.4 RoPE 语义：GPU 原地，NPU 返回新 tensor

GPU CUDA `ops.rotary_embedding()` 是原地 op：

```python
ops.rotary_embedding(..., all_k_flat, None, ...)
```

因此上游代码不需要接返回值。

NPU 的 `AscendRotaryEmbedding.forward_oot()` 是：

```python
return torch.ops.vllm.npu_rotary_embedding(...)
```

并且 custom op 注册时 `mutates_args=[]`，从接口语义看不是原地修改输入。

影响：

- NPU 当前 patch 如果不接返回值，可能没有把 rotated K 写回。
- `tmpv.clone()` 还引入额外大拷贝。
- GPU 没有这部分额外 copy，也没有返回值丢弃风险。

### 2.5 graph padding：NPU bucket 放大可能更明显

NPU DFlash first pass 会经过：

```python
cudagraph_dispatcher.dispatch(...)
_pad_query_start_loc_for_fia(...)
```

如果 capture size 不合适，实际 batch/token 数会被 pad 到更大的 bucket。

影响：

- GPU 上也有 cudagraph/bucket，但 DFlash 上游链路更成熟。
- NPU 这版 first pass 还叠加了 single-grid kernel、hidden copy 和 patch precompute，padding 后放大更明显。

### 2.6 差异汇总

| 模块 | GPU 上游 vLLM | NPU vllm-ascend 当前实现 | 对 TTFT 的影响 |
| --- | --- | --- | --- |
| input expand | `(batch_size, num_blocks)` 二维 grid | single-grid 内双层 for loop | 高并发下 NPU 更容易线性放大 |
| hidden states | 直接保存引用 | copy 到 `_dflash_hidden_states` buffer | 高并发/长 prompt 多一次大拷贝 |
| RMSNorm | CUDA `ops.rms_norm` | module forward / per-layer loop | kernel launch 和 Python 调度更多 |
| RoPE | CUDA 原地 `ops.rotary_embedding` | `npu_rotary_embedding` 返回 tuple | 需要接返回值，当前还有 clone |
| KV cache update | 上游 fused 路径较成熟 | 逐层 `do_kv_cache_update` | 多层多次 cache write |
| graph padding | 上游路径较成熟 | draft first pass 可能 pad 到较大 bucket | 实际 tokens 被放大 |
| 首轮 draft | 两边都会做 | NPU 每个子步骤更重 | TTFT 差异被放大 |

所以，GPU 不明显变慢，并不代表 DFlash 首轮没有额外成本；而是 GPU 实现把这部分成本压得更低。NPU 当前的高并发 TTFT 问题更像是实现层面的并行度、拷贝和 patch 适配成本叠加。

## 3. 首 Token 调用链

target forward 结束后，在 `sample_tokens()` 中：

```text
model_runner_v1.py::sample_tokens
  -> sample_token
    -> self._sample(...)
  -> _bookkeeping_sync(...)
  -> draft_token
    -> propose_draft_token_ids(...)
      -> self.drafter.prepare_next_token_ids_padded(...)
      -> self.drafter._propose(...)
        -> AscendDflashProposer.set_inputs_first_pass(...)
        -> AscendSpecDecodeBaseProposer._run_merged_draft(...)
          -> AscendDflashProposer.build_model_inputs_first_pass(...)
            -> self.model.precompute_and_store_context_kv(...)
          -> self.model(...)
          -> compute_draft_token_ids(...)
      -> _copy_draft_token_ids_to_cpu(...)
    -> finalize_kv_connector()
  -> build ModelRunnerOutput
```

关键点：`ModelRunnerOutput` 在 `draft_token` 之后才构造，所以首 token 返回会等待 DFlash draft 准备完成。

代码位置：

- `vllm_ascend/worker/model_runner_v1.py`
  - `sample_token`: around line 2140
  - `draft_token`: around line 2185
  - `ModelRunnerOutput`: around line 2255
- `vllm_ascend/spec_decode/dflash_proposer.py`
  - `set_inputs_first_pass`
  - `build_model_inputs_first_pass`
- `vllm_ascend/patch/worker/patch_qwen3_dflash.py`
  - `precompute_and_store_context_kv`

## 4. 高并发下的主要放大点

### 4.1 DFlash 首轮 query 数按并发和 K 放大

代码：

```python
batch_size = cad.num_reqs
num_query_per_req = 1 + self.num_speculative_tokens
num_query_total = batch_size * num_query_per_req
```

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

影响：

- 并发越高，draft first-pass query tokens 越多。
- `num_speculative_tokens` 越大，首 token 前 draft query forward 越重。

优化方向：

- 高并发时降低 `num_speculative_tokens`，例如从 8 降到 2 或 4。
- 做动态 K：低并发 K=8，高并发 K=2/4。
- 首轮 prefill 跳过 DFlash，第二轮开始投机。

### 4.2 首 token 前同步执行 DFlash draft

代码：

```python
with record_function_or_nullcontext("draft_token"):
    if self.speculative_config:
        ...
        if use_padded_batch:
            if input_fits_in_drafter:
                propose_draft_token_ids(sampler_output.sampled_token_ids)
```

文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

影响：

- 首 token 返回前必须等 draft token ids 准备好。
- 高并发时 DFlash drafter first pass 变重，TTFT 被直接拉长。

优化方向 A：首轮跳过 DFlash

在 `spec_decode_metadata is None` 且当前 batch 包含 prefill 请求时，不调用 `propose_draft_token_ids()`，直接返回 target 首 token。下一轮 decode 再开始 speculative decoding。

伪代码方向：

```python
is_first_spec_round = spec_decode_metadata is None
if (
    self.speculative_config
    and self.speculative_config.use_dflash()
    and is_first_spec_round
    and get_ascend_config().dflash_skip_first_token_draft
):
    self._draft_token_ids = torch.zeros(
        1,
        device=self.device,
        dtype=torch.int32,
    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
else:
    propose_draft_token_ids(...)
```

收益：

- 对 TTFT 最直接。
- 不改变 target 首 token 质量。

代价：

- 第一轮没有 draft tokens，少一轮 speculative 加速。
- 后续吞吐可能略降，但高并发首 token SLA 会明显改善。

优化方向 B：阈值跳过

按 batch 和 context 动态跳过：

```python
skip_dflash = (
    self.speculative_config.use_dflash()
    and spec_decode_metadata is None
    and (
        self.input_batch.num_reqs >= dflash_ttft_skip_batch_threshold
        or scheduler_output.total_num_scheduled_tokens >= dflash_ttft_skip_token_threshold
    )
)
```

适合高并发/长 prompt 场景。

### 4.3 Ascend DFlash input expand kernel 单 grid 串行化

当前代码：

```python
copy_and_expand_dflash_inputs_kernel_single_grid[1,](...)
```

kernel 内部：

```python
for req_idx in range(0, batch_size):
    ...
    for j in range(0, num_ctx):
        ...
```

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
vllm_ascend/ops/triton/spec_decode/utils.py
```

问题：

- 只有一个 Triton program。
- 高并发和长 context 下，所有 request/context token 被串行 loop 处理。
- 上游 vLLM 使用二维 grid `(batch_size, num_blocks)` 并行处理。

上游参考：

```text
vllm/v1/spec_decode/utils.py::copy_and_expand_dflash_inputs_kernel
```

优化方向：

把 Ascend 的 single-grid kernel 改成类似上游的二维 grid。

具体改造目标：

- 每个 request 分配一个 `axis=0` program。
- 每个 request 内按 token block 分配 `axis=1` program。
- 一个 program 处理 `BLOCK_SIZE` 个逻辑 token，逻辑 token 范围是 `context tokens + query tokens`。
- context 部分写：
  - `out_context_positions`
  - `out_context_slot_mapping`
- query 部分写：
  - `out_query_positions`
  - `out_query_slot_mapping`
  - `out_input_ids`
  - `out_token_indices`

调用侧方向：

```python
max_ctx_per_req = cad.max_query_len
max_tokens_per_req = max_ctx_per_req + num_query_per_req
BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
grid = (batch_size, num_blocks)

copy_and_expand_dflash_inputs_kernel[grid](
    ...,
    BLOCK_SIZE=BLOCK_SIZE,
    HAS_NUM_REJECTED=has_num_rejected,
)
```

kernel 方向：

- `req_idx = tl.program_id(axis=0)`
- `block_idx = tl.program_id(axis=1)`
- 用 `j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` 并行处理 context/query。

建议新增一个 kernel，而不是直接覆盖旧 kernel：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel(
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
    tl.store(
        out_token_indices_ptr + sample_out_idx,
        query_out,
        mask=is_sample,
    )
```

这个 kernel 基本就是上游 GPU kernel 的 NPU 侧移植版本，保留 Ascend 当前参数命名和 buffer 语义。

预期收益：

- 高并发下明显减少 input expand 时间。
- 这是最像“高并发特别慢”的代码点，建议优先做。

风险：

- Ascend Triton grid 和 block size 需要实测。
- 注意 `block_table_stride` 边界、`HAS_NUM_REJECTED`、query/token index 写入一致性。
- 注意 `query_off = j - num_ctx` 在 context token 上是负值，但所有 query 写入都必须用 `mask=is_query` 保护。
- 注意 `valid_ctx_end - 1` 在异常空 context 下可能越界；DFlash 正常首轮应有 context token，但建议保留 assert 或 defensive check。
- 如果 Ascend Triton 对二维 grid 或较大 `BLOCK_SIZE` 表现不稳定，可以先用较小 `BLOCK_SIZE=64/128` 做 A/B。

验证方式：

- 随机生成不同 `batch_size`、不同 `query_start_loc` 的输入，对比 single-grid 和二维 grid 输出：
  - `out_input_ids`
  - `out_context_positions`
  - `out_query_positions`
  - `out_context_slot_mapping`
  - `out_query_slot_mapping`
  - `out_token_indices`
- 覆盖 `HAS_NUM_REJECTED=False/True`。
- 覆盖非均匀 prompt length。
- 对比高并发 profiler 中 `copy_and_expand_dflash_inputs_*` 的耗时。

### 4.4 target hidden states 大拷贝

当前 Ascend 代码：

```python
self._dflash_num_context = num_context
self._dflash_hidden_states[:num_context] = target_hidden_states
```

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

上游 vLLM：

```python
self._dflash_hidden_states = target_hidden_states
```

文件：

```text
vllm/v1/spec_decode/dflash.py
```

影响：

- 高并发/长 prompt 下，`target_hidden_states` 是大 tensor。
- 拷贝发生在首 token 前。

优化方向：

在不进入 graph capture 的路径中，直接保存引用：

```python
if can_use_reference_for_dflash_hidden_states:
    self._dflash_hidden_states = target_hidden_states
else:
    self._dflash_hidden_states_buffer[:num_context].copy_(target_hidden_states)
    self._dflash_hidden_states = self._dflash_hidden_states_buffer
```

注意：

- 当前 Ascend 预分配 buffer 可能是为了 ACL graph shape 稳定。
- 需要区分 eager / graph capture / full graph runtime。
- 可以先只在 `spec_decode_metadata is None` 且非 graph capture 路径做引用优化。

### 4.5 aux hidden states cat 和 combine_hidden_states

当前代码：

```python
if self.use_aux_hidden_state_outputs:
    target_hidden_states = torch.cat(
        [h[:num_scheduled_tokens] for h in aux_hidden_states],
        dim=-1,
    )
```

随后：

```python
target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
```

文件：

```text
vllm_ascend/worker/model_runner_v1.py
vllm_ascend/spec_decode/llm_base_proposer.py
vllm/model_executor/models/qwen3_dflash.py
```

影响：

- `torch.cat` 分配大 tensor。
- `combine_hidden_states` 通常会走 draft model 的 `fc`。
- 高并发时这部分内存带宽和 GEMM 都会进入 TTFT。

优化方向：

- 如果 DFlash head 配置支持 `use_aux_hidden_state=False`，优先验证质量/接受率变化后关闭 aux hidden states。
- 避免先 cat 再 fc：将 `fc` 拆成多个分片线性层，对 aux hidden states 分别 matmul 后累加。
- 只选必要 layer ids，减少 aux hidden states 数量。

风险：

- 会影响 DFlash draft 质量和接受率。
- 需要结合模型配置验证。

### 4.6 precompute_and_store_context_kv 成本

当前代码路径：

```text
precompute_and_store_context_kv
  -> hidden_norm
  -> fused KV projection
  -> per-layer K RMSNorm
  -> RoPE
  -> per-layer KV cache update
```

文件：

```text
vllm_ascend/patch/worker/patch_qwen3_dflash.py
```

已知可优化点：

- RoPE 返回值未接收，且 `tmpv = all_k_flat.clone()` 成本高。
- per-layer `k_norm` 是 Python/module loop。
- per-layer `do_kv_cache_update` 逐层写 cache。

推荐先做：

```python
all_k_flat = all_k_normed.view(L * num_ctx, kv)
positions_repeated = context_positions.repeat(L)

key_scratch = torch.empty_like(all_k_flat)
all_k_flat, _ = self.layers[0].self_attn.rotary_emb(
    positions_repeated,
    all_k_flat,
    key_scratch,
)
```

更完整说明见：

```text
D:\workspace\speculative\qwen3_dflash_patch_optimization_README.md
```

### 4.7 graph padding / capture bucket 放大

DFlash proposer 会进入：

```python
aclgraph_runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(...)
num_input_tokens = batch_descriptor.num_tokens
```

并且 full graph 时会 pad query start loc：

```python
num_reqs_padded = self.runner._pad_query_start_loc_for_fia(...)
```

文件：

```text
vllm_ascend/spec_decode/llm_base_proposer.py
```

影响：

- capture size 不合适时，小幅实际 batch 可能 pad 到很大的 bucket。
- 高并发下 draft first-pass 的 `num_input_tokens` 可能被放大。

优化方向：

- 单独调小/细化 DFlash drafter 的 capture sizes。
- 首 token/prefill 阶段强制 DFlash eager 或跳过 DFlash。
- 监控 `batch_descriptor.num_tokens` 与真实 `num_tokens` 的差距。

## 5. 推荐优化优先级

| 优先级 | 方向 | 代码位置 | 预期收益 | 风险 |
| --- | --- | --- | --- | --- |
| P0 | 首轮跳过 DFlash draft | `model_runner_v1.py::sample_tokens` | TTFT 最直接改善 | 第一轮无投机收益 |
| P0 | input expand kernel 二维 grid 并行化 | `ops/triton/spec_decode/utils.py` | 高并发明显改善 | 需验证 Triton/NPU kernel |
| P0 | 修 RoPE 返回值并去掉 clone | `patch_qwen3_dflash.py` | 正确性 + 减少拷贝 | 低 |
| P1 | 避免 target_hidden_states 大拷贝 | `dflash_proposer.py::set_inputs_first_pass` | 降低高并发内存带宽 | graph capture 需谨慎 |
| P1 | 高并发动态降低 K | 配置/调度策略 | 降低首轮 draft forward | 后续吞吐可能下降 |
| P1 | DFlash 按阈值跳过 | `sample_tokens` / proposer | 改善 tail TTFT | 部分请求无投机 |
| P2 | 优化 aux hidden cat + FC | `model_runner_v1.py` / model | 降低内存和 GEMM | 可能影响接受率 |
| P2 | multi-layer KV cache update | attention backend | 减少 kernel launch | 改动大 |
| P2 | query-only RoPE op | `ops/rotary_embedding.py` | 避免无用 key RoPE | 需要新增 custom op |

## 6. 建议第一阶段 Patch

本节给出更具体的代码修改点。行号基于当前 workspace 的 `vllm-ascend-0.20.2rc1`，如果后续源码有变动，以函数名和附近代码为准。

### 6.0 第一阶段代码修改清单

| 目标 | 文件 | 当前行附近 | 修改方式 |
| --- | --- | ---: | --- |
| 增加首轮/高并发跳过 DFlash | `vllm_ascend/worker/model_runner_v1.py` | 20-29, 2185-2227 | 增加 `os` import，在 `draft_token` 块里加 skip 判断 |
| DFlash input expand 并行化 | `vllm_ascend/ops/triton/spec_decode/utils.py` | 68-136 | 保留旧 single-grid kernel，新增二维 grid kernel |
| 调用二维 input expand kernel | `vllm_ascend/spec_decode/dflash_proposer.py` | 1-12, 95-120 | 增加 import，替换 kernel 调用为可 A/B 的新旧路径 |
| 减少 hidden states 拷贝 | `vllm_ascend/spec_decode/dflash_proposer.py` | 57-59, 84-85, 254-255 | buffer 改名，非 capture 路径直接引用 target hidden states |
| 修 DFlash RoPE 返回值和 clone | `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | 40-43 | `clone()` 改 `empty_like()` scratch，并接收 RoPE 返回值 |

### 6.0.1 修改 `model_runner_v1.py`：首轮/高并发跳过 DFlash

文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

当前位置：文件头部 import 附近，当前约 20-29 行：

```python
import math
import sys
import time
from collections import defaultdict
```

建议改为：

```python
import math
import os
import sys
import time
from collections import defaultdict
```

原因：

- 用环境变量做 A/B，避免先引入配置项影响面太大。
- 后续确认有效后，可以再迁移到 `AscendConfig`。

当前位置：`sample_tokens()` 的 `draft_token` 块，当前约 2185-2227 行：

```python
with record_function_or_nullcontext("draft_token"):
    if self.speculative_config:
        input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
            <= self.effective_drafter_max_model_len
        )
        use_padded_batch = (
            self.speculative_config
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
                or self.speculative_config.use_ngram_gpu()
            )
            and not self.speculative_config.disable_padded_drafter_batch
        )
        if use_padded_batch:
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                propose_draft_token_ids(sampler_output.sampled_token_ids)
            ...
```

建议在 `input_fits_in_drafter` 后面加 skip 判断，并在 `use_padded_batch` 分支里优先处理：

```python
with record_function_or_nullcontext("draft_token"):
    if self.speculative_config:
        input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
            <= self.effective_drafter_max_model_len
        )

        # DFlash first pass can be very expensive for high-concurrency prefill
        # batches.  When enabled, skip preparing draft tokens before returning
        # the first target token; speculative decoding resumes in later rounds.
        dflash_skip_first = os.getenv(
            "VLLM_ASCEND_DFLASH_SKIP_FIRST_DRAFT",
            "0",
        ) == "1"
        dflash_skip_batch_threshold = int(os.getenv(
            "VLLM_ASCEND_DFLASH_SKIP_BATCH_THRESHOLD",
            "0",
        ))
        dflash_skip_token_threshold = int(os.getenv(
            "VLLM_ASCEND_DFLASH_SKIP_TOKEN_THRESHOLD",
            "0",
        ))
        is_dflash_first_round = (
            self.speculative_config.use_dflash()
            and spec_decode_metadata is None
        )
        skip_dflash_draft = (
            is_dflash_first_round
            and (
                dflash_skip_first
                or (
                    dflash_skip_batch_threshold > 0
                    and self.input_batch.num_reqs >= dflash_skip_batch_threshold
                )
                or (
                    dflash_skip_token_threshold > 0
                    and scheduler_output.total_num_scheduled_tokens
                    >= dflash_skip_token_threshold
                )
            )
        )

        use_padded_batch = (
            self.speculative_config
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
                or self.speculative_config.use_ngram_gpu()
            )
            and not self.speculative_config.disable_padded_drafter_batch
        )
        if use_padded_batch:
            sampled_token_ids = sampler_output.sampled_token_ids
            if skip_dflash_draft:
                # Tell scheduler there are no draft tokens for this round.
                # This improves TTFT by moving DFlash setup out of the first
                # token critical path. Later rounds can still use DFlash.
                self._draft_token_ids = torch.zeros(
                    1,
                    device=self.device,
                    dtype=torch.int32,
                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                self._copy_draft_token_ids_to_cpu(
                    scheduler_output,
                    zeros_only=True,
                )
            elif input_fits_in_drafter:
                propose_draft_token_ids(sampler_output.sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                ...
```

为什么这样改：

- `spec_decode_metadata is None` 表示当前不是验证上一轮 draft 的 speculative decode 轮次，常见于 prefill 后准备首轮 draft。
- 跳过这一轮 DFlash draft，可以让首 token 先返回。
- 用全 0 draft tokens 回传 CPU，是为了保持 scheduler 侧状态更新路径不崩；这和当前 `input_fits_in_drafter=False` 时的 zeros-only 分支语义接近。

注意：

- 这会让第二轮再开始承担 DFlash 初始化成本，第二个 token ITL 可能升高。
- 建议先用阈值打开，而不是全局无条件打开。
- 如果 scheduler 对全 0 draft tokens 和 no draft tokens 的语义不同，需要结合实际 scheduler 输出再做微调。

### 6.0.2 修改 `dflash_proposer.py`：import 和 input expand 调用

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

当前位置：文件头部，当前约 1-12 行：

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
```

建议改为：

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
```

原因：

- `os` 用于 A/B 开关。
- `triton` 用于 `next_power_of_2` 和 `cdiv`。
- 同时保留新旧 kernel，方便快速回退。

当前位置：`set_inputs_first_pass()` 中，当前约 95-120 行：

```python
copy_and_expand_dflash_inputs_kernel_single_grid[1,](
    ...
    batch_size=batch_size,
    HAS_NUM_REJECTED=has_num_rejected,
)
```

建议替换为：

```python
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
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=self.input_ids,
        out_context_positions_ptr=self._context_positions_buffer,
        out_query_positions_ptr=self.positions,
        out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
        out_query_slot_mapping_ptr=self._slot_mapping_buffer,
        out_token_indices_ptr=token_indices_to_sample,
        block_table_ptr=cad.block_table_tensor,
        block_table_stride=cad.block_table_tensor.stride(0),
        query_start_loc_ptr=cad.query_start_loc,
        num_rejected_tokens_ptr=(
            num_rejected_tokens_gpu if has_num_rejected else 0
        ),
        parallel_drafting_token_id=self.parallel_drafting_token_id,
        block_size=self.kernel_block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=self.num_speculative_tokens,
        total_input_tokens=num_context,
        BLOCK_SIZE=block_size_tl,
        HAS_NUM_REJECTED=has_num_rejected,
    )
else:
    # Fallback path. Keep this until the 2-D grid kernel is fully validated on
    # all target NPU SKUs and prompt length distributions.
    copy_and_expand_dflash_inputs_kernel_single_grid[1,](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=self.input_ids,
        out_context_positions_ptr=self._context_positions_buffer,
        out_query_positions_ptr=self.positions,
        out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
        out_query_slot_mapping_ptr=self._slot_mapping_buffer,
        out_token_indices_ptr=token_indices_to_sample,
        block_table_ptr=cad.block_table_tensor,
        block_table_stride=cad.block_table_tensor.stride(0),
        query_start_loc_ptr=cad.query_start_loc,
        num_rejected_tokens_ptr=(
            num_rejected_tokens_gpu if has_num_rejected else 0
        ),
        parallel_drafting_token_id=self.parallel_drafting_token_id,
        block_size=self.kernel_block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=self.num_speculative_tokens,
        total_input_tokens=num_context,
        batch_size=batch_size,
        HAS_NUM_REJECTED=has_num_rejected,
    )
```

为什么这样改：

- 当前 single-grid kernel 的 `for req_idx` 和 `for j` 会把高并发请求串在一个 program 内。
- 二维 grid 让 request 和 token block 并行，和上游 GPU 逻辑一致。
- 先用 `BLOCK_SIZE=128` 比 `256` 更保守，NPU 上更容易稳定；后续可以 profile `64/128/256`。

### 6.0.3 修改 `utils.py`：新增二维 grid kernel

文件：

```text
vllm_ascend/ops/triton/spec_decode/utils.py
```

当前位置：当前约 68-136 行是旧 kernel：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(...):
    for req_idx in range(0, batch_size):
        ...
```

建议保留旧 kernel，在它后面新增：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel(
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
    # axis 0: request id, axis 1: token block inside this request.
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

    # Context positions come from the target model positions.
    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)

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

    # Map absolute positions to KV-cache slots through the request block table.
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

    # Query token 0 is the sampled bonus token; query tokens 1..K are DFlash
    # mask tokens that will be sampled as draft tokens.
    bonus_token = tl.load(next_token_ids_ptr + req_idx)
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)

    # Store sample indices only for mask tokens, not the bonus token.
    is_sample = is_query & (query_off > 0)
    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
    tl.store(
        out_token_indices_ptr + sample_out_idx,
        query_out,
        mask=is_sample,
    )
```

为什么这样改：

- 每个 request/block 独立 program，避免一个 program 串行处理所有请求。
- `mask=is_ctx/is_query/is_sample` 保证 context 和 query 写不同 buffer 时不会互相污染。
- 逻辑基本复刻上游 vLLM DFlash kernel，便于对齐验证。

### 6.0.4 修改 `dflash_proposer.py`：减少 hidden states copy

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

当前位置：`__init__()` 中，当前约 57-59 行：

```python
self._dflash_hidden_states = torch.zeros(
    (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
)
```

建议改为：

```python
self._dflash_hidden_states_buffer = torch.zeros(
    (self.max_num_tokens, self.hidden_size),
    dtype=self.dtype,
    device=self.device,
)
self._dflash_hidden_states = self._dflash_hidden_states_buffer
```

当前位置：`set_inputs_first_pass()` 中，当前约 84-85 行：

```python
self._dflash_num_context = num_context
self._dflash_hidden_states[:num_context] = target_hidden_states
```

建议改为：

```python
self._dflash_num_context = num_context

use_hidden_ref = os.getenv(
    "VLLM_ASCEND_DFLASH_USE_HIDDEN_REF",
    "1",
) == "1"

if use_hidden_ref and not _EXTRA_CTX.capturing:
    # Eager path: precompute_and_store_context_kv consumes the tensor before
    # returning from the current draft pass, so a direct reference avoids a
    # large device-to-device copy on the TTFT critical path.
    self._dflash_hidden_states = target_hidden_states
else:
    # Graph/capture fallback: keep the stable preallocated buffer.
    self._dflash_hidden_states_buffer[:num_context].copy_(target_hidden_states)
    self._dflash_hidden_states = self._dflash_hidden_states_buffer
```

为什么这样改：

- 上游 GPU DFlash 直接保存 `target_hidden_states` 引用，没有 copy。
- NPU 当前 copy 在高并发/长 prompt 下会进入首 token 路径。
- graph capture 可能需要稳定 buffer，因此只在非 capture 路径默认引用。

### 6.0.5 修改 `patch_qwen3_dflash.py`：修 RoPE 返回值并去掉 clone

文件：

```text
vllm_ascend/patch/worker/patch_qwen3_dflash.py
```

当前位置：当前约 40-43 行：

```python
all_k_flat = all_k_normed.view(L * num_ctx, kv)
positions_repeated = context_positions.repeat(L)
tmpv = all_k_flat.clone()
self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
```

建议改为：

```python
all_k_flat = all_k_normed.view(L * num_ctx, kv)
positions_repeated = context_positions.repeat(L)

# Ascend rotary embedding returns rotated tensors instead of mutating inputs.
# This path only needs rotated K, so use an uninitialized scratch key to avoid
# cloning the full K buffer.
key_scratch = torch.empty_like(all_k_flat)
all_k_flat, _ = self.layers[0].self_attn.rotary_emb(
    positions_repeated,
    all_k_flat,
    key_scratch,
)
```

为什么这样改：

- GPU 上游 `ops.rotary_embedding` 是原地 op；NPU `npu_rotary_embedding` 注册为 `mutates_args=[]`，返回 `(query, key)`。
- 当前代码不接返回值，可能没有把 rotated K 写回。
- `tmpv = all_k_flat.clone()` 是大拷贝；`empty_like` scratch 避免复制内容。

### 6.1 增加首轮跳过 DFlash 开关

增加配置项或环境变量，例如：

```text
VLLM_ASCEND_DFLASH_SKIP_FIRST_DRAFT=1
VLLM_ASCEND_DFLASH_SKIP_BATCH_THRESHOLD=64
VLLM_ASCEND_DFLASH_SKIP_TOKEN_THRESHOLD=8192
```

在 `model_runner_v1.py::sample_tokens` 的 `draft_token` 块中判断：

```python
skip_first_dflash = (
    self.speculative_config is not None
    and self.speculative_config.use_dflash()
    and spec_decode_metadata is None
    and (
        os.getenv("VLLM_ASCEND_DFLASH_SKIP_FIRST_DRAFT", "0") == "1"
        or self.input_batch.num_reqs >= dflash_skip_batch_threshold
        or scheduler_output.total_num_scheduled_tokens >= dflash_skip_token_threshold
    )
)
```

如果跳过：

```python
self._draft_token_ids = torch.zeros(
    1,
    device=self.device,
    dtype=torch.int32,
).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
```

否则走原来的 `propose_draft_token_ids(...)`。

### 6.2 将 DFlash input expand kernel 改成二维 grid

#### 6.2.1 新增 kernel

```text
vllm_ascend/ops/triton/spec_decode/utils.py
```

保留旧的：

```python
copy_and_expand_dflash_inputs_kernel_single_grid
```

新增：

```text
copy_and_expand_dflash_inputs_kernel[grid=(batch_size, num_blocks)]
```

这样可以通过开关做 A/B，而不是一次性替换。

#### 6.2.2 修改调用侧 import

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py::set_inputs_first_pass
```

当前 import：

```python
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_inputs_kernel_single_grid
```

建议改为：

```python
from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    copy_and_expand_dflash_inputs_kernel_single_grid,
)
```

#### 6.2.3 调用侧计算 grid

在 `set_inputs_first_pass()` 中，替换 single-grid 调用前先计算：

```python
max_ctx_per_req = cad.max_query_len
max_tokens_per_req = max_ctx_per_req + num_query_per_req
block_size_tl = min(256, triton.next_power_of_2(max_tokens_per_req))
num_blocks = triton.cdiv(max_tokens_per_req, block_size_tl)
grid = (batch_size, num_blocks)
```

需要新增：

```python
from vllm.triton_utils import triton
```

如果担心 `BLOCK_SIZE=256` 在 NPU 上过大，可以先保守：

```python
block_size_tl = min(128, triton.next_power_of_2(max_tokens_per_req))
```

#### 6.2.4 用开关选择新旧 kernel

建议先加环境变量开关：

```python
use_parallel_expand = os.getenv(
    "VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND",
    "1",
) == "1"
```

需要新增：

```python
import os
```

新路径：

```python
if use_parallel_expand:
    max_ctx_per_req = cad.max_query_len
    max_tokens_per_req = max_ctx_per_req + num_query_per_req
    block_size_tl = min(128, triton.next_power_of_2(max_tokens_per_req))
    num_blocks = triton.cdiv(max_tokens_per_req, block_size_tl)
    grid = (batch_size, num_blocks)

    copy_and_expand_dflash_inputs_kernel[grid](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=self.input_ids,
        out_context_positions_ptr=self._context_positions_buffer,
        out_query_positions_ptr=self.positions,
        out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
        out_query_slot_mapping_ptr=self._slot_mapping_buffer,
        out_token_indices_ptr=token_indices_to_sample,
        block_table_ptr=cad.block_table_tensor,
        block_table_stride=cad.block_table_tensor.stride(0),
        query_start_loc_ptr=cad.query_start_loc,
        num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
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
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=self.input_ids,
        out_context_positions_ptr=self._context_positions_buffer,
        out_query_positions_ptr=self.positions,
        out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
        out_query_slot_mapping_ptr=self._slot_mapping_buffer,
        out_token_indices_ptr=token_indices_to_sample,
        block_table_ptr=cad.block_table_tensor,
        block_table_stride=cad.block_table_tensor.stride(0),
        query_start_loc_ptr=cad.query_start_loc,
        num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
        parallel_drafting_token_id=self.parallel_drafting_token_id,
        block_size=self.kernel_block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=self.num_speculative_tokens,
        total_input_tokens=num_context,
        batch_size=batch_size,
        HAS_NUM_REJECTED=has_num_rejected,
    )
```

启动时：

```bash
export VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND=1
```

回退时：

```bash
export VLLM_ASCEND_DFLASH_PARALLEL_INPUT_EXPAND=0
```

#### 6.2.5 替换前后要保持的输出语义

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

- `HAS_NUM_REJECTED=True` 时，`last_pos` 要用 `valid_ctx_end = ctx_end - num_rejected`。
- `q_idx == 0` 是 bonus token，不写 `out_token_indices`。
- `q_idx > 0` 是 speculative mask token，需要写 `out_token_indices`。
- `block_num` 应 clamp 到 `block_table_stride - 1`，避免位置接近上限时越界。

#### 6.2.6 单元测试建议

写一个小测试构造随机 batch：

```text
batch_size = 1 / 8 / 32 / 128
num_speculative_tokens = 1 / 4 / 8
prompt lengths = uniform / non-uniform
HAS_NUM_REJECTED = False / True
```

分别调用：

```python
copy_and_expand_dflash_inputs_kernel_single_grid
copy_and_expand_dflash_inputs_kernel
```

比较：

```python
torch.testing.assert_close(out_input_ids_old, out_input_ids_new)
torch.testing.assert_close(out_context_positions_old, out_context_positions_new)
torch.testing.assert_close(out_query_positions_old, out_query_positions_new)
torch.testing.assert_close(out_context_slot_mapping_old, out_context_slot_mapping_new)
torch.testing.assert_close(out_query_slot_mapping_old, out_query_slot_mapping_new)
torch.testing.assert_close(out_token_indices_old, out_token_indices_new)
```

#### 6.2.7 性能验收

profiler 中重点看：

```text
copy_and_expand_dflash_inputs_kernel_single_grid
copy_and_expand_dflash_inputs_kernel
```

预期：

- batch 小时收益不一定明显。
- batch 大、prompt length 非均匀时收益更明显。
- kernel 时间不应随 `batch_size * avg_prompt_len` 近似线性恶化。

如果二维 grid kernel 在 NPU 上启动开销过大，可以尝试：

- `BLOCK_SIZE=64`
- `BLOCK_SIZE=128`
- `BLOCK_SIZE=256`
- 对短 prompt 仍走 single-grid，对高并发/长 prompt 走 parallel kernel。

示例阈值：

```python
use_parallel_expand = (
    batch_size >= 16
    or num_context >= 2048
)
```

原始简化替换关系：

旧：

```python
copy_and_expand_dflash_inputs_kernel_single_grid[1,](...)
```

新：

```python
grid = (batch_size, num_blocks)
copy_and_expand_dflash_inputs_kernel[grid](...)
```

### 6.3 修正 `patch_qwen3_dflash.py` RoPE 段

替换：

```python
tmpv = all_k_flat.clone()
self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
```

为：

```python
key_scratch = torch.empty_like(all_k_flat)
all_k_flat, _ = self.layers[0].self_attn.rotary_emb(
    positions_repeated,
    all_k_flat,
    key_scratch,
)
```

## 7. Profiler 验证项

建议在 NPU profiler 中按 `record_function` 或 kernel name 分解：

```text
sample_token
draft_token
prepare_next_token_ids_padded
copy_and_expand_dflash_inputs_kernel_single_grid
precompute_and_store_context_kv
npu_rotary_embedding
RMSNorm kernels
reshape_and_cache
DFlash model forward
compute_draft_token_ids
_copy_draft_token_ids_to_cpu
finalize_kv_connector
```

重点观察：

- `draft_token` 是否占 TTFT 大头。
- `copy_and_expand_dflash_inputs_kernel_single_grid` 是否随并发线性增长。
- `precompute_and_store_context_kv` 是否随 `num_context` 增长过快。
- graph padding 后 `num_input_tokens` 是否远大于真实 tokens。
- K 从 8 降到 4/2 后 TTFT 是否显著改善。

## 8. 压测矩阵

建议至少跑：

| 场景 | 目的 |
| --- | --- |
| no spec decode | baseline TTFT |
| DFlash K=8 | 当前问题复现 |
| DFlash K=4 / K=2 | 验证 K 对 TTFT 的放大 |
| DFlash K=8 + skip first draft | 验证首轮跳过收益 |
| DFlash K=8 + 二维 input expand kernel | 验证高并发 kernel 优化 |
| DFlash K=8 + RoPE clone fix | 验证 precompute 优化 |
| DFlash K=8 + disable graph for first pass | 验证 padding/capture 影响 |

指标：

- TTFT p50 / p90 / p99。
- ITL / TPOT。
- throughput。
- speculative acceptance rate。
- `draft_token` 时间。
- `precompute_and_store_context_kv` 时间。

## 9. 预期收益排序

如果目标是快速改善高并发 TTFT：

1. 首轮跳过 DFlash draft。
2. 高并发动态降低 K。
3. DFlash input expand kernel 二维并行化。

如果目标是在保持首轮 DFlash 的同时降成本：

1. input expand kernel 二维并行化。
2. 修 `precompute_and_store_context_kv` 的 RoPE clone/返回值问题。
3. 减少 hidden states copy。
4. 优化 K RMSNorm 和 KV cache update。

如果目标是长期代码质量：

1. query-only RoPE op。
2. multi-layer KV cache update。
3. aux hidden states 分片 FC，避免大 cat。

## 10. 总结

DFlash 高并发 TTFT 变慢的直接原因是首 token 返回前同步执行了 draft first-pass。这个 first-pass 的成本随并发、prompt/context token 数和 `num_speculative_tokens` 同时放大。

最值得先动的代码是：

- `model_runner_v1.py::sample_tokens`：增加首轮/高并发跳过 DFlash 的策略。
- `ops/triton/spec_decode/utils.py`：把 single-grid input expand kernel 改成二维 grid。
- `patch_qwen3_dflash.py`：修 RoPE 返回值并去掉 `clone()`。
- `dflash_proposer.py::set_inputs_first_pass`：减少 target hidden states 拷贝。

这四处能覆盖 TTFT 放大的主要来源。
