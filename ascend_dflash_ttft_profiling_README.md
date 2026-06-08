# vLLM-Ascend DFlash TTFT 耗时拆解打点说明

本文说明如何在 `vllm-ascend v0.20.2rc1` 中拆解 DFlash 推理时的首 token 时延，重点回答两个问题：

1. DFlash 自身的额外耗时主要花在哪里。
2. 完整首 token 时延应该按哪些阶段拆。

相关代码路径基于当前 workspace：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1
D:\workspace\speculative\vllm-v0.20.2
```

## 1. 结论先行

DFlash 场景下，首 token 时延不只包含 target model 的 prefill/forward，还可能包含：

- 输入准备、scheduler output 处理、padding/bucket 选择。
- target model forward。
- logits 计算。
- sample。
- bookkeeping 和 CPU/GPU/NPU 数据同步。
- DFlash draft token proposal。
- DFlash context hidden states 到 draft KV cache 的预处理。
- draft model forward。
- draft token ids 拷贝到 CPU。
- KV connector finalize。
- output 构造和异步状态更新。

因此建议分两层打点：

| 层级 | 目的 | 典型位置 |
| --- | --- | --- |
| 完整 TTFT 拆解 | 判断首 token 慢是 target、sample、draft、bookkeeping 还是 finalize | `model_runner_v1.py` |
| DFlash 内部拆解 | 判断 DFlash 慢在 input expand、hidden copy、precompute 还是 draft forward | `dflash_proposer.py`、`llm_base_proposer.py`、`patch_qwen3_dflash.py` |

## 2. 计时方式选择

NPU/CUDA 执行是异步的。普通 `time.perf_counter()` 只能测 Python launch 时间，不能测真实 device 执行时间。为了拆大头，建议先使用同步计时。

| 方式 | 优点 | 缺点 | 推荐用途 |
| --- | --- | --- | --- |
| `record_function_or_nullcontext` | 已有代码大量使用，可配合 profiler | 不直接打印耗时 | profiler 抓全链路 |
| `time.perf_counter()` + `torch.npu.synchronize()` | 最直观，能直接看每段 ms | 强制同步，会干扰吞吐 | 先定位大头 |
| torch/npu profiler | 能看算子级细节 | 配置和结果分析更重 | 定位具体慢算子 |

下面的代码示例使用环境变量开关，只有打开 `VLLM_ASCEND_TTFT_PROFILE=1` 或 `VLLM_ASCEND_DFLASH_PROFILE=1` 时才会打印。

## 3. 通用同步 timer

### 3.1 加到 `model_runner_v1.py`

文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

顶部已有 `time` 和 `contextmanager`，需要补 `os`：

```diff
 import math
+import os
 import sys
 import time
```

然后在 import 区后面增加：

```python
_ASCEND_TTFT_PROFILE = os.getenv("VLLM_ASCEND_TTFT_PROFILE", "0") == "1"


def _ttft_sync_device():
    if torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _ttft_timer(name: str):
    if not _ASCEND_TTFT_PROFILE:
        yield
        return

    _ttft_sync_device()
    start = time.perf_counter()
    yield
    _ttft_sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"[TTFTProfile] {name}: {elapsed_ms:.3f} ms", flush=True)
```

说明：

- 这是强同步计时，会影响性能，只用于定位。
- 打开后建议只跑少量请求或固定压测样本。
- 如果多卡/多进程启动，每个 worker 都会打印，需要根据 rank 过滤日志。

### 3.2 加到 `dflash_proposer.py`

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

顶部增加：

```diff
 from typing import Any
 
+import os
+import time
+from contextlib import contextmanager
 import torch
```

增加 timer：

```python
_DFLASH_PROFILE = os.getenv("VLLM_ASCEND_DFLASH_PROFILE", "0") == "1"


def _dflash_sync_device():
    if torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _dflash_timer(name: str):
    if not _DFLASH_PROFILE:
        yield
        return

    _dflash_sync_device()
    start = time.perf_counter()
    yield
    _dflash_sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"[DFlashProfile] {name}: {elapsed_ms:.3f} ms", flush=True)
```

### 3.3 加到 `patch_qwen3_dflash.py`

文件：

```text
vllm_ascend/patch/worker/patch_qwen3_dflash.py
```

顶部增加：

```diff
+import os
+import time
+from contextlib import contextmanager
 import torch
 import torch.nn.functional as F
```

增加 timer：

```python
_DFLASH_PROFILE = os.getenv("VLLM_ASCEND_DFLASH_PROFILE", "0") == "1"


def _dflash_sync_device():
    if torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _dflash_timer(name: str):
    if not _DFLASH_PROFILE:
        yield
        return

    _dflash_sync_device()
    start = time.perf_counter()
    yield
    _dflash_sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"[DFlashProfile] {name}: {elapsed_ms:.3f} ms", flush=True)
```

## 4. 完整首 token 时延怎么拆

完整首 token 链路主要在：

```text
vllm_ascend/worker/model_runner_v1.py
```

关键函数：

```text
execute_model()
sample_tokens()
```

调用链可以简化为：

```text
execute_model
  -> prepare input
  -> _prepare_inputs
  -> _determine_batch_execution_and_padding
  -> _build_attention_metadata
  -> _preprocess
  -> target model forward
  -> post process
     -> sample_hidden_states = hidden_states[logits_indices]
     -> compute_logits
  -> 保存 execute_model_state

sample_tokens
  -> _sample
  -> _bookkeeping_sync
  -> draft_token
     -> propose_draft_token_ids
     -> _copy_draft_token_ids_to_cpu
     -> finalize_kv_connector
  -> ModelRunnerOutput
  -> async_state_update
```

### 4.1 拆 `execute_model()`

位置：

```text
vllm_ascend/worker/model_runner_v1.py:1665
```

建议把已有的 `record_function_or_nullcontext` 外层再套同步 timer。

修改前：

```python
with record_function_or_nullcontext("prepare input"):
    with self.synchronize_input_prep():
        ...
```

修改后：

```python
with _ttft_timer("execute.prepare_input"):
    with record_function_or_nullcontext("prepare input"):
        with self.synchronize_input_prep():
            ...
```

target forward 位置在 `record_function_or_nullcontext("forward")`：

```python
with _ttft_timer("execute.target_forward"):
    with (
        record_function_or_nullcontext("forward"),
        set_ascend_forward_context(...),
        self.maybe_get_kv_connector_output(...),
    ):
        hidden_states = self._model_forward(
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )
```

post process 建议继续拆 logits：

```python
with _ttft_timer("execute.post_process_total"):
    with record_function_or_nullcontext("post process"):
        ...

        with _ttft_timer("execute.gather_sample_hidden_states"):
            sample_hidden_states = hidden_states[logits_indices]

        with _ttft_timer("execute.compute_logits"):
            logits = self.model.compute_logits(sample_hidden_states)
```

如果不想大改缩进，可以先只打三段：

```text
execute.prepare_input
execute.target_forward
execute.post_process_total
```

这三段足以判断 target prefill 是否是 TTFT 主因。

### 4.2 拆 `sample_tokens()`

位置：

```text
vllm_ascend/worker/model_runner_v1.py:2089
```

sample：

```python
with _ttft_timer("sample_tokens.sample"):
    with record_function_or_nullcontext("sample_token"):
        sampler_output = self._sample(logits, spec_decode_metadata)
```

bookkeeping：

```python
with _ttft_timer("sample_tokens.bookkeeping_sync"):
    (
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
        invalid_req_indices,
    ) = self._bookkeeping_sync(
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        scheduler_output.total_num_scheduled_tokens,
        spec_decode_metadata,
    )
```

draft token：

```python
with _ttft_timer("sample_tokens.draft_token_total"):
    with record_function_or_nullcontext("draft_token"):
        if self.speculative_config:
            ...
```

KV connector finalize 可以在 `draft_token` 内再拆：

```python
with _ttft_timer("sample_tokens.finalize_kv_connector"):
    self.finalize_kv_connector()
```

output 构造：

```python
with _ttft_timer("sample_tokens.build_output"):
    model_runner_output = ModelRunnerOutput(...)
```

异步状态更新：

```python
with _ttft_timer("sample_tokens.async_state_update"):
    with (
        record_function_or_nullcontext("async_state_update"),
        torch.npu.stream(global_stream()),
    ):
        ...
```

### 4.3 完整 TTFT 读数解释

如果看到：

| 大头 | 说明 |
| --- | --- |
| `execute.prepare_input` 很大 | scheduler output 处理、padding、metadata、输入预处理可能是瓶颈 |
| `execute.target_forward` 很大 | target model prefill/forward 是主因 |
| `execute.compute_logits` 很大 | vocab logits 或 TP/lmhead 路径重 |
| `sample_tokens.sample` 很大 | sampling、grammar mask、top-k/top-p、CPU 往返可能重 |
| `sample_tokens.bookkeeping_sync` 很大 | 输出同步、logprobs、CPU 拷贝、请求状态维护重 |
| `sample_tokens.draft_token_total` 很大 | speculative/DFlash 是主因 |
| `sample_tokens.finalize_kv_connector` 很大 | KV connector/外部 KV 传输影响 |

如果对比 DFlash 和 MTP，重点看：

```text
sample_tokens.draft_token_total
```

如果 DFlash 明显大于 MTP，继续拆 DFlash 内部。

## 5. DFlash 内部怎么拆

### 5.1 拆 `_propose()`

文件：

```text
vllm_ascend/spec_decode/llm_base_proposer.py
```

位置：

```text
_propose(): around line 559
```

DFlash/EAGLE3 会执行：

```python
target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
```

建议打点：

```python
if self.method in ("eagle3", "dflash"):
    with _dflash_timer("_propose.combine_hidden_states"):
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
```

继续拆：

```python
with _dflash_timer("_propose.set_inputs_first_pass"):
    num_tokens, token_indices_to_sample, common_attn_metadata, long_seq_args = (
        self.set_inputs_first_pass(...)
    )

with _dflash_timer("_propose.run_merged_draft"):
    draft_token_ids = self._run_merged_draft(...)
```

注意：`llm_base_proposer.py` 当前没有 `_dflash_timer`，可以复制第 3 节的 timer，或者从 `dflash_proposer.py` 导入。为了避免循环 import，建议复制小工具函数。

### 5.2 拆 `AscendDflashProposer.set_inputs_first_pass()`

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

位置：

```text
set_inputs_first_pass(): line 63
```

建议打印 shape：

```python
if _DFLASH_PROFILE:
    print(
        "[DFlashProfile] "
        f"batch_size={batch_size}, "
        f"num_context={num_context}, "
        f"num_query_per_req={num_query_per_req}, "
        f"num_query_total={num_query_total}, "
        f"target_hidden_states={tuple(target_hidden_states.shape)}",
        flush=True,
    )
```

拆 hidden states copy：

```python
with _dflash_timer("set_inputs_first_pass.copy_hidden_states"):
    self._dflash_num_context = num_context
    self._dflash_hidden_states[:num_context] = target_hidden_states
```

拆 input expand kernel：

```python
with _dflash_timer("set_inputs_first_pass.input_expand_kernel"):
    copy_and_expand_dflash_inputs_kernel_single_grid[1,](
        ...
    )
```

这两段能判断：

- 大 hidden states copy 是否进入 TTFT。
- single-grid input expand 是否仍是高并发瓶颈。

### 5.3 拆 `build_model_inputs_first_pass()`

文件：

```text
vllm_ascend/spec_decode/dflash_proposer.py
```

位置：

```text
build_model_inputs_first_pass(): line 248
```

建议改成：

```python
def build_model_inputs_first_pass(self, num_input_tokens: int) -> dict[str, Any]:
    num_context = self._dflash_num_context

    if _DFLASH_PROFILE:
        print(
            "[DFlashProfile] "
            f"build_model_inputs_first_pass: "
            f"num_context={num_context}, "
            f"num_input_tokens={num_input_tokens}, "
            f"hidden_shape={tuple(self._dflash_hidden_states[:num_context].shape)}, "
            f"context_positions_shape={tuple(self._context_positions_buffer[:num_context].shape)}, "
            f"context_slot_mapping_shape={tuple(self._context_slot_mapping_buffer[:num_context].shape)}",
            flush=True,
        )

    with _dflash_timer("build_model_inputs_first_pass.precompute_context_kv"):
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )

    return dict(
        input_ids=self.input_ids[:num_input_tokens],
        positions=self.positions[:num_input_tokens],
        inputs_embeds=None,
    )
```

如果这一段很大，DFlash 首 token 慢基本就落在 context KV precompute 上。

### 5.4 拆 `_run_merged_draft()`

文件：

```text
vllm_ascend/spec_decode/llm_base_proposer.py
```

位置：

```text
_run_merged_draft(): line 907
```

建议：

```python
if self.method == "dflash":
    with _dflash_timer("_run_merged_draft.build_model_inputs_first_pass"):
        model_kwargs = self.build_model_inputs_first_pass(num_input_tokens)
else:
    ...

with _dflash_timer("_run_merged_draft.draft_model_forward"):
    ret_hidden_states = self.model(**model_kwargs)
```

采样 draft token ids：

```python
with _dflash_timer("_run_merged_draft.compute_draft_token_ids"):
    draft_token_ids = self.compute_draft_token_ids(sample_hidden_states)
```

解释：

- `build_model_inputs_first_pass` 大：DFlash context KV 预处理重。
- `draft_model_forward` 大：DFlash query forward 重。
- `compute_draft_token_ids` 大：lm_head/logits/sampling 路径重。

### 5.5 拆 `precompute_and_store_context_kv()`

文件：

```text
vllm_ascend/patch/worker/patch_qwen3_dflash.py
```

位置：

```text
precompute_and_store_context_kv(): line 6
```

建议完整拆成：

```python
with _dflash_timer("precompute.build_fused_kv_buffers"):
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

num_ctx = context_states.shape[0]
L = self._num_attn_layers
kv = self._kv_size
hd = self._head_dim
nkv = self._num_kv_heads

if _DFLASH_PROFILE:
    print(
        "[DFlashProfile] "
        f"precompute shape: num_ctx={num_ctx}, L={L}, "
        f"kv={kv}, nkv={nkv}, hd={hd}, "
        f"context_states={tuple(context_states.shape)}, "
        f"context_positions={tuple(context_positions.shape)}, "
        f"has_slot_mapping={context_slot_mapping is not None}",
        flush=True,
    )

with _dflash_timer("precompute.hidden_norm"):
    normed_context_states = self.hidden_norm(context_states)

with _dflash_timer("precompute.fused_kv_linear"):
    all_kv_flat = F.linear(
        normed_context_states,
        self._fused_kv_weight,
        self._fused_kv_bias,
    )

with _dflash_timer("precompute.view_permute_contiguous"):
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(
        2, 1, 0, 3, 4
    ).contiguous()
    all_k = all_kv[0]
    all_v = all_kv[1]

with _dflash_timer("precompute.k_norm_loop"):
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

with _dflash_timer("precompute.positions_repeat"):
    positions_repeated = context_positions.repeat(L)

with _dflash_timer("precompute.rope"):
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

if context_slot_mapping is None:
    return

with _dflash_timer("precompute.kv_cache_update_loop"):
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    for i in range(L):
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            context_slot_mapping,
        )
```

重点看：

| 字段 | 含义 |
| --- | --- |
| `num_ctx` | DFlash 预处理的 context token 数，长 prompt/高并发会放大 |
| `L` | DFlash 需要预计算的 attention layer 数 |
| `kv` | 每 token 每层 K 或 V 的维度 |
| `precompute.fused_kv_linear` | hidden states 到所有层 K/V 的大 GEMM |
| `precompute.view_permute_contiguous` | 大 tensor 重排和 contiguous copy |
| `precompute.k_norm_loop` | 逐层 K RMSNorm |
| `precompute.rope` | RoPE 和当前 clone 开销 |
| `precompute.kv_cache_update_loop` | 逐层写 draft KV cache |

## 6. 如何启动

只拆完整 TTFT：

```bash
export VLLM_ASCEND_TTFT_PROFILE=1
python -m vllm.entrypoints.openai.api_server ...
```

只拆 DFlash：

```bash
export VLLM_ASCEND_DFLASH_PROFILE=1
python -m vllm.entrypoints.openai.api_server ...
```

两者都拆：

```bash
export VLLM_ASCEND_TTFT_PROFILE=1
export VLLM_ASCEND_DFLASH_PROFILE=1
python -m vllm.entrypoints.openai.api_server ...
```

建议先固定压测参数：

```text
batch/concurrency 固定
prompt length 固定
max_tokens=1 或很小
temperature=0
同一套请求分别跑 MTP、DFlash、无投机
```

## 7. 日志应该怎么读

### 7.1 判断 DFlash 是否是 TTFT 主因

先看：

```text
[TTFTProfile] sample_tokens.draft_token_total
```

如果 DFlash 比 MTP 或无投机大很多，再看：

```text
[DFlashProfile] build_model_inputs_first_pass.precompute_context_kv
[DFlashProfile] _run_merged_draft.draft_model_forward
```

### 7.2 判断是不是 context KV precompute 主因

如果：

```text
build_model_inputs_first_pass.precompute_context_kv
```

接近或占据大部分：

```text
sample_tokens.draft_token_total
```

说明 DFlash 首轮慢主要是 hidden states -> draft KV cache 的预处理。

继续看：

```text
precompute.fused_kv_linear
precompute.view_permute_contiguous
precompute.k_norm_loop
precompute.rope
precompute.kv_cache_update_loop
```

哪个最大，优化方向就对应哪里。

### 7.3 判断是不是模型配置差异

对 qwen3-30b-a3b 和 qwen3.6-35b-a3b 分别记录：

```text
num_ctx
L
kv
nkv
hd
target_hidden_states.shape
num_query_per_req
num_query_total
```

如果 qwen3.6-35b-a3b 的 `num_ctx * L * kv` 明显更大，那么它 DFlash 首 token 慢很多就是预期结果。

如果形状接近，但 qwen3.6-35b-a3b 仍然慢很多，则重点怀疑：

- 是否没有命中 graph。
- 是否走了不同 attention/cache update backend。
- 是否 patch 路径触发了更多 fallback。
- 是否 `combine_hidden_states` 更重。
- 是否某些 NPU 算子在 qwen3.6 的 shape 下性能退化。

## 8. 和 profiler 配合使用

同步日志能快速定位“哪一段慢”，但不能直接告诉你“哪个 NPU kernel 慢”。定位到大段后，再开 profiler。

已有代码中已经有很多：

```python
record_function_or_nullcontext("prepare input")
record_function_or_nullcontext("forward")
record_function_or_nullcontext("post process")
record_function_or_nullcontext("sample_token")
record_function_or_nullcontext("draft_token")
```

你新增的 `_ttft_timer` 和 `_dflash_timer` 主要用于打印 ms。  
如果要看算子级细节，建议用 vllm-ascend 自带 profiler 或 torch-npu profiler 抓：

```text
forward
draft_token
precompute_context_kv
```

对应的慢段。

## 9. 注意事项

1. 同步 timer 会破坏异步流水，因此只用于定位，不用于最终性能数据。
2. 高并发时多 worker 打印会很多，可以只在 rank 0 打印。
3. 如果使用 ACL graph，强同步 timer 不一定影响 graph capture，但会影响 replay 外的调度节奏；建议 profiling 和正式压测分开跑。
4. 打点不要长期保留在热路径里，或者必须受环境变量控制。
5. 对比两个模型时，必须保证 batch、prompt length、max_tokens、spec tokens、graph 配置、并发压测方式一致。

## 10. 推荐排查顺序

1. 打开 `VLLM_ASCEND_TTFT_PROFILE=1`，确认完整 TTFT 中哪段最大。
2. 如果 `sample_tokens.draft_token_total` 最大，打开 `VLLM_ASCEND_DFLASH_PROFILE=1`。
3. 看 `build_model_inputs_first_pass.precompute_context_kv` 是否是 DFlash 大头。
4. 如果是，继续拆 `precompute_and_store_context_kv()`。
5. 对比 qwen3-30b-a3b 和 qwen3.6-35b-a3b 的 `num_ctx/L/kv/nkv/hd`。
6. 如果 shape 接近但耗时差异大，用 profiler 看具体 NPU kernel。
7. 根据大头选择优化：
   - input expand 大：看 2-D grid 或 threshold skip。
   - hidden copy 大：看减少 copy/引用 buffer。
   - fused linear 大：看 DFlash window、减少 target layers、降低触发阈值。
   - RoPE 大：修返回值、去 clone、换更合适的 fused op。
   - cache update 大：看批量 cache update、减少逐层 Python loop。

