# vLLM-Ascend DFlash 阈值禁用投机 Patch 说明

本文只说明“高并发或长输入时直接不走 DFlash 投机”的实现方法。

这里的目标不是“只跳过首轮 DFlash draft”，而是：

- 当前 batch 并发数高时，不再产生下一轮 DFlash draft tokens。
- 当前 batch 输入/上下文长度长时，不再产生下一轮 DFlash draft tokens。
- 阈值持续命中时，后续轮次也持续不投机。
- 阈值不命中时，恢复原来的 DFlash 投机逻辑。

## 1. 结论

可以实现，而且只需要修改一个文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

需要改两处：

| 位置 | 当前行附近 | 修改内容 |
| --- | ---: | --- |
| 文件 import | 20-23 | 增加 `import os` |
| `sample_tokens()` 的 `draft_token` 块 | 2152-2227 | 增加阈值判断，命中时清空 draft，不调用 `propose_draft_token_ids()` |

不需要修改：

```text
vllm_ascend/spec_decode/dflash_proposer.py
vllm_ascend/spec_decode/llm_base_proposer.py
vllm_ascend/ops/triton/spec_decode/utils.py
```

原因是这个 patch 不改变 DFlash 内部实现，只是在 Python 调度层决定“本轮是否继续为下一轮生成 draft tokens”。

## 2. 需要先理解的语义

当前调用链是：

```text
model_runner_v1.py::sample_tokens
  -> sample_token
  -> _bookkeeping_sync
  -> draft_token
      -> propose_draft_token_ids(...)
          -> self.propose_draft_token_ids(...)
              -> drafter._propose(...)
                  -> DFlash proposer / graph dispatch / draft model forward
```

`draft_token` 发生在 target model forward 之后。因此：

- 如果当前轮没有上一轮 draft tokens，命中阈值后可以直接不投机。
- 如果当前轮已经带着上一轮 draft tokens 在验证，那么验证成本已经发生了；本 patch 会阻止继续产生下一轮 draft tokens。
- 所以这是“下一轮开始不投机”的调度层禁用。通常最多需要一轮把已有 draft 消耗掉。

如果想在 scheduler 层连“当前轮的验证”也完全避免，需要更早地在调度器选择 tokens 前禁用 speculative slots；那是另一条更大的改动，不属于本文这个最小 patch。

## 3. 具体修改 1：增加 `os` import

文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

当前约 20-23 行：

```python
import math
import sys
import time
from collections import defaultdict
```

改成：

```python
import math
import os
import sys
import time
from collections import defaultdict
```

为什么：

- 用环境变量做 A/B 开关。
- 不需要先改 `AscendConfig`，风险小，方便快速回退。

## 4. 具体修改 2：新增清空 draft 的本地函数

文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

当前位置：`sample_tokens()` 中，当前约 2152-2167 行：

```python
def propose_draft_token_ids(sampled_token_ids):
    assert spec_decode_common_attn_metadata is not None
    self._draft_token_ids = self.propose_draft_token_ids(
        sampled_token_ids,
        self.input_batch.sampling_metadata,
        scheduler_output,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        positions,
        scheduler_output.total_num_scheduled_tokens,
        hidden_states,
        aux_hidden_states,
        sample_hidden_states,
        batch_desc,
    )
    self._copy_draft_token_ids_to_cpu(scheduler_output)
```

在这个函数后面新增：

```python
def clear_draft_token_ids_for_this_round():
    # Keep the scheduler-side draft buffer valid, but report zero draft
    # tokens for this round. This prevents stale draft tokens from the
    # previous step from being reused when DFlash is disabled by threshold.
    self._draft_token_ids = torch.zeros(
        1,
        device=self.device,
        dtype=torch.int32,
    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
    self._copy_draft_token_ids_to_cpu(
        scheduler_output,
        zeros_only=True,
    )
```

为什么不能写成：

```python
self._draft_token_ids = None
```

因为 `_copy_draft_token_ids_to_cpu()` 里如果发现 `_draft_token_ids` 不是 tensor，会直接 return。这样 CPU 侧 draft buffer 可能保留上一轮 stale draft tokens，scheduler 可能继续拿旧 draft。

当前 Ascend fallback 和上游 vLLM 都是用 zero tensor + `zeros_only=True` 清空 draft。

## 5. 具体修改 3：计算“禁用 DFlash 投机”的阈值条件

当前位置：`draft_token` 块中，当前约 2185-2200 行：

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
```

建议在 `use_padded_batch` 后面新增：

```python
dflash_disable_batch_threshold = int(os.getenv(
    "VLLM_ASCEND_DFLASH_DISABLE_BATCH_THRESHOLD",
    "0",
))
dflash_disable_seq_len_threshold = int(os.getenv(
    "VLLM_ASCEND_DFLASH_DISABLE_SEQ_LEN_THRESHOLD",
    "0",
))
dflash_disable_scheduled_token_threshold = int(os.getenv(
    "VLLM_ASCEND_DFLASH_DISABLE_SCHEDULED_TOKEN_THRESHOLD",
    "0",
))

current_max_seq_len = (
    spec_decode_common_attn_metadata.max_seq_len
    if spec_decode_common_attn_metadata is not None
    else 0
)

# Batch-level policy: while any threshold is hit, do not generate
# DFlash draft tokens for the next scheduler round. This disables DFlash
# speculation for high-concurrency or long-context batches without changing
# the target model forward path.
disable_dflash_by_threshold = (
    self.speculative_config.use_dflash()
    and (
        (
            dflash_disable_batch_threshold > 0
            and self.input_batch.num_reqs >= dflash_disable_batch_threshold
        )
        or (
            dflash_disable_seq_len_threshold > 0
            and current_max_seq_len >= dflash_disable_seq_len_threshold
        )
        or (
            dflash_disable_scheduled_token_threshold > 0
            and scheduler_output.total_num_scheduled_tokens
            >= dflash_disable_scheduled_token_threshold
        )
    )
)
```

注意这里**不要再加“当前没有 speculative decode metadata”这个首轮判断**。

原因：

- 加上它就只会跳过首轮。
- 现在目标是阈值命中时持续不投机，所以无论当前是不是 speculative verify 轮次，都不能继续生成下一轮 draft。

三个阈值的含义：

| 环境变量 | 判断对象 | 适合场景 |
| --- | --- | --- |
| `VLLM_ASCEND_DFLASH_DISABLE_BATCH_THRESHOLD` | `self.input_batch.num_reqs` | 并发数高时关闭 DFlash |
| `VLLM_ASCEND_DFLASH_DISABLE_SEQ_LEN_THRESHOLD` | `spec_decode_common_attn_metadata.max_seq_len` | 上下文长度长时关闭 DFlash |
| `VLLM_ASCEND_DFLASH_DISABLE_SCHEDULED_TOKEN_THRESHOLD` | `scheduler_output.total_num_scheduled_tokens` | chunked prefill / 一轮 token 总量大时关闭 DFlash |

推荐至少保留 `SEQ_LEN_THRESHOLD`，因为用户说的“输入长度长”更接近 `max_seq_len`，而不是只看本轮 scheduled tokens。

## 6. 具体修改 4：命中阈值时短路 DFlash proposer

当前位置：当前约 2201-2227 行：

```python
if use_padded_batch:
    # EAGLE speculative decoding can use the GPU sampled tokens
    # as inputs, and does not need to wait for bookkeeping to finish.
    sampled_token_ids = sampler_output.sampled_token_ids
    if input_fits_in_drafter:
        propose_draft_token_ids(sampler_output.sampled_token_ids)
    elif self.valid_sampled_token_count_event is not None:
        ...
if self.speculative_config and not use_padded_batch and input_fits_in_drafter:
    # ngram and other speculative decoding methods use the sampled
    # tokens on the CPU, so they are run after bookkeeping.
    propose_draft_token_ids(valid_sampled_token_ids)
```

建议改成：

```python
if use_padded_batch:
    # EAGLE/DFlash-style speculative decoding can use the device-side
    # sampled tokens directly.
    sampled_token_ids = sampler_output.sampled_token_ids
    if disable_dflash_by_threshold:
        clear_draft_token_ids_for_this_round()
    elif input_fits_in_drafter:
        propose_draft_token_ids(sampler_output.sampled_token_ids)
    elif self.valid_sampled_token_count_event is not None:
        assert spec_decode_common_attn_metadata is not None
        if self.drafter is not None:  # Fix mypy type check for drafter None check
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids,
                valid_sampled_tokens_count,
            )
            clear_draft_token_ids_for_this_round()

if self.speculative_config and not use_padded_batch:
    # DFlash normally uses the padded-batch path. This fallback keeps the
    # threshold policy safe if the config changes later.
    if disable_dflash_by_threshold:
        clear_draft_token_ids_for_this_round()
    elif input_fits_in_drafter:
        # ngram and other speculative decoding methods use the sampled
        # tokens on the CPU, so they are run after bookkeeping.
        propose_draft_token_ids(valid_sampled_token_ids)
```

为什么这样改：

- 命中阈值时，不进入 `self.propose_draft_token_ids()`，因此不会进入 DFlash proposer。
- `clear_draft_token_ids_for_this_round()` 会清空 CPU 侧 draft buffer，避免 scheduler 使用旧 draft。
- 非 DFlash 方法不受影响，因为 `disable_dflash_by_threshold` 要求 `use_dflash()`。

## 7. 最终代码形态

`sample_tokens()` 中相关部分最终应类似这样：

```python
def propose_draft_token_ids(sampled_token_ids):
    assert spec_decode_common_attn_metadata is not None
    self._draft_token_ids = self.propose_draft_token_ids(
        sampled_token_ids,
        self.input_batch.sampling_metadata,
        scheduler_output,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        positions,
        scheduler_output.total_num_scheduled_tokens,
        hidden_states,
        aux_hidden_states,
        sample_hidden_states,
        batch_desc,
    )
    self._copy_draft_token_ids_to_cpu(scheduler_output)

def clear_draft_token_ids_for_this_round():
    # Keep the scheduler-side draft buffer valid, but report zero draft
    # tokens for this round. This prevents stale draft tokens from the
    # previous step from being reused when DFlash is disabled by threshold.
    self._draft_token_ids = torch.zeros(
        1,
        device=self.device,
        dtype=torch.int32,
    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
    self._copy_draft_token_ids_to_cpu(
        scheduler_output,
        zeros_only=True,
    )

...

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

        dflash_disable_batch_threshold = int(os.getenv(
            "VLLM_ASCEND_DFLASH_DISABLE_BATCH_THRESHOLD",
            "0",
        ))
        dflash_disable_seq_len_threshold = int(os.getenv(
            "VLLM_ASCEND_DFLASH_DISABLE_SEQ_LEN_THRESHOLD",
            "0",
        ))
        dflash_disable_scheduled_token_threshold = int(os.getenv(
            "VLLM_ASCEND_DFLASH_DISABLE_SCHEDULED_TOKEN_THRESHOLD",
            "0",
        ))
        current_max_seq_len = (
            spec_decode_common_attn_metadata.max_seq_len
            if spec_decode_common_attn_metadata is not None
            else 0
        )
        disable_dflash_by_threshold = (
            self.speculative_config.use_dflash()
            and (
                (
                    dflash_disable_batch_threshold > 0
                    and self.input_batch.num_reqs >= dflash_disable_batch_threshold
                )
                or (
                    dflash_disable_seq_len_threshold > 0
                    and current_max_seq_len >= dflash_disable_seq_len_threshold
                )
                or (
                    dflash_disable_scheduled_token_threshold > 0
                    and scheduler_output.total_num_scheduled_tokens
                    >= dflash_disable_scheduled_token_threshold
                )
            )
        )

        if use_padded_batch:
            sampled_token_ids = sampler_output.sampled_token_ids
            if disable_dflash_by_threshold:
                clear_draft_token_ids_for_this_round()
            elif input_fits_in_drafter:
                propose_draft_token_ids(sampler_output.sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                assert spec_decode_common_attn_metadata is not None
                if self.drafter is not None:
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_indices.gpu,
                            self.num_discarded_requests,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids,
                        valid_sampled_tokens_count,
                    )
                    clear_draft_token_ids_for_this_round()

        if self.speculative_config and not use_padded_batch:
            if disable_dflash_by_threshold:
                clear_draft_token_ids_for_this_round()
            elif input_fits_in_drafter:
                propose_draft_token_ids(valid_sampled_token_ids)

    if self.speculative_config is not None:
        self.finalize_kv_connector()
```

## 8. 启动参数

默认不开启，因为阈值默认都是 `0`。

高并发时关闭 DFlash：

```bash
export VLLM_ASCEND_DFLASH_DISABLE_BATCH_THRESHOLD=64
```

长上下文时关闭 DFlash：

```bash
export VLLM_ASCEND_DFLASH_DISABLE_SEQ_LEN_THRESHOLD=4096
```

本轮 scheduled tokens 过多时关闭 DFlash：

```bash
export VLLM_ASCEND_DFLASH_DISABLE_SCHEDULED_TOKEN_THRESHOLD=8192
```

三个阈值可以同时设置，满足任意一个就关闭 DFlash：

```bash
export VLLM_ASCEND_DFLASH_DISABLE_BATCH_THRESHOLD=64
export VLLM_ASCEND_DFLASH_DISABLE_SEQ_LEN_THRESHOLD=4096
export VLLM_ASCEND_DFLASH_DISABLE_SCHEDULED_TOKEN_THRESHOLD=8192
```

关闭该策略：

```bash
export VLLM_ASCEND_DFLASH_DISABLE_BATCH_THRESHOLD=0
export VLLM_ASCEND_DFLASH_DISABLE_SEQ_LEN_THRESHOLD=0
export VLLM_ASCEND_DFLASH_DISABLE_SCHEDULED_TOKEN_THRESHOLD=0
```

## 9. 对图模式的影响

不会破坏主模型 graph。

原因：

- 主模型 forward 已经在 `draft_token` 前执行完。
- 该 patch 只决定是否调用 `propose_draft_token_ids()`。
- 命中阈值时，DFlash drafter graph 本轮不 dispatch。
- 未命中阈值时，DFlash 仍按原逻辑 dispatch graph/eager。

需要注意：

- 如果当前轮已经在验证上一轮 draft，主模型 verify 成本已经发生，本 patch 无法回退这部分成本。
- 命中阈值后不会继续生成下一轮 draft，因此通常下一轮开始就不再有 DFlash verify。
- 如果阈值来回抖动，会出现 DFlash 开关频繁切换；建议阈值设置得保守一些，或者后续增加 hysteresis。

## 10. 对其他投机方法的影响

默认不影响其他投机方法。

关键条件是：

```python
self.speculative_config.use_dflash()
```

因此只有 DFlash 会被阈值禁用。EAGLE、MTP、ngram、draft model 等仍按原逻辑执行。

如果希望所有投机方法都按阈值禁用，可以把条件改成：

```python
self.speculative_config is not None
```

但不建议一开始这么做，因为不同投机方法的开销和收益不同，统一禁用会影响面更大。

## 11. 建议验证

至少测 5 组：

```text
不开阈值
只开 batch threshold
只开 seq len threshold
只开 scheduled token threshold
三个 threshold 同时开
```

建议记录：

```text
TTFT p50/p90/p99
ITL p50/p90/p99
整体吞吐 tokens/s
DFlash draft 被禁用的 batch 数
投机接受率
```

为了确认是否命中阈值，可以临时加 debug log：

```python
if disable_dflash_by_threshold:
    logger.debug(
        "Disable DFlash speculation by threshold: num_reqs=%s, "
        "max_seq_len=%s, total_tokens=%s, batch_threshold=%s, "
        "seq_len_threshold=%s, scheduled_token_threshold=%s",
        self.input_batch.num_reqs,
        current_max_seq_len,
        scheduler_output.total_num_scheduled_tokens,
        dflash_disable_batch_threshold,
        dflash_disable_seq_len_threshold,
        dflash_disable_scheduled_token_threshold,
    )
```

生产环境建议不要高频打印，避免日志本身影响性能。
