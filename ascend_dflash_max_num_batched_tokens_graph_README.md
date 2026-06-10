# Ascend DFlash 首 token 时延与 max_num_batched_tokens / 图模式配置说明

本文总结一次实际问题定位结论：在 `vllm-ascend v0.20.2rc1` 上部署 DFlash speculative decoding 时，`max_num_batched_tokens=4096` 会导致高并发首 token 时延异常变差；将其调大到 `16384` 后，首 token 时延恢复正常。

相关源码基于当前 workspace：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1
D:\workspace\speculative\vllm-v0.20.2
```

## 1. 现象

实验现象：

- 开启 DFlash 时，`max_num_batched_tokens=4096` 下高并发首 token 时延明显恶化。
- 将 `max_num_batched_tokens` 从 `4096` 调到 `16384` 后，首 token 时延明显恢复。
- 这个收益不只出现在长 prompt；即使 prompt 不长，高并发下也有明显收益。
- 不开启 DFlash 时，同样修改 `max_num_batched_tokens`，性能影响没有这么大。

结论：

`4096` 在 DFlash 场景下不是一个普通的 batch token 上限，而是会被 speculative decoding 的 draft slot 预留显著扣减。高并发时，实际可用于 target/prefill/decode 的调度预算可能已经很小，导致请求被切成很多小 chunk，进而放大调度、metadata、graph padding、DFlash first pass 等开销。

## 2. 根因总结

开启 speculative decoding 后，vLLM 不会把 `max_num_batched_tokens` 全部交给 scheduler 使用，而是会预留一部分 slots 给 drafter。

核心公式：

```text
max_num_scheduled_tokens
  = max_num_batched_tokens
    - max_num_new_slots_for_drafting * max_num_seqs
```

DFlash 会设置 `parallel_drafting=True`，因此：

```text
max_num_new_slots_for_drafting = num_speculative_tokens - 1
```

所以 DFlash 下实际可调度 token 预算近似为：

```text
effective_scheduler_budget
  = max_num_batched_tokens
    - max_num_seqs * (num_speculative_tokens - 1)
```

例如：

```text
max_num_batched_tokens = 4096
max_num_seqs = 256
num_speculative_tokens = 15
```

则：

```text
effective_scheduler_budget = 4096 - 256 * 14 = 512
```

这意味着你以为每轮可以调度 4096 个 token，但 DFlash 开启后，scheduler 实际可能只有 512 个 token 预算。这在高并发下非常容易造成大量 chunked prefill / 小 batch step。

如果改成：

```text
max_num_batched_tokens = 16384
max_num_seqs = 256
num_speculative_tokens = 15
```

则：

```text
effective_scheduler_budget = 16384 - 256 * 14 = 12800
```

这时调度预算恢复到健康区间，所以首 token 时延显著改善。

## 3. 为什么不开 DFlash 时影响不大

不开 DFlash 时：

- 不需要为 DFlash parallel drafting 预留 `max_num_seqs * (num_speculative_tokens - 1)` slots。
- 首 token 返回前不会同步执行 DFlash draft first pass。
- 不会执行 DFlash 的 `hidden states -> context KV cache` 预处理。
- 没有 `batch_size * (1 + num_speculative_tokens)` 的 draft query 放大。

因此 `max_num_batched_tokens=4096` 在非 DFlash 场景下仍然大体可用；但在 DFlash 场景下，它会被 draft slot 预留吃掉大部分。

## 4. 相关源码

### 4.1 speculative decoding 会扣减调度预算

文件：

```text
D:\workspace\speculative\vllm-v0.20.2\vllm\config\vllm.py
```

函数：

```text
VllmConfig._set_max_num_scheduled_tokens()
```

关键代码：

```python
if self.speculative_config is not None:
    scheduled_token_delta = (
        self.speculative_config.max_num_new_slots_for_drafting
        * self.scheduler_config.max_num_seqs
    )
    max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
    if self.scheduler_config.max_num_scheduled_tokens is None:
        self.scheduler_config.max_num_scheduled_tokens = (
            max_num_batched_tokens - scheduled_token_delta
        )
```

含义：

- `max_num_batched_tokens` 是总容量。
- `scheduled_token_delta` 是为 drafter 预留的额外 slots。
- `max_num_scheduled_tokens` 才是 scheduler 每轮真正可用于调度请求 token 的预算。

### 4.2 DFlash 会开启 parallel drafting

文件：

```text
D:\workspace\speculative\vllm-v0.20.2\vllm\config\speculative.py
```

关键代码：

```python
if self.method == "dflash":
    self.parallel_drafting = True
```

含义：

DFlash 一次并行生成多个 speculative tokens，而不是串行逐 token draft。因此它需要额外的 parallel draft slots。

### 4.3 parallel drafting 的预留 slots 数

文件：

```text
D:\workspace\speculative\vllm-v0.20.2\vllm\config\speculative.py
```

函数：

```text
SpeculativeConfig.max_num_new_slots_for_drafting
```

关键代码：

```python
slots_per_req = 0
if self.parallel_drafting:
    # For parallel drafting, we need one new slot per 'masked' token
    slots_per_req = self.num_speculative_tokens - 1
if self.uses_draft_model():
    slots_per_req += 1
return slots_per_req
```

含义：

DFlash 下通常：

```text
max_num_new_slots_for_drafting = num_speculative_tokens - 1
```

这会乘以 `max_num_seqs`，从 `max_num_batched_tokens` 中扣除。

### 4.4 DFlash 首轮会构造 K+1 个 query tokens

文件：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\spec_decode\dflash_proposer.py
```

函数：

```text
AscendDflashProposer.set_inputs_first_pass()
```

关键代码：

```python
batch_size = cad.num_reqs
num_context = target_token_ids.shape[0]
num_query_per_req = 1 + self.num_speculative_tokens
num_query_total = batch_size * num_query_per_req
```

含义：

DFlash first pass 里，每个请求不只是一个 query，而是：

```text
1 个 bonus token + K 个 mask/speculative tokens
```

也就是 `1 + num_speculative_tokens`。

这也是为什么高并发时 DFlash 对 batch token 预算、graph bucket、padding 更敏感。

### 4.5 DFlash 会额外分配 max_query_tokens 和 hidden buffer

文件：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\spec_decode\dflash_proposer.py
```

关键代码：

```python
self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
self.max_positions = self.max_num_tokens + self.max_query_tokens

self._dflash_hidden_states = torch.zeros(
    (self.max_num_tokens, self.hidden_size),
    dtype=self.dtype,
    device=self.device,
)
```

含义：

`max_num_batched_tokens` 也会影响 DFlash proposer 的 buffer 上限。  
过小会影响调度预算，过大则会增加预分配 buffer 和 graph capture 的内存压力。

## 5. 为什么高并发下收益尤其明显

高并发时，`max_num_seqs` 通常较大，DFlash 预留槽位按请求数线性放大：

```text
reserved_draft_slots = max_num_seqs * (num_speculative_tokens - 1)
```

所以并发越高，`max_num_batched_tokens=4096` 越容易被吃空。

例如：

| max_num_seqs | num_speculative_tokens | reserved slots | max_num_batched_tokens=4096 后剩余 |
| --- | --- | --- | --- |
| 64 | 15 | 896 | 3200 |
| 128 | 15 | 1792 | 2304 |
| 256 | 15 | 3584 | 512 |
| 512 | 15 | 7168 | 不合法或无法正常调度 |

因此“prompt 不长也有收益”是合理的。因为瓶颈不是单个 prompt 是否长，而是高并发下每轮总调度 token 预算是否被 DFlash 预留槽位压缩。

## 6. 推荐配置原则

### 6.1 max_num_batched_tokens

DFlash 下建议按下面公式估算：

```text
max_num_batched_tokens
  >= max_num_seqs * (num_speculative_tokens - 1)
     + target_scheduler_budget
```

其中：

```text
target_scheduler_budget
```

表示你希望每轮真正交给 scheduler 的 token 数。

经验建议：

| 场景 | 建议 |
| --- | --- |
| 低并发、小 K | `8192` 起步 |
| 高并发、K=8~15 | `16384` 起步 |
| 高并发、长 prompt、K=15 | `16384` 或更高 |
| 如果剩余预算低于 8192 | 需要重点关注首 token 和吞吐 |

可以直接计算：

```text
effective_scheduler_budget
  = max_num_batched_tokens
    - max_num_seqs * (num_speculative_tokens - 1)
```

建议让这个值至少保持在：

```text
8192 或更高
```

尤其是高并发服务场景。

### 6.2 max_num_seqs

如果不能继续增大 `max_num_batched_tokens`，可以降低：

```text
max_num_seqs
```

因为 DFlash 的 reserved slots 与 `max_num_seqs` 线性相关。

### 6.3 num_speculative_tokens

如果高并发下首 token 或内存压力较大，可以降低：

```text
num_speculative_tokens
```

例如从 `15` 降到 `8` 或 `4`。  
这会降低 reserved slots，也会减少 DFlash first pass 的 query tokens。

## 7. 图模式档位建议

开启 DFlash 时，推荐优先使用：

```text
cudagraph_mode = FULL_AND_PIECEWISE
```

原因：

- DFlash decode/draft 的 query length 是固定的 `1 + num_speculative_tokens`，适合 FULL graph。
- prefill 或 mixed prefill-decode 形状变化更大，更适合 PIECEWISE。
- vLLM 的 dispatcher 会把 speculative decoding 的 uniform decode query length 设置为 `K + 1`。

相关源码：

```text
D:\workspace\speculative\vllm-v0.20.2\vllm\v1\cudagraph_dispatcher.py
```

关键代码：

```python
self.uniform_decode_query_len = (
    1
    if not self.vllm_config.speculative_config
    else 1 + self.vllm_config.speculative_config.num_speculative_tokens
)
```

当 batch 是 uniform decode 且 FULL graph 可用时，会按 `uniform_decode_query_len` 计算 request 数：

```python
if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
    num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
    assert num_tokens_padded % uniform_decode_query_len == 0
```

## 8. 图模式推荐顺序

| 档位 | 推荐程度 | 说明 |
| --- | --- | --- |
| `FULL_AND_PIECEWISE` | 首选 | decode/DFlash draft 用 FULL，mixed/prefill 用 PIECEWISE，通常最适合 DFlash |
| `FULL_DECODE_ONLY` | 保守生产选项 | 如果 mixed PIECEWISE 有问题，保留 decode FULL graph 收益 |
| `PIECEWISE` | 排查/兼容选项 | 形状更灵活，但 decode/DFlash draft 固定形状收益可能不如 FULL |
| `NONE` | 调试正确性 | 用于排查 graph capture/replay 问题，不建议作为最终性能配置 |

建议策略：

```text
1. 先用 FULL_AND_PIECEWISE。
2. 如果 ACL graph capture/replay 不稳定，退到 FULL_DECODE_ONLY。
3. 如果仍不稳定，退到 PIECEWISE。
4. 只有定位问题时使用 NONE。
```

## 9. 建议启动配置示例

示例，仅保留关键参数：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/target_model \
  --speculative-config '{
    "model": "/path/to/dflash_model",
    "method": "dflash",
    "num_speculative_tokens": 15
  }' \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{
    "cudagraph_mode": "FULL_AND_PIECEWISE"
  }'
```

如果线上图模式不稳定，可以改为：

```json
{
  "cudagraph_mode": "FULL_DECODE_ONLY"
}
```

## 10. 验证方法

建议在启动日志或临时打点中打印：

```text
max_num_batched_tokens
max_num_seqs
num_speculative_tokens
max_num_new_slots_for_drafting
max_num_scheduled_tokens
effective_scheduler_budget
cudagraph_mode
cudagraph_capture_sizes
max_cudagraph_capture_size
```

重点确认：

```text
max_num_scheduled_tokens
```

是否低于 `8192`。如果低于，vLLM 本身也会提示性能可能不是最优。

可以在 `VllmConfig._set_max_num_scheduled_tokens()` 后临时打印：

```python
logger.warning(
    "DFlash budget: max_num_batched_tokens=%s, max_num_seqs=%s, "
    "num_speculative_tokens=%s, max_num_new_slots_for_drafting=%s, "
    "max_num_scheduled_tokens=%s",
    self.scheduler_config.max_num_batched_tokens,
    self.scheduler_config.max_num_seqs,
    self.speculative_config.num_speculative_tokens,
    self.speculative_config.max_num_new_slots_for_drafting,
    self.scheduler_config.max_num_scheduled_tokens,
)
```

## 11. 最终结论

这次首 token 问题的直接解决方法是：

```text
max_num_batched_tokens: 4096 -> 16384
```

根因是：

```text
DFlash 开启 parallel drafting 后，会按 max_num_seqs * (num_speculative_tokens - 1)
从 max_num_batched_tokens 中预留 draft slots，导致 4096 下实际 scheduler budget
被压得过小，高并发时大量请求被切成小 chunk，首 token 时延被显著放大。
```

因此，DFlash 部署时不要只看 `max_num_batched_tokens` 的原始值，而要看扣除 draft slots 后的：

```text
max_num_scheduled_tokens / effective_scheduler_budget
```

图模式上，优先使用：

```text
FULL_AND_PIECEWISE
```

如果 NPU ACL graph 稳定性有问题，再逐级退到：

```text
FULL_DECODE_ONLY -> PIECEWISE -> NONE
```

## 12. cudagraph_capture_sizes 计算与配置

`cudagraph_capture_sizes` 表示图模式要捕获的 **batch token 数档位**，不是请求数档位。

普通 decode 时，每个请求一轮通常只有 1 个 query token：

```text
capture_size ~= 并发请求数 * 1
```

但开启 DFlash / speculative decoding 后，一轮 uniform decode 或 draft 相关 forward 的 query length 变成：

```text
uniform_decode_query_len = 1 + num_speculative_tokens
```

相关源码：

```text
D:\workspace\speculative\vllm-v0.20.2\vllm\v1\cudagraph_dispatcher.py
```

关键代码：

```python
self.uniform_decode_query_len = (
    1
    if not self.vllm_config.speculative_config
    else 1 + self.vllm_config.speculative_config.num_speculative_tokens
)
```

DFlash proposer 里也会构造同样的形状：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\spec_decode\dflash_proposer.py
```

关键代码：

```python
num_query_per_req = 1 + self.num_speculative_tokens
num_query_total = batch_size * num_query_per_req
```

因此 DFlash 下 FULL graph 对应的 token 档位应该按下面公式估算：

```text
capture_size = 并发请求数 * (num_speculative_tokens + 1)
```

例如：

```text
num_speculative_tokens = 15
uniform_decode_query_len = 16
```

则：

| 并发请求数 | DFlash 每轮 token 数 | 建议覆盖的 capture size |
| --- | --- | --- |
| 32 | 32 * 16 = 512 | 512 |
| 64 | 64 * 16 = 1024 | 1024 |
| 128 | 128 * 16 = 2048 | 2048 |
| 256 | 256 * 16 = 4096 | 4096 |

所以如果线上高并发目标是：

```text
max_num_seqs = 256
num_speculative_tokens = 15
```

则建议至少让 `max_cudagraph_capture_size` 覆盖到：

```text
4096
```

而不是使用默认较小的 `512`。

### 12.1 为什么不直接设成 16384

`max_num_batched_tokens=16384` 是 scheduler 总 token 预算，用来避免 DFlash 预留 slots 后 `max_num_scheduled_tokens` 太小。

但 `max_cudagraph_capture_size=16384` 代表捕获非常大的 graph，可能带来：

- 启动和 graph capture 时间明显变长。
- ACL graph 内存占用增加。
- 大 bucket padding 更重。
- NPU graph replay 稳定性风险增加。

因此两者不应该简单设成一样。

推荐理解为：

```text
max_num_batched_tokens:
  控制 scheduler 总 token 容量，DFlash 下建议较大，例如 16384。

max_cudagraph_capture_size:
  控制 graph 捕获的最大 token 档位，主要覆盖常见 decode/DFlash draft 并发形状。
```

例如：

```text
max_num_batched_tokens = 16384
max_num_seqs = 256
num_speculative_tokens = 15
max_cudagraph_capture_size = 4096
```

这是一个比较合理的组合。

### 12.2 capture sizes 推荐配置

`cudagraph_capture_sizes` 最好都是：

```text
num_speculative_tokens + 1
```

的整数倍。vLLM 在 spec decode 场景下会做 round up，但手动配成整数倍更清晰，也更容易确认实际命中的 graph 档位。

对于 `num_speculative_tokens=15`，即 `K+1=16`，可以使用：

```json
{
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "max_cudagraph_capture_size": 4096,
  "cudagraph_capture_sizes": [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096
  ]
}
```

如果主要并发低于 128，可以先用：

```json
{
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "max_cudagraph_capture_size": 2048,
  "cudagraph_capture_sizes": [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048
  ]
}
```

### 12.3 capture size 与图命中

图模式 dispatch 时，如果本轮 token 数超过 `max_cudagraph_capture_size`，则不会命中 cudagraph：

```text
num_tokens > max_cudagraph_capture_size -> CUDAGraphMode.NONE
```

相关源码：

```text
D:\workspace\speculative\vllm-v0.20.2\vllm\v1\cudagraph_dispatcher.py
```

关键代码：

```python
if (
    not self.keys_initialized
    or self.cudagraph_mode == CUDAGraphMode.NONE
    or max_size is None
    or num_tokens > max_size
    or allowed_modes <= {CUDAGraphMode.NONE}
):
    return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```

所以 DFlash 下如果 `max_cudagraph_capture_size` 仍是默认 512，而实际高并发 draft/decode token 数是 2048 或 4096，就会频繁无法命中 FULL graph。

### 12.4 启动配置示例

高并发 DFlash，`K=15`，目标覆盖 256 并发：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/target_model \
  --speculative-config '{
    "model": "/path/to/dflash_model",
    "method": "dflash",
    "num_speculative_tokens": 15
  }' \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "max_cudagraph_capture_size": 4096,
    "cudagraph_capture_sizes": [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
  }'
```

如果 `FULL_AND_PIECEWISE` 在 NPU ACL graph 上不稳定，可以先退到：

```json
{
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "max_cudagraph_capture_size": 4096,
  "cudagraph_capture_sizes": [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
}
```
