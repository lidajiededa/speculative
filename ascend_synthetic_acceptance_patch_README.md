# Ascend 上固定投机接受长度/接受率的 Patch 步骤

本文说明如何在 `vllm-ascend v0.20.2rc1` 上打一个性能压测用 patch，使 speculative decoding 不按真实 draft/target token 是否一致来决定接受，而是通过开关选择固定每轮接受的 draft token 数量，或固定每个 speculative 位置的无条件接受率。

适用场景：

- 只测推理性能、吞吐、ITL、TPOT。
- 不关心输出文本质量。
- 希望隔离投机接受率/接受长度对性能的影响。

不适用场景：

- 质量评测。
- 正式线上推理。
- 需要保证输出语义正确的任务。

## 1. 背景

上游 vLLM `v0.20.2` 已有 `rejection_sample_method="synthetic"`，可以通过 synthetic rejection sampling 人为控制接受率。

但当前 `vllm-ascend v0.20.2rc1` 的 NPU 路径并未完整接通该能力：

- `vllm_ascend/sample/rejection_sampler.py` 的 `rejection_sample()` 函数签名虽然预留了 `synthetic_mode` 和 `synthetic_conditional_rates`，但内部没有真正使用。
- `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` 中明确写了 NPU 暂不支持 synthetic rejection sampling。

因此推荐打一个更简单、稳定、适合性能测试的 patch：同时支持固定接受长度和每位置无条件接受率。

重要说明：这个 patch 不会减少当前轮 target model 的验证计算。target verify 仍然会按调度到的 speculative tokens 跑完整的 `1 + K` token forward；patch 只是在验证完成后的 rejection sampler 阶段忽略真实比较结果，改写“接受多少 token”的结果。因此它影响的是每轮 decode 实际推进 token 数、后续轮次数、吞吐和延迟统计，不是把当前轮验证算子裁掉。

### 1.1 与上游 vLLM synthetic 逻辑的关系

本文 patch 参考了上游 vLLM 的 synthetic rejection sampling 逻辑，但不是逐行照搬。

上游相关代码在：

- `vllm/config/speculative.py`
- `vllm/v1/spec_decode/utils.py`
- `vllm/v1/sample/rejection_sampler.py`
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `vllm/v1/worker/gpu/spec_decode/synthetic_rejection_sampler_utils.py`

上游逻辑可以概括为：

1. `SpeculativeConfig` 支持 `rejection_sample_method="synthetic"`。
2. 用户必须二选一传入 `synthetic_acceptance_rates` 或 `synthetic_acceptance_length`。
3. 如果传入 `synthetic_acceptance_length`，上游会先将它解析成每位置无条件接受率：

```python
num_drafts = length - 1
num_full = int(num_drafts)
rates = [1.0] * num_full + [num_drafts - num_full] + [0.0] * ...
```

4. 然后通过 `unconditional_to_conditional_rates()` 转成每位置条件接受率：

```python
q_i = p_i / p_{i-1}, p_{-1}=1
```

5. rejection sampler 在每个 speculative 位置抛随机数：
   - synthetic 模式下：`accepted = uniform_prob < q_i`
   - 一旦某个位置拒绝，后续位置停止接受
   - 如果所有 draft token 都接受，则追加 bonus token

因此，上游的 `synthetic_acceptance_rates` 语义是“每位置无条件接受率”，不是每个位置独立接受率。

本文 patch 的 `rates` 模式基本沿用这个思路：读取上游已经解析好的 `synthetic_acceptance_rates`，转成条件接受率，再在 Ascend greedy rejection 路径中构造 prefix accept mask。

本文 patch 的 `fixed` 模式是针对压测额外加的稳定模式。它利用上游 `synthetic_acceptance_length` 会被解析为 rates 这一点，取 `sum(rates)` 并 round 成固定接受的 draft token 数。对于整数长度，例如 `synthetic_acceptance_length=4.0`，它与上游 synthetic 行为等价于固定接受 3 个 draft token；对于小数长度，例如 `4.5`，上游会在第 4 个 draft 位置以 0.5 概率接受，而本文 `fixed` 模式会 round 成固定长度，因此更稳定，但不再严格模拟上游的随机均值行为。

## 2. 目标行为

假设：

- `num_speculative_tokens = K`
- `synthetic_acceptance_length = L`

则 patch 后：

- 当 `VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=fixed` 时，每轮固定接受 `round(L - 1)` 个 draft token。
- 如果固定接受数为 `K`，则等价于全接受 draft，并追加 bonus token。
- 如果固定接受数为 `0`，则等价于完全不接受 draft，只输出 target token。

例如 K=8：

| synthetic_acceptance_length | 固定接受 draft token 数 | 每轮最多输出 token 数 |
| --- | ---: | ---: |
| 1.0 | 0 | 1 |
| 2.0 | 1 | 2 |
| 4.0 | 3 | 4 |
| 9.0 | 8 | 9 |

当 `VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=rates` 时：

- 使用 `synthetic_acceptance_rates=[p0,p1,...]` 固定每个位置的无条件接受率。
- `p_i` 表示前 `i + 1` 个 draft token 都被接受的概率。
- sampler 内部转换为条件接受率并随机采样，因此结果有方差，但更接近真实接受率曲线。

当 `VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=off` 时：

- 保留代码 patch，但不覆盖真实验收结果。
- 适合在不回滚源码的情况下切回真实 rejection sampling。

## 3. 修改文件

基于当前 workspace：

- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\worker\model_runner_v1.py`
- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\sample\rejection_sampler.py`

如果你在真实部署环境中 patch，路径通常是 Python site-packages 下的：

- `.../site-packages/vllm_ascend/worker/model_runner_v1.py`
- `.../site-packages/vllm_ascend/sample/rejection_sampler.py`

## 4. Patch 1：Runner 传入 speculative_config

文件：

```text
vllm_ascend/worker/model_runner_v1.py
```

找到：

```python
self.rejection_sampler = AscendRejectionSampler(self.sampler)
```

改成：

```python
self.rejection_sampler = AscendRejectionSampler(
    self.sampler,
    self.speculative_config,
)
```

作用：让 Ascend rejection sampler 能读取 `rejection_sample_method` 和 `synthetic_acceptance_rates`。

## 5. Patch 2：Sampler 读取 synthetic 配置

文件：

```text
vllm_ascend/sample/rejection_sampler.py
```

先在文件顶部加入：

```python
import os
```

找到 `AscendRejectionSampler.__init__`：

```python
def __init__(self, sampler):
    super().__init__(sampler)
    # Store Ascend-specific optimizations
    self._ascend_optimizations_enabled = True
    self.top_k = None
```

改成：

```python
def __init__(self, sampler, spec_config=None):
    super().__init__(sampler)
    # Store Ascend-specific optimizations
    self._ascend_optimizations_enabled = True
    self.top_k = None
    self.synthetic_acceptance_mode = "off"
    self.synthetic_fixed_accept_len = None
    self.synthetic_conditional_rates = None

    if (
        spec_config is not None
        and spec_config.rejection_sample_method == "synthetic"
        and spec_config.synthetic_acceptance_rates is not None
    ):
        self.synthetic_acceptance_mode = os.getenv(
            "VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE",
            "fixed",
        ).lower()
        if self.synthetic_acceptance_mode not in ("fixed", "rates", "off"):
            raise ValueError(
                "VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE must be one of "
                "'fixed', 'rates', or 'off'."
            )

        # Upstream SpeculativeConfig resolves synthetic_acceptance_length into
        # per-position unconditional rates. Their sum is the expected number
        # of accepted draft tokens.
        fixed_len = int(round(sum(spec_config.synthetic_acceptance_rates)))
        self.synthetic_fixed_accept_len = max(
            0,
            min(fixed_len, spec_config.num_speculative_tokens),
        )

        # For random synthetic acceptance mode, convert unconditional
        # per-position rates into conditional rates:
        #   q_i = P(accept position i | all previous positions accepted)
        #       = p_i / p_{i-1}, p_{-1}=1.
        rates = list(spec_config.synthetic_acceptance_rates)
        conditional_rates = []
        prev = 1.0
        for rate in rates:
            conditional_rates.append(0.0 if prev <= 0.0 else rate / prev)
            prev = rate
        self.synthetic_conditional_rates = conditional_rates
```

说明：

- 上游 `SpeculativeConfig` 会把 `synthetic_acceptance_length` 转成 `synthetic_acceptance_rates`。
- 对固定长度测试而言，只需要取 `sum(rates)`，再 round 成固定接受的 draft token 数。

## 6. Patch 3A：统一 sampler patch

本节实现统一的 sampler patch，同时支持 `fixed`、`rates` 和 `off` 三种模式。`fixed` 最适合做稳定的阶梯压测，例如固定接受 0/1/3/5/8 个 draft token。

### 6.1 Sampler 调用 rejection_sample 时传入 synthetic payload

同一文件：

```text
vllm_ascend/sample/rejection_sampler.py
```

找到 `AscendRejectionSampler.forward()` 里的调用：

```python
output_token_ids = rejection_sample(
    metadata.draft_token_ids,
    metadata.num_draft_tokens,
    metadata.max_spec_len,
    metadata.cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    sampling_metadata,
)
```

改成：

```python
synthetic_payload = None
if (
    self.synthetic_acceptance_mode == "fixed"
    and self.synthetic_fixed_accept_len is not None
):
    synthetic_payload = torch.tensor(
        self.synthetic_fixed_accept_len,
        dtype=torch.int32,
        device=target_logits.device,
    )
elif (
    self.synthetic_acceptance_mode == "rates"
    and self.synthetic_conditional_rates is not None
):
    synthetic_payload = torch.tensor(
        self.synthetic_conditional_rates,
        dtype=torch.float32,
        device=target_logits.device,
    )

output_token_ids = rejection_sample(
    metadata.draft_token_ids,
    metadata.num_draft_tokens,
    metadata.max_spec_len,
    metadata.cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    sampling_metadata,
    synthetic_mode=self.synthetic_acceptance_mode,
    synthetic_conditional_rates=synthetic_payload,
)
```

### 6.2 rejection_sample 中按 mode 伪造 target_argmax

同一文件：

```text
vllm_ascend/sample/rejection_sampler.py
```

先把 `rejection_sample()` 签名里的 synthetic 参数改成字符串模式：

```python
synthetic_mode: str = "off",
synthetic_conditional_rates: torch.Tensor | None = None,
```

在 `rejection_sample()` 中找到 greedy 分支：

```python
# For greedy sampling, we need to do allgather first to get global argmax
if not sampling_metadata.all_random:
    if get_ascend_config().enable_reduce_sample:
        target_argmax = greedy_sample(target_logits)
    else:
        target_argmax = target_logits.argmax(dim=-1).view(-1)
```

替换为：

```python
# For greedy sampling, we need to do allgather first to get global argmax.
# In synthetic mode, bypass real argmax comparison and force accept/reject.
if not sampling_metadata.all_random:
    if synthetic_mode != "off" and synthetic_conditional_rates is not None:
        draft_counts = torch.tensor(
            num_draft_tokens,
            device=device,
            dtype=torch.long,
        )
        start_indices = cu_num_draft_tokens.to(torch.long) - draft_counts
        req_ids = torch.arange(batch_size, device=device)
        token_req_ids = torch.repeat_interleave(req_ids, draft_counts)
        pos_in_req = (
            torch.arange(num_tokens, device=device)
            - start_indices[token_req_ids]
        )

        if synthetic_mode == "fixed":
            fixed_accept_len = int(synthetic_conditional_rates.item())
            accept_mask = pos_in_req < fixed_accept_len
        elif synthetic_mode == "rates":
            rates_for_token = synthetic_conditional_rates[pos_in_req]
            uniform_probs = torch.rand(
                num_tokens,
                device=device,
                dtype=torch.float32,
            )
            raw_accept = uniform_probs < rates_for_token

            # A later token can only be accepted if all previous tokens in the
            # same request were accepted. cumprod implements the prefix rule.
            accept_int = raw_accept.to(torch.int32)
            accept_matrix = torch.zeros(
                (batch_size, max_spec_len),
                device=device,
                dtype=torch.int32,
            )
            valid_pos = pos_in_req < max_spec_len
            accept_matrix[token_req_ids[valid_pos], pos_in_req[valid_pos]] = (
                accept_int[valid_pos]
            )
            prefix_accept_matrix = torch.cumprod(
                accept_matrix,
                dim=1,
            ).bool()
            accept_mask = prefix_accept_matrix[token_req_ids, pos_in_req]
        else:
            raise ValueError(f"Unknown synthetic_mode: {synthetic_mode}")

        forced_reject = torch.where(
            draft_token_ids == 0,
            draft_token_ids + 1,
            draft_token_ids - 1,
        ).to(draft_token_ids.dtype)
        target_argmax = torch.where(
            accept_mask,
            draft_token_ids,
            forced_reject,
        )
    elif get_ascend_config().enable_reduce_sample:
        target_argmax = greedy_sample(target_logits)
    else:
        target_argmax = target_logits.argmax(dim=-1).view(-1)
```

原理：

- Ascend 原有 greedy rejection 逻辑会比较 `draft_token_ids` 和 `target_argmax`。
- patch 后：
  - `fixed` 模式下，前 `fixed_accept_len` 个位置让 `target_argmax = draft_token_ids`，因此一定接受；后续位置让 `target_argmax` 变成一个与 `draft_token_ids` 不同的 token id，因此一定拒绝。
  - `rates` 模式下，先按每位置条件接受率随机生成 prefix accept mask，再用同样方式强制接受或拒绝。
- 后续仍走原来的 `rejection_greedy_sample_with_triton()` 或 PyTorch fallback，因此整体代码路径仍接近真实推理。

## 7. Patch 3B：每个位置无条件接受率模式

第 6 节的统一代码已经同时支持每位置无条件接受率模式，不需要再替换另一套 sampler 代码。你只需要：

1. 启动前设置 `VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=rates`。
2. 在 `--speculative-config` 中传入 `synthetic_acceptance_rates`。

例如：

```json
"synthetic_acceptance_rates": [0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.06, 0.03]
```

它的含义是：

- 第 `i` 个 speculative 位置的无条件接受率为 `p_i`。
- 也就是 `P(前 i 个 draft token 都被接受) = p_i`。
- `AscendRejectionSampler.__init__` 内部会转成条件接受率：
  - `q_0 = p_0`
  - `q_i = p_i / p_{i-1}`
- 每个 request 在每轮 rejection sampling 中，从左到右按 `q_i` 抛随机数；一旦某个位置拒绝，后续位置全部拒绝。

这样能更接近真实 speculative decoding 的接受分布，但会引入 `torch.rand` 和少量 tensor 操作开销。用于性能压测时建议记录这部分额外开销，或与固定长度模式一起对照。

注意：这段仍然是在 target verify 完成后生效，不会减少当前轮验证开销。该实现主要面向 greedy 压测，即 `temperature=0`。

## 8. 两种 patch 模式怎么选

| 模式 | 开关 | 配置 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| 固定接受长度 | `VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=fixed` | `synthetic_acceptance_length` | 性能结果稳定，容易解释 | 不模拟真实接受分布 |
| 每位置无条件接受率 | `VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=rates` | `synthetic_acceptance_rates` | 更接近真实接受率曲线 | 有随机数开销，结果有方差 |
| 关闭 synthetic override | `VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=off` | 无 | 保留 patch 后回到真实验收 | 不固定接受率 |

如果目标是画系统性能上限/下限曲线，先用固定接受长度。

如果目标是模拟某个 draft head 的真实接受率分布，用每位置无条件接受率。

## 9. 对其他投机方法的生效范围

这个改动不依赖 DFlash proposer 本身，而是改在 v1 推理路径的 `AscendRejectionSampler` 里。因此只要某个投机方法最终走到：

```text
model_runner_v1.py::_sample()
  -> self.rejection_sampler(...)
  -> AscendRejectionSampler.forward()
  -> rejection_sample()
```

就会受到这个 synthetic acceptance patch 影响。

在 `vllm-ascend v0.20.2rc1` 里，`vllm_ascend/spec_decode/__init__.py` 注册的这些 method 都会通过同一个 v1 runner 创建 drafter，并在 target verify 之后进入 `AscendRejectionSampler`：

| method | proposer | 是否理论生效 | 备注 |
| --- | --- | --- | --- |
| `dflash` | `AscendDflashProposer` | 是 | 本文主要示例。 |
| `eagle` / `eagle3` / `mtp` | `AscendEagleProposer` | 是 | 只要最终 metadata 进入同一个 sampler。 |
| `draft_model` | `AscendDraftModelProposer` | 是 | 目标模型验证后仍走 rejection sampler。 |
| `ngram` / `ngram_gpu` | `AscendNgramProposer` / `AscendNgramProposerNPU` | 是 | `draft_probs` 可能为 `None`，但 greedy 验收仍通过 `target_argmax` 比较。 |
| `suffix` | `AscendSuffixDecodingProposer` | 是 | 与 ngram 类似，注意真实验收语义会被覆盖。 |
| `medusa` | `AscendMedusaProposer` | 需实测 | 如果输出 metadata 后仍进入同一个 sampler，则会生效；树形/特殊验收路径需要单独确认。 |
| `extract_hidden_states` | `AscendExtractHiddenStatesProposer` | 通常不作为验收压测目标 | 主要用于抽取 hidden states，不建议用这个 patch 解释投机性能。 |

不生效或需要另打 patch 的情况：

- 不走 `model_runner_v1.py` 的 v1 推理路径，例如某些 v2 worker 路径。当前 `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` 仍显式提示 NPU 不支持 synthetic rejection sampling。
- 非 greedy sampling。本文 patch 的核心是在 greedy 分支伪造 `target_argmax`，`temperature > 0` 会进入随机 rejection 逻辑，除非额外 patch random sampling 分支。
- 某个投机方法如果有自己独立的 acceptance kernel、树形 verify 或绕过 `AscendRejectionSampler`，则不会自动生效，需要沿它自己的验收输出位置打 patch。

一句话：这个 patch 是“验收 sampler 层”的通用改动，不是 DFlash 专属；但它只覆盖同一个 Ascend v1 rejection sampler 代码链路。

## 10. 推荐启动参数

以 DFlash K=8 为例：

```bash
export VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=fixed

vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --speculative-config '{"method":"dflash","model":"/path/to/dflash-head","num_speculative_tokens":8,"rejection_sample_method":"synthetic","synthetic_acceptance_length":4.0}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,18,27,36]}'
```

建议测试矩阵：

```text
synthetic_acceptance_length = 1.0  # 固定接受 0 个 draft
synthetic_acceptance_length = 2.0  # 固定接受 1 个 draft
synthetic_acceptance_length = 4.0  # 固定接受 3 个 draft
synthetic_acceptance_length = 6.0  # 固定接受 5 个 draft
synthetic_acceptance_length = 9.0  # K=8，全接受 + bonus
```

注意：`synthetic_acceptance_length` 的合法范围是 `[1, K + 1]`。

每位置接受率模式示例：

```bash
export VLLM_ASCEND_SYNTHETIC_ACCEPTANCE_MODE=rates

vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --speculative-config '{"method":"dflash","model":"/path/to/dflash-head","num_speculative_tokens":8,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.8,0.6,0.45,0.3,0.2,0.12,0.06,0.03]}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,18,27,36]}'
```

## 11. 验证方法

### 11.1 看日志和 metrics

启动后观察 Prometheus metrics 或 benchmark 输出中的 speculative decoding 指标：

- `vllm:spec_decode_num_drafts`
- `vllm:spec_decode_num_draft_tokens`
- `vllm:spec_decode_num_accepted_tokens`
- `vllm:spec_decode_num_accepted_tokens_per_pos`

如果固定接受长度为 3，则前 3 个位置的接受计数应接近 draft 次数，后续位置应接近 0。

### 11.2 用短 prompt sanity check

使用 `temperature=0`，生成少量 token。

预期：

- 输出文本可能明显错误。
- 性能指标正常输出。
- speculative metrics 与固定接受长度一致。

### 11.3 对比基线

建议至少跑这些组：

1. 不开 DFlash。
2. 开 DFlash，真实 rejection sampling。
3. 开 DFlash，固定接受长度为 0/1/3/5/8。

这样能区分：

- draft head 额外成本。
- target 一次验证多 token 的收益。
- acceptance length 对 throughput/latency 的影响。

## 12. 性能解释口径

固定接受长度或每位置 synthetic rates 都不会跳过 target 的 verify forward。

对 K=8，当前轮 target 仍然会验证 `1 + 8 = 9` 个 token。patch 改的是 rejection sampler 输出的 accepted token 数：

- 接受越多，每轮推进越多 token，需要的 decode iteration 越少。
- 接受越少，每轮推进越少 token，需要的 decode iteration 越多。
- 单轮 verify compute 基本不变，但端到端吞吐和平均时延会随推进 token 数改变。

因此它适合回答的问题是：

> 在固定 speculative verify 宽度 K 下，如果接受长度/接受率是某个值，系统端到端性能是多少？

它不适合回答：

> 如果验证阶段也根据接受长度提前停止，性能是多少？

后者需要额外实现动态裁剪 verify token 或 D-Cut 类优化，不是本 patch 的目标。

## 13. 限制与注意事项

1. 这个 patch 主要适合 greedy 解码，即 `temperature=0`。
2. 非 greedy sampling 下仍会进入随机 rejection 逻辑，不建议用该 patch 做非 greedy 压测。
3. 输出文本不可信。
4. 如果打开 logprobs，logprobs 可能与人为接受逻辑不完全符合质量评测语义。
5. 如果 vllm-ascend 升级到完整支持 upstream synthetic rejection sampling，应优先使用官方实现。
6. 随机每位置接受率模式会增加 `torch.rand` 和 prefix mask 构造开销；固定接受长度模式开销更低。

## 14. 回滚方法

回滚两个改动即可：

1. `model_runner_v1.py` 中恢复：

```python
self.rejection_sampler = AscendRejectionSampler(self.sampler)
```

2. `rejection_sampler.py` 中：

- `AscendRejectionSampler.__init__` 恢复单参数版本。
- `AscendRejectionSampler.forward()` 恢复原始 `rejection_sample(...)` 调用。
- `rejection_sample()` greedy 分支恢复真实 `target_argmax` 计算。

如果是在 git 仓库中修改，可以用：

```bash
git diff
git checkout -- vllm_ascend/worker/model_runner_v1.py
git checkout -- vllm_ascend/sample/rejection_sampler.py
```

正式部署环境中不要用 `git checkout`，建议先备份原文件再替换。
