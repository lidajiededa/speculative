# vLLM-Ascend DFlash 投机解码推理代码链路

本文基于本 workspace 中的两份源码：

- vllm-ascend: `D:\workspace\speculative\vllm-ascend-0.20.2rc1`
- upstream vLLM: `D:\workspace\speculative\vllm-v0.20.2`

关注问题：`vllm-ascend v0.20.2rc1` 支持 DFlash 投机解码后，以 `Qwen/Qwen3-8B` 和 `Qwen3.6-35B-A3B` 类模型为例，部署带 DFlash 投机头时推理会走哪些上游 vLLM 与 vllm-ascend 代码链路。

## 1. 一句话结论

DFlash 在 vLLM 中被接入为一种 EAGLE-style speculative method，但它不是串行自回归 draft。它的核心是：

1. target model 先正常 forward 并采样出一个 `next_token`。
2. DFlash proposer 将 target 的多层 hidden states 作为 context。
3. DFlash draft model 先把 context hidden states 预投影成 K/V，写入 draft KV cache。
4. draft query 只包含 `1 + K` 个 token：第 0 个是 target 刚采样出的 bonus token，后面 K 个是 `mask_token_id`。
5. draft model 一次非因果/并行 forward 生成 K 个候选 token。
6. 下一轮 target 一次验证 `1 + K` 个 token，rejection sampler 接受最长匹配前缀，若全接受则追加 bonus token。

其中上游 vLLM 提供 DFlash 的配置解析、通用 V1 runner、通用 DFlash proposer、DFlash Qwen3 draft model、rejection sampler；vllm-ascend 在这些路径上替换为 NPU runner、Ascend proposer、Ascend attention metadata、Triton/Ascend kernel、NPU KV cache 写入与 graph 适配。

## 2. 部署配置形态

### Qwen3-8B

vllm-ascend 的 E2E 测试使用：

- target: `Qwen/Qwen3-8B`
- draft: `z-lab/Qwen3-8B-DFlash-b16`
- method: `dflash`
- `num_speculative_tokens`: `8`

示例：

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-8B-DFlash-b16","num_speculative_tokens":8}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,18]}'
```

`cudagraph_capture_sizes` 需要按 `batch_size * (1 + num_speculative_tokens)` 配。例如 K=8 时，batch 1/2 对应 `[9,18]`。

### Qwen3.6-35B-A3B

这类模型要区分 target 与 draft：

- target 是大模型本体，通常是 MoE/VLM/Qwen3.x 路径。
- DFlash draft head 是单独训练和保存的轻量 draft checkpoint。
- 推理侧 draft config 需要能被 vLLM 识别成 DFlash draft model，例如 `DFlashQwen3ForCausalLM` 或 speculator config 中的 `DFlashDraftModel`。
- 如果 target 是 Qwen3.6/Qwen3.5 VLM/MoE，必须额外核对 `target_layer_ids`、`target_hidden_size`、M-RoPE/position 语义、special token 和 `mask_token_id`。

示例形态：

```bash
vllm serve /models/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --speculative-config '{"method":"dflash","model":"/models/qwen36-35b-a3b-dflash-head","num_speculative_tokens":8}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,18,27,36]}'
```

## 3. 上游 vLLM 代码链路

### 3.1 参数解析与 SpeculativeConfig

入口文件：

- `vllm\engine\arg_utils.py`
- `vllm\config\speculative.py`

关键逻辑：

- `--speculative-config` 由 `EngineArgs.create_speculative_config()` 构造成 `SpeculativeConfig`。
- `SpeculativeConfig` 支持 `DFlashModelTypes = Literal["dflash"]`。
- 如果用户显式传 `"method":"dflash"`，直接保留。
- 如果未显式传 method，但 draft model 名称中包含 `dflash`，自动推断 `method = "dflash"`。
- 对 `eagle/eagle3/dflash`，会把 draft hf config 包成 `EAGLEConfig(method=...)`。
- 对 DFlash，强制 `parallel_drafting = True`。
- `use_eagle()` 对 `dflash` 返回 true；另外有 `use_dflash()` 专门判断。

相关位置：

- `vllm\config\speculative.py`: `DFlashModelTypes`
- `vllm\config\speculative.py`: method 自动推断 `"dflash" in draft_model_config.model.lower()`
- `vllm\config\speculative.py`: `if self.method == "dflash": self.parallel_drafting = True`
- `vllm\config\speculative.py`: `use_eagle()` / `use_dflash()`

### 3.2 DFlash architecture 映射

入口文件：

- `vllm\transformers_utils\configs\eagle.py`
- `vllm\transformers_utils\configs\speculators\algos.py`
- `vllm\model_executor\models\registry.py`

两种常见路径：

1. 常规 HF draft config 经 `EAGLEConfig(method="dflash")` 包装后，architecture 会加 `DFlash` 前缀。例如 `Qwen3ForCausalLM -> DFlashQwen3ForCausalLM`。
2. Speculators config 的 `@register_speculator("dflash")` 会把 architecture 设成 `DFlashDraftModel`，同时填充 `dflash_config.target_layer_ids` 和 `eagle_aux_hidden_state_layer_ids`。

模型注册表中：

```python
"DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM")
```

因此最终 draft 模型类落到：

- `vllm\model_executor\models\qwen3_dflash.py`
- `DFlashQwen3ForCausalLM`

### 3.3 上游 GPUModelRunner 初始化 drafter

入口文件：

- `vllm\v1\worker\gpu_model_runner.py`

关键逻辑：

- import `DFlashProposer`
- 如果 `self.speculative_config.use_dflash()`，创建：

```python
self.drafter = DFlashProposer(self.vllm_config, self.device, self)
```

- 同时创建 `RejectionSampler`，用于 target verify 后的接受/拒绝。

对应代码位置：

- `vllm\v1\worker\gpu_model_runner.py`: import `DFlashProposer`
- `vllm\v1\worker\gpu_model_runner.py`: `_set_up_drafter()`

### 3.4 上游 DFlashProposer

入口文件：

- `vllm\v1\spec_decode\dflash.py`
- `vllm\v1\spec_decode\llm_base_proposer.py`
- `vllm\v1\spec_decode\utils.py`

`DFlashProposer` 继承 `SpecDecodeBaseProposer`，但重写 DFlash 专属的第一轮输入构造。

关键函数：

- `DFlashProposer.__init__()`
- `DFlashProposer.set_inputs_first_pass()`
- `DFlashProposer.build_model_inputs_first_pass()`
- `DFlashProposer.build_per_group_and_layer_attn_metadata()`

关键行为：

1. `pass_hidden_states_to_model=True`，说明 DFlash 需要 target hidden states。
2. `max_query_tokens = max_batch_size * (1 + K)`，DFlash draft forward 只看 bonus/mask 这些 query token。
3. 单独维护 context positions、context slot mapping、query positions、query slot mapping。
4. `set_inputs_first_pass()` 中：
   - `num_query_per_req = 1 + num_speculative_tokens`
   - 保存 `_dflash_hidden_states = target_hidden_states`
   - 调用 `copy_and_expand_dflash_inputs_kernel`
   - 第 0 个 query 写入 `next_token_ids`
   - 后续 query 写入 `parallel_drafting_token_id`
   - attention metadata 设置 `causal=False`
5. `build_model_inputs_first_pass()` 中：
   - 先调用 `self.model.precompute_and_store_context_kv(...)`
   - 然后 draft model forward 只处理 query token。

上游 `SpecDecodeBaseProposer` 还负责：

- 从 `dflash_config.mask_token_id` 读取 `parallel_drafting_token_id`
- 对 `eagle3/dflash` 调用 draft model 的 `combine_hidden_states()`
- 因为 `parallel_drafting=True`，只跑一次 draft forward，然后直接返回 `[batch, K]` 的 draft tokens。

### 3.5 上游 DFlash Qwen3 draft model

入口文件：

- `vllm\model_executor\models\qwen3_dflash.py`

主要类：

- `DFlashQwen3Attention`
- `DFlashQwen3DecoderLayer`
- `DFlashQwen3Model`
- `DFlashQwen3ForCausalLM`

关键行为：

1. `DFlashQwen3ForCausalLM` 从 `vllm_config.speculative_config.draft_model_config.hf_config` 取 draft config。
2. `DFlashQwen3Model` 会读取：
   - `dflash_config`
   - `target_layer_ids`
   - `use_aux_hidden_state`
   - `target_hidden_size`
3. 如果使用 aux hidden state，`fc` 会把拼接后的 target hidden states 压回 draft hidden size。
4. `_build_fused_kv_buffers()` 把所有 draft layer 的 K/V projection 权重拼成 fused buffer。
5. `precompute_and_store_context_kv()`：
   - 对 context hidden 做 RMSNorm
   - 一次大 GEMM 得到所有层 K/V
   - K 做 per-layer RMSNorm
   - 对 K 做 RoPE
   - 逐层写入 KV cache
6. `forward()` 只处理 query token。
7. `compute_logits()` 用 draft lm head 产生 token；如果 draft vocab 与 target vocab 不一致，使用 `draft_id_to_target_id` 映射。

这就是 DFlash 与普通 EAGLE 的根本差异：context hidden 不作为普通输入 token 进入 draft forward，而是先变成 draft KV cache。

### 3.6 上游 target verify 与 rejection sampling

入口文件：

- `vllm\v1\worker\gpu_model_runner.py`
- `vllm\v1\sample\rejection_sampler.py`

target verify 的关键准备在 `_prepare_inputs()` 与 `_calc_spec_decode_metadata()`：

- 如果本轮有 `scheduled_spec_decode_tokens`，说明 target 要验证上一轮 draft tokens。
- `_calc_spec_decode_metadata()` 计算：
  - `logits_indices`
  - `target_logits_indices`
  - `bonus_logits_indices`
  - `draft_token_ids`
  - `num_draft_tokens`
  - cumulative offsets

采样时：

```python
self.rejection_sampler(spec_decode_metadata, None, logits, sampling_metadata)
```

rejection sampler 根据 target logits 与 draft tokens 决定接受多少 token：

- greedy 场景：逐位置比较 `draft_token_ids == target_argmax`。
- 接受最长连续匹配前缀。
- 如果所有 draft token 都被接受，则追加 bonus token。
- 非 greedy 场景使用概率式 rejection sampling。

### 3.7 上游 runner 中 target hidden 传给 drafter

入口文件：

- `vllm\v1\worker\gpu_model_runner.py`

在 `propose_draft_token_ids()` 分支中，对于 `use_eagle()/use_dflash()/uses_draft_model()`：

1. 先从 target sampler output 里得到 `next_token_ids`。
2. 如果启用了 aux hidden states：

```python
target_hidden_states = torch.cat(
    [h[:num_scheduled_tokens] for h in aux_hidden_states],
    dim=-1,
)
```

3. 调用：

```python
draft_token_ids = self.drafter.propose(
    target_token_ids=target_token_ids,
    target_positions=target_positions,
    target_hidden_states=target_hidden_states,
    next_token_ids=next_token_ids,
    ...
)
```

DFlashProposer 接收到这些 hidden states 后，才进行 context KV 预计算与并行 draft。

## 4. vllm-ascend 覆盖/适配链路

vllm-ascend 的主线与上游一致，但把 GPU runner/proposer/attention 替换成 Ascend 实现。

### 4.1 drafter 创建

入口文件：

- `vllm_ascend\worker\model_runner_v1.py`
- `vllm_ascend\spec_decode\__init__.py`

`NPUModelRunner._set_up_drafter()` 中：

```python
self.drafter = self._get_drafter()
```

`_get_drafter()` 调用：

```python
get_spec_decode_method(self.speculative_config.method, ...)
```

最终：

```python
elif method == "dflash":
    return AscendDflashProposer(vllm_config, device, runner)
```

### 4.2 AscendDflashProposer

入口文件：

- `vllm_ascend\spec_decode\dflash_proposer.py`

它继承 `AscendEagleProposer`，但重写 DFlash 专属逻辑：

- `set_inputs_first_pass()`
- `dummy_run()`
- `build_model_inputs_first_pass()`

与上游相比，关键差异：

- dtype 使用 `int32` buffer，更贴合 NPU kernel。
- `copy_and_expand_dflash_inputs_kernel_single_grid` 是 Ascend 侧 Triton kernel。
- attention metadata 使用 `AscendCommonAttentionMetadata` 和 `AscendAttentionState.ChunkedPrefill`。
- 支持 `FULL_DECODE_ONLY` graph 下的 draft metadata 构建。
- `_dflash_hidden_states` 会先 copy 到预分配 buffer，保持 graph/runner buffer 稳定。

### 4.3 Ascend DFlash KV 预计算 patch

入口文件：

- `vllm_ascend\patch\worker\__init__.py`
- `vllm_ascend\patch\worker\patch_qwen3_dflash.py`

`patch_qwen3_dflash.py` monkey patch：

```python
DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv
```

Ascend 版本做：

1. `hidden_norm(context_states)`
2. fused K/V projection
3. reshape 成 `[2, L, num_ctx, nkv, hd]`
4. 每层 K-norm
5. 对所有层 K 做 RoPE
6. 调用 attention backend 的 `do_kv_cache_update()` 写入 NPU KV cache

这段是 DFlash 性能关键路径之一，因为每个 decode step 的 context hidden 都要先变成 draft KV。

### 4.4 Ascend runner 中 target hidden 传递

入口文件：

- `vllm_ascend\worker\model_runner_v1.py`

与上游一致，target forward 后：

- 如果 `use_aux_hidden_state_outputs=True`，runner 从 target output 中取 `aux_hidden_states`。
- DFlash/EAGLE3 分支把 aux hidden 拼接：

```python
target_hidden_states = torch.cat([h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1)
```

然后调用：

```python
self.drafter._propose(...)
```

`AscendSpecDecodeBaseProposer._propose()` 会对 `eagle3/dflash` 调：

```python
target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
```

也就是用 DFlash draft head 的 `fc` 把多层 target hidden 压成 draft hidden。

### 4.5 Ascend target verify 与 rejection sampler

入口文件：

- `vllm_ascend\worker\model_runner_v1.py`
- `vllm_ascend\sample\rejection_sampler.py`
- `vllm_ascend\ops\triton\reject_sample.py`

逻辑与上游一致，但实现上用 Ascend/NPU 版本：

1. `_calc_spec_decode_metadata()` 计算 target verify 所需索引。
2. `_sample()` 中如果存在 `spec_decode_metadata`，调用：

```python
self.rejection_sampler(spec_decode_metadata, None, logits, sampling_metadata)
```

3. Ascend rejection sampler 使用 NPU/Triton kernel 加速 greedy/probabilistic rejection。
4. 输出 `valid_sampled_token_ids` 后，下一轮再交给 DFlash proposer 生成新的 draft。

## 5. Qwen3-8B 与 Qwen3.6-35B-A3B 的差异点

### Qwen3-8B

链路最干净：

- target `model_type` 是 Qwen3 dense。
- draft head 可直接使用已有 `z-lab/Qwen3-8B-DFlash-b16`。
- draft config 可以自然映射到 `DFlashQwen3ForCausalLM`。
- target aux hidden layer ids 与 draft `fc` 输入维度容易对齐。

关注点：

- `num_speculative_tokens` 与 `cudagraph_capture_sizes` 匹配。
- `mask_token_id` 必须存在于 draft config 的 `dflash_config`。
- `target_layer_ids` 与训练 DFlash head 时使用的层完全一致。

### Qwen3.6-35B-A3B

这条链路更容易出错：

- target 可能是 `qwen3_5_moe`/VLM/M-RoPE 形态。
- vLLM `SpeculativeConfig` 对 `dflash/eagle3/extract_hidden_states` 的 target model type 支持检查依赖 `hf_text_config.model_type` 包含支持字符串。Qwen3.x 一般会命中 `"qwen"`，但仍需以实际 config 为准。
- DFlash draft model 本身仍要能映射到 `DFlashQwen3ForCausalLM` 或 `DFlashDraftModel`。
- 如果训练态来自 AngelSlim 的 `QwenDFlashDraftModel`，直接拿训练 config serving 可能不够，需要转换成 vLLM 认识的 DFlash draft config/权重命名。
- VLM/M-RoPE 场景要核对：
  - `rope_parameters`
  - `mrope_interleaved`
  - `partial_rotary_factor`
  - `target_layer_ids`
  - `target_hidden_size`
  - image/video special token 是否不参与 loss 或不参与 draft 预测

## 6. 端到端调用栈概览

### 初始化阶段

```text
vllm serve / LLM(...)
  -> EngineArgs.create_speculative_config()
  -> SpeculativeConfig(...)
      -> method = dflash
      -> draft_model_config = ModelConfig(draft head)
      -> EAGLEConfig(method="dflash")
      -> parallel_drafting = True
      -> architecture = DFlashQwen3ForCausalLM / DFlashDraftModel
  -> ModelRunner._set_up_drafter()
      upstream: DFlashProposer
      ascend:   AscendDflashProposer
  -> load draft model
      -> DFlashQwen3ForCausalLM
      -> DFlashQwen3Model
      -> _build_fused_kv_buffers()
```

### 每个 decode iteration

```text
target forward
  -> hidden_states, aux_hidden_states, logits
  -> sample target next_token_ids

propose draft
  -> gather target_token_ids / target_positions / target_hidden_states
  -> concat aux hidden states if needed
  -> DFlash combine_hidden_states(fc)
  -> DFlash set_inputs_first_pass()
      -> context = target hidden states
      -> query = [bonus_token, mask_token, ..., mask_token]
      -> build positions and slot_mapping
      -> causal = False
  -> precompute_and_store_context_kv()
      -> context hidden -> draft KV cache
  -> draft model forward(query only)
  -> draft logits -> draft_token_ids [B, K]

next target verify
  -> scheduler schedules K draft tokens
  -> target forward over 1 + K tokens
  -> _calc_spec_decode_metadata()
  -> rejection_sampler()
      -> accept longest prefix
      -> append bonus token if all accepted
  -> output valid_sampled_token_ids
```

## 7. 关键文件速查

上游 vLLM：

- `D:\workspace\speculative\vllm-v0.20.2\vllm\config\speculative.py`
- `D:\workspace\speculative\vllm-v0.20.2\vllm\transformers_utils\configs\eagle.py`
- `D:\workspace\speculative\vllm-v0.20.2\vllm\transformers_utils\configs\speculators\algos.py`
- `D:\workspace\speculative\vllm-v0.20.2\vllm\v1\worker\gpu_model_runner.py`
- `D:\workspace\speculative\vllm-v0.20.2\vllm\v1\spec_decode\dflash.py`
- `D:\workspace\speculative\vllm-v0.20.2\vllm\v1\spec_decode\llm_base_proposer.py`
- `D:\workspace\speculative\vllm-v0.20.2\vllm\model_executor\models\qwen3_dflash.py`
- `D:\workspace\speculative\vllm-v0.20.2\vllm\v1\sample\rejection_sampler.py`

vllm-ascend：

- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\worker\model_runner_v1.py`
- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\spec_decode\__init__.py`
- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\spec_decode\dflash_proposer.py`
- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\spec_decode\llm_base_proposer.py`
- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\ops\triton\spec_decode\utils.py`
- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\patch\worker\patch_qwen3_dflash.py`
- `D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\sample\rejection_sampler.py`

## 8. 实操检查清单

部署 DFlash head 前建议检查：

1. draft checkpoint config 能映射到 `DFlashQwen3ForCausalLM` 或 `DFlashDraftModel`。
2. `dflash_config.mask_token_id` 存在，且是 tokenizer 中的有效 token。
3. `dflash_config.target_layer_ids` 与训练时一致。
4. 如果 target hidden size 与 draft hidden size 不同，draft config/权重中有正确的 `target_hidden_size` 和 `fc` 权重。
5. `num_speculative_tokens=K` 与 graph capture size 对齐为 `batch * (K + 1)`。
6. Qwen3.6/Qwen3.5 VLM/MoE 场景确认 M-RoPE、视觉 token、special token、aux hidden layer 输出都与训练阶段一致。
7. 如果从 AngelSlim 训练产物导出给 vLLM serving，确认权重命名、architecture、`lm_head`/embedding 共享或加载策略已经转换到 vLLM DFlash 格式。
