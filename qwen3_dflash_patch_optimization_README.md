# Qwen3 DFlash Ascend Patch Optimization Notes

本文整理 `vllm-ascend v0.20.2rc1` 中 `patch_qwen3_dflash.py` 的可优化点，重点对象是：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\patch\worker\patch_qwen3_dflash.py
```

该 patch 替换了上游 vLLM 的：

```python
vllm.model_executor.models.qwen3_dflash.DFlashQwen3Model.precompute_and_store_context_kv
```

它的职责是在 DFlash first pass 前，将 target/context hidden states 预计算成 draft model 各 attention layer 的 K/V，并写入每层 KV cache。

## 1. 当前调用栈

patch import 阶段：

```text
vllm_ascend.patch.worker.__init__
  -> import vllm_ascend.patch.worker.patch_qwen3_dflash
    -> DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv
```

DFlash 推理阶段：

```text
AscendDflashProposer
  -> self.model.precompute_and_store_context_kv(...)
    -> patch_qwen3_dflash.py::precompute_and_store_context_kv
      -> self._build_fused_kv_buffers()             # first call only
      -> self.hidden_norm(context_states)
      -> F.linear(..., self._fused_kv_weight, ...)
      -> view/permute/contiguous -> all_k/all_v
      -> per-layer k_norm
      -> self.layers[0].self_attn.rotary_emb(...)
        -> AscendRotaryEmbedding.forward_oot(...)
          -> torch.ops.vllm.npu_rotary_embedding(...)
            -> rope_forward_oot(...)
              -> torch_npu._npu_rotary_embedding(...)
      -> per-layer attn.impl.do_kv_cache_update(...)
        -> DeviceOperator.reshape_and_cache(...)
```

## 2. 与上游实现的关键差异

上游 vLLM `qwen3_dflash.py` 使用 CUDA custom ops：

```python
ops.rms_norm(...)
ops.rotary_embedding(...)
```

其中上游 CUDA `ops.rotary_embedding()` 是原地修改 query/key，所以可以不接返回值。

Ascend patch 不能直接调用这些 CUDA ops，因此改为：

```python
self.hidden_norm(...)
self.layers[i].self_attn.k_norm(...)
self.layers[0].self_attn.rotary_emb(...)
```

但 Ascend RoPE wrapper 的语义不是 CUDA 原地 op：

```python
AscendRotaryEmbedding.forward_oot(...)
  -> return torch.ops.vllm.npu_rotary_embedding(...)
```

并且 `npu_rotary_embedding` 注册时 `mutates_args=[]`，因此从接口语义上应当接收返回值。

## 3. 优化点总览

| 优先级 | 优化点 | 类型 | 建议 |
| --- | --- | --- | --- |
| P0 | RoPE 返回值未接收，且 `clone()` 成本高 | 正确性 + 性能 | 先修 |
| P1 | `positions_repeated` / RoPE scratch 每轮分配 | 性能 | 可缓存，但注意 graph shape |
| P1 | per-layer `k_norm` Python/module 循环 | 性能 | profiler 后优化 |
| P2 | hidden RMSNorm module 调用 | 性能 | 尝试底层 Ascend RMSNorm op |
| P2 | per-layer KV cache update 循环 | 性能 | 高风险，需证明瓶颈 |
| P2 | RoPE query-only custom op | 性能 | 最优但需要新增 op |
| P3 | 保留上游 warning / 加 shape assert | 可维护性 | 低风险 |

## 4. P0：修正 RoPE 返回值并去掉 clone

当前代码：

```python
all_k_flat = all_k_normed.view(L * num_ctx, kv)
positions_repeated = context_positions.repeat(L)
tmpv = all_k_flat.clone()
self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
```

问题：

- Ascend `rotary_emb` 返回 `(query, key)`，当前返回值被丢弃。
- `tmpv = all_k_flat.clone()` 是完整 device copy，成本高。
- 这里实际只需要对 K 做 RoPE，`tmpv` 只是为了满足 Ascend RoPE 接口的 key 参数。

推荐先改成：

```python
all_k_flat = all_k_normed.view(L * num_ctx, kv)
positions_repeated = context_positions.repeat(L)

# Ascend rotary embedding requires both query and key tensors. This path only
# needs rotated K, so use an uninitialized scratch key instead of cloning K.
key_scratch = torch.empty_like(all_k_flat)
all_k_flat, _ = self.layers[0].self_attn.rotary_emb(
    positions_repeated,
    all_k_flat,
    key_scratch,
)
```

不要直接写成：

```python
all_k_flat, _ = self.layers[0].self_attn.rotary_emb(
    positions_repeated,
    all_k_flat,
    all_k_flat,
)
```

原因是 query/key alias 同一块 storage 时，底层 `_npu_rotary_embedding` 是否支持输入输出别名没有明确保证。为了避免边读边写互相覆盖，先用 `empty_like` scratch 更稳。

预期收益：

- 修正 RoPE 结果可能未写回的问题。
- 去掉一次 `clone()` 的大拷贝。

风险：

- `key_scratch` 未初始化，但只作为 key 输入参与 RoPE。返回的 key 被丢弃，因此数值无关。
- 如果底层 op 读取 key 造成额外计算仍然存在，但比 clone 更便宜。

## 5. P1：缓存 positions 和 RoPE scratch

上面 P0 的版本仍会每次分配：

```python
positions_repeated = context_positions.repeat(L)
key_scratch = torch.empty_like(all_k_flat)
```

可以进一步缓存：

```python
positions_size = L * num_ctx
if (
    not hasattr(self, "_dflash_positions_repeated")
    or self._dflash_positions_repeated.numel() < positions_size
    or self._dflash_positions_repeated.device != context_positions.device
):
    self._dflash_positions_repeated = torch.empty(
        positions_size,
        dtype=context_positions.dtype,
        device=context_positions.device,
    )
positions_repeated = self._dflash_positions_repeated[:positions_size]
positions_repeated.copy_(context_positions.repeat(L))

if (
    not hasattr(self, "_dflash_rope_key_scratch")
    or self._dflash_rope_key_scratch.shape != all_k_flat.shape
    or self._dflash_rope_key_scratch.dtype != all_k_flat.dtype
    or self._dflash_rope_key_scratch.device != all_k_flat.device
):
    self._dflash_rope_key_scratch = torch.empty_like(all_k_flat)

all_k_flat, _ = self.layers[0].self_attn.rotary_emb(
    positions_repeated,
    all_k_flat,
    self._dflash_rope_key_scratch,
)
```

注意：

- 这个版本仍然用了 `context_positions.repeat(L)` 再 `copy_`，只是复用了目标 buffer；要完全避免 `repeat`，需要自己构造 repeat kernel 或用更复杂的 view/index 方案。
- 如果运行在 ACL graph / full graph capture 中，动态创建 `self._dflash_*` buffer 可能影响 capture。建议先在 warmup/profile 阶段完成最大 shape 的 buffer 初始化。

更保守的落地顺序：

1. 先做 P0 非缓存版。
2. profiler 证明 allocation 开销明显后，再做缓存版。

## 6. P1：优化 per-layer K RMSNorm

当前代码：

```python
all_k_normed = torch.empty_like(all_k)
for i in range(L):
    k_norm_layer = self.layers[i].self_attn.k_norm
    all_k_normed[i] = k_norm_layer(all_k[i])
```

问题：

- 每层一次 Python loop。
- 每层一次 module forward。
- 上游 CUDA 版本使用底层 `ops.rms_norm(...)`，没有走 module forward。

低风险小优化：

```python
if not hasattr(self, "_dflash_k_norm_layers"):
    self._dflash_k_norm_layers = [
        layer.self_attn.k_norm for layer in self.layers
    ]

all_k_normed = torch.empty_like(all_k)
for i, k_norm_layer in enumerate(self._dflash_k_norm_layers):
    all_k_normed[i] = k_norm_layer(all_k[i])
```

中等风险优化：

- 如果 Ascend 提供可直接调用的 RMSNorm op，改成类似上游：

```python
ascend_rms_norm(out, inp, weight, eps)
```

- 使用 `_k_norm_weights[i]`，绕过 module forward。

高风险优化：

- 将 `[L, num_ctx, nkv, hd]` reshape 成 `[L * num_ctx * nkv, hd]`，做 grouped/batched RMSNorm。
- 难点是每层 `k_norm.weight` 不同，需要 per-layer weight 选择或扩展，可能引入额外 gather/broadcast，未必比逐层快。

建议：

- 先通过 profiler 确认 `k_norm` loop 是否是热点。
- 如果热点明显，优先尝试底层 Ascend RMSNorm op，而不是手写广播版。

## 7. P2：hidden RMSNorm module 调用

当前代码：

```python
normed_context_states = self.hidden_norm(context_states)
```

上游 CUDA 版本是：

```python
normed_context_states = torch.empty_like(context_states)
ops.rms_norm(
    normed_context_states,
    context_states,
    self._hidden_norm_weight,
    self._rms_norm_eps,
)
```

可优化方向：

- 如果 Ascend 可直接调用底层 RMSNorm op，改成预分配输出 + op 写入。
- 避免 module forward 内部可能发生的额外 reshape、contiguous 或 tuple 分支判断。

建议优先级低于 RoPE 和 K RMSNorm，因为这里只调用一次，而 K RMSNorm 是每层一次。

## 8. P2：per-layer KV cache update 循环

当前代码：

```python
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

`do_kv_cache_update()` 内部调用：

```text
DeviceOperator.reshape_and_cache(...)
```

问题：

- 每层一次 cache update。
- 多层时会产生多次 kernel launch / 调度。

潜在优化：

- 新增 multi-layer reshape-and-cache kernel，一次或分组写入多层 KV cache。
- 前提是 KV cache layout、layer cache list、slot_mapping 语义都能统一。

风险：

- 改动大，容易破坏 cache layout。
- 不同 attention backend 可能有不同 KV cache 格式。

建议：

- 先用 profiler 证明这段是热点。
- 如果热点明显，再单独设计 multi-layer cache update。
- 不建议作为第一轮 patch。

## 9. P2：新增 query-only RoPE custom op

当前 Ascend RoPE 接口要求 query/key 都传入：

```python
torch.ops.vllm.npu_rotary_embedding(positions, query, key, ...)
```

DFlash context KV precompute 实际只需要旋转 K。现在用 `key_scratch` 是为了适配接口。

更理想的实现是新增 query-only 或 single-tensor RoPE op：

```python
all_k_flat = torch.ops.vllm.npu_rotary_embedding_single(
    positions_repeated,
    all_k_flat,
    cos_sin_cache,
    head_size,
    rotary_dim,
    is_neox_style,
)
```

收益：

- 不需要 `key_scratch`。
- 避免对无用 key 做 RoPE 计算。

风险：

- 需要新增 custom op 注册、fake impl、NPU kernel 包装。
- 需要确认与已有 RoPE dtype/cache/shape 逻辑一致。

建议：

- 如果 P0 后 profiler 显示 RoPE 仍是瓶颈，再做这个优化。

## 10. P3：保留上游 warning 和增加校验

上游在 `_num_attn_layers` 不存在时会 warning：

```python
logger.warning_once(
    "DFlash buffer initialization was skipped. ..."
)
self._build_fused_kv_buffers()
```

Ascend patch 当前直接 build，没有 warning。建议保留类似 warning，方便排查权重加载或 dummy run 时序问题。

还可以加轻量 assert：

```python
assert context_positions.shape[0] == num_ctx
assert all_k_flat.shape == (L * num_ctx, kv)
```

注意：

- assert 在性能路径可能有轻微开销，可以只在 debug/config 开关下启用。

## 11. 建议落地顺序

第一阶段，低风险修正：

1. 接收 RoPE 返回值。
2. `tmpv.clone()` 改为 `torch.empty_like()` scratch。
3. 保留上游 warning。

第二阶段，profile 后优化：

1. 缓存 RoPE scratch。
2. 缓存 `k_norm_layers`。
3. 尝试底层 Ascend RMSNorm op 替代 module forward。

第三阶段，高风险专项优化：

1. query-only RoPE custom op。
2. multi-layer KV cache update。
3. grouped/batched K RMSNorm。

## 12. 推荐第一阶段 patch 片段

建议优先把 RoPE 段改为：

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

如果后续确认 graph capture 下每轮分配 scratch 有明显影响，再改成缓存版。

## 13. 验证建议

正确性验证：

- 使用固定 prompt，对比 patch 前后 DFlash 输出 token 是否稳定。
- 对比 `context_slot_mapping is None` 的 profile/dummy run 是否正常返回。
- 开启 DFlash 后跑短输出，确认没有 NaN、shape error、KV cache 写入异常。

性能验证：

- 用 torch_npu profiler 或 vllm-ascend 现有 profile 工具，关注：
  - `npu_rotary_embedding`
  - RMSNorm kernel
  - `reshape_and_cache`
  - device memory allocation / copy
- 对比以下版本：
  - 原始 patch。
  - P0 修正版。
  - P0 + scratch 缓存版。

指标：

- 首轮 DFlash precompute latency。
- 单请求 / batch 请求 TPOT。
- NPU kernel launch 数。
- HBM copy / allocation 开销。

## 14. 结论

最值得先做的是 RoPE 段：

```python
tmpv = all_k_flat.clone()
self.layers[0].self_attn.rotary_emb(...)
```

应该改成接收返回值，并用 `empty_like` scratch 替代 clone。这个改动同时解决潜在正确性问题和明显的内存拷贝开销。

其余优化需要 profiler 排序：如果 `k_norm` 热，优先优化 RMSNorm；如果 cache update 热，再考虑 multi-layer cache update；如果 RoPE 仍热，再考虑新增 query-only RoPE custom op。
