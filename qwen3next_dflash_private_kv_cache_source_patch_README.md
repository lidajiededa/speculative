# Qwen3Next DFlash 私有 KV Cache 源码修改方案

本文档给出一个面向 `vllm-ascend v0.20.2rc1` 的源码补丁方案，用来验证下面这个性能假设：

开启 DFlash 后，DFlash draft head 的 attention 层被纳入全局 KV cache 规划；在 Qwen3.6-35B-A3B 这类 hybrid linear-attention 模型上，全局 KV cache 又会受到 GDN/Mamba state padding 影响，导致实际可用 KV block 明显下降。解决方向是：target model 的 KV cache 仍由 vLLM 全局 KV manager 管理，但 DFlash draft head 的 KV cache 改为 proposer 私有 scratch cache，不再占用全局 KV block 预算。

推荐先用环境变量保护：

```bash
export VLLM_ASCEND_DFLASH_PRIVATE_KV_CACHE=1
```

默认不打开，避免影响已有 MTP/EAGLE/普通 DFlash 路径。

## 总体代码改动

涉及 3 个文件：

```text
vllm_ascend/worker/model_runner_v1.py
vllm_ascend/spec_decode/dflash_proposer.py
vllm_ascend/ops/triton/spec_decode/utils.py
```

核心变化：

```text
原始路径:
  target attention KV + Qwen3Next linear_attn state + DFlash draft attention KV
  全部进入全局 KV cache spec -> KVCacheManager 统一按 group/page_size 计算 num_blocks

修改后:
  target attention KV + Qwen3Next linear_attn state
  继续进入全局 KV cache spec

  DFlash draft attention KV
  不进入全局 KV cache spec，改由 AscendDflashProposer 自己分配固定 scratch KV cache
```

注意：这个补丁是实验型优化，建议先只覆盖 DFlash + 标准 attention draft head 路径；如果开启 sparse/MLA/compress，应先保持关闭或额外补齐对应 KV layout。

## 1. 修改 model_runner_v1.py

文件：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\worker\model_runner_v1.py
```

### 1.1 增加 import

在文件顶部已有 import 区域增加：

```python
import os
```

如果文件里已经有 `os`，不要重复加。

### 1.2 增加私有 KV 开关 helper

在 `class GPUModelRunner` 内，建议放在 `get_kv_cache_spec()` 前面：

```python
    def _use_dflash_private_kv_cache(self) -> bool:
        """Whether DFlash draft attention KV cache should be private.

        DFlash draft attention layers are auxiliary layers. Putting them into
        the global KV cache planner can reduce target-model KV blocks heavily
        for Qwen3Next hybrid attention models, because attention pages may be
        padded/aligned with Mamba/GDN state pages.
        """
        if not self.speculative_config:
            return False
        if self.speculative_config.method != "dflash":
            return False
        return os.getenv("VLLM_ASCEND_DFLASH_PRIVATE_KV_CACHE", "0") == "1"

    def _is_private_dflash_attn_layer(self, layer_name: str) -> bool:
        if not self._use_dflash_private_kv_cache():
            return False
        drafter = getattr(self, "drafter", None)
        if drafter is None:
            return False
        return layer_name in getattr(drafter, "_draft_attn_layer_names", set())
```

### 1.3 从全局 KV spec 跳过 DFlash draft attention 层

在当前 `get_kv_cache_spec()` 里有如下循环：

```python
        for layer_name, attn_module in attn_layers.items():
            if (isinstance(attn_module, Attention)
                    and (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None):
                ...
```

在该循环开头插入：

```python
        for layer_name, attn_module in attn_layers.items():
            if self._is_private_dflash_attn_layer(layer_name):
                logger.info(
                    "Skip DFlash draft attention layer %s from global KV cache "
                    "spec because VLLM_ASCEND_DFLASH_PRIVATE_KV_CACHE=1.",
                    layer_name,
                )
                continue

            if (isinstance(attn_module, Attention)
                    and (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None):
                ...
```

为什么要在这里跳过：

`get_kv_cache_spec()` 是全局 KVCacheManager 计算 `kv_cache_groups`、`page_size`、`num_blocks` 的入口。DFlash draft attention 层如果留在这里，就会参与全局 KV block 数计算；跳过后，target model 的 KV block 数不会再被 DFlash draft KV 拉低。

### 1.4 初始化全局 KV 后，给 DFlash proposer 分配私有 KV

当前 `initialize_kv_cache()` 中已有：

```python
        if self.speculative_config and (
            self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(self.drafter, AscendEagleProposer | AscendDflashProposer | AscendDraftModelProposer)
            block_size = (self.kernel_block_sizes[0] if isinstance(
            self.kernel_block_sizes, list) else self.kernel_block_sizes)
            self.drafter.initialize_attn_backend(kv_cache_config, block_size)
```

替换为：

```python
        if self.speculative_config and (
            self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(self.drafter, AscendEagleProposer | AscendDflashProposer | AscendDraftModelProposer)
            block_size = (
                self.kernel_block_sizes[0]
                if isinstance(self.kernel_block_sizes, list)
                else self.kernel_block_sizes
            )
            self.drafter.initialize_attn_backend(kv_cache_config, block_size)

            if (
                isinstance(self.drafter, AscendDflashProposer)
                and self._use_dflash_private_kv_cache()
            ):
                self.drafter.initialize_private_kv_cache(block_size)
```

为什么放在这里：

这里全局 KV cache 已经完成初始化，DFlash proposer 也已经加载出 draft model 和 draft attention layer names。此时补充私有 KV cache 最稳，不影响 target model 的全局 KV cache 分配。

## 2. 修改 dflash_proposer.py

文件：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\spec_decode\dflash_proposer.py
```

### 2.1 增加 import

当前文件开头类似：

```python
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
...
```

修改为：

```python
import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.abstract import AttentionLayerBase
from vllm.utils import cdiv
from vllm.v1.worker.utils import AttentionGroup, get_layers_from_vllm_config
from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs
```

如果实际版本里 `Attention` 或 `AttentionLayerBase` 的 import 路径已经不同，以本仓库现有 import 为准。`llm_base_proposer.py` 已经在使用 `get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)`，可以直接对齐它的 import。

### 2.2 在 `__init__()` 里增加私有 KV buffer 字段

当前 `AscendDflashProposer.__init__()` 末尾类似：

```python
        self.parallel_drafting_hidden_state_tensor = None
```

后面追加：

```python
        self.use_private_kv_cache = (
            os.getenv("VLLM_ASCEND_DFLASH_PRIVATE_KV_CACHE", "0") == "1"
        )
        self.private_kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.private_num_blocks = 0
        self.private_block_size = 0
        self.private_max_blocks_per_req = 0
        self._private_block_table = None
        self._private_context_slot_mapping_buffer = None
        self._private_slot_mapping_buffer = None
```

### 2.3 重写 DFlash 的 `initialize_attn_backend()`

因为开启私有 KV 后，DFlash draft layer 已经从全局 `kv_cache_config` 移除，不能再走上游 `validate_same_kv_cache_group(kv_cache_config)`。在 `AscendDflashProposer` 类里新增方法：

```python
    def initialize_attn_backend(self, kv_cache_config, kernel_block_size=None) -> None:
        """Initialize draft attention metadata builders.

        Normal DFlash follows the upstream/global KV-cache path. With private
        KV cache enabled, DFlash draft attention layers are intentionally absent
        from kv_cache_config, so we build AttentionGroup directly from the draft
        layer's own KVCacheSpec.
        """
        if not self.use_private_kv_cache:
            return super().initialize_attn_backend(kv_cache_config, kernel_block_size)

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        attention_groups: dict[str, AttentionGroup] = {}
        for layer_name in self._draft_attn_layer_names:
            attn_layer = all_attn_layers[layer_name]
            kv_cache_spec = attn_layer.get_kv_cache_spec(self.vllm_config)
            if kv_cache_spec is None:
                continue

            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
            if not isinstance(kv_cache_spec, AttentionSpec):
                raise TypeError(
                    "DFlash private KV cache currently supports AttentionSpec "
                    f"only, but {layer_name} uses {type(kv_cache_spec).__name__}."
                )

            attn_backend = attn_layer.get_attn_backend()
            backend_key = attn_backend.full_cls_name()
            if backend_key not in attention_groups:
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=kv_cache_spec,
                    # Private DFlash no longer belongs to a global KV group.
                    # Keep 0 so metadata builders that require an integer gid
                    # still have a stable value.
                    kv_cache_group_id=0,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[backend_key] = attn_group
            else:
                attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        if not self.draft_attn_groups:
            raise RuntimeError("No DFlash draft attention group is initialized.")

        self.block_size = self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        self.kv_cache_gid = 0
```

### 2.4 新增私有 KV cache 分配函数

在 `AscendDflashProposer` 类里继续新增：

```python
    def initialize_private_kv_cache(self, block_size: int) -> None:
        """Allocate private scratch KV cache for DFlash draft attention layers.

        This cache stores only DFlash context K/V and query K/V generated by the
        draft head. It is not managed by global KVCacheManager, so it does not
        reduce target-model available KV blocks.

        The first implementation intentionally supports regular AttentionSpec
        with non-sparse K/V split. If sparse/MLA/compressed draft attention is
        required, add a dedicated layout branch matching model_runner_v1.py.
        """
        if not self.use_private_kv_cache:
            return
        if self.runner.use_sparse or self.runner.use_compress:
            raise RuntimeError(
                "DFlash private KV cache patch does not cover sparse/compress "
                "layouts yet. Disable sparse/compress for first validation."
            )

        self.private_block_size = block_size

        # Private cache only needs to cover the tokens processed by DFlash in a
        # proposer step: context hidden states plus bonus/mask query tokens.
        max_private_tokens = self.max_num_tokens + self.max_query_tokens
        self.private_num_blocks = cdiv(max_private_tokens, block_size)

        # Each request owns a fixed private block range:
        #   req_idx -> [req_idx * private_max_blocks_per_req, ...)
        # This avoids dynamic allocation and keeps graph shapes stable.
        max_tokens_per_req = self.runner.scheduler_config.max_model_len + self.num_speculative_tokens + 1
        self.private_max_blocks_per_req = cdiv(max_tokens_per_req, block_size)
        needed_blocks = self.max_batch_size * self.private_max_blocks_per_req
        self.private_num_blocks = max(self.private_num_blocks, needed_blocks)

        self._private_block_table = torch.empty(
            (self.max_batch_size, self.private_max_blocks_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        self._private_context_slot_mapping_buffer = torch.empty(
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self._private_slot_mapping_buffer = torch.empty(
            self.max_query_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        self.private_kv_caches.clear()
        for layer_name in self.attn_layer_names:
            attn_layer = all_attn_layers[layer_name]
            kv_cache_spec = attn_layer.get_kv_cache_spec(self.vllm_config)
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
            if not isinstance(kv_cache_spec, AttentionSpec):
                raise TypeError(
                    "DFlash private KV cache currently supports AttentionSpec "
                    f"only, but {layer_name} uses {type(kv_cache_spec).__name__}."
                )

            attn_backend = attn_layer.get_attn_backend()
            kv_cache_shape = attn_backend.get_kv_cache_shape(
                self.private_num_blocks,
                block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

            # AscendAttentionImpl.do_kv_cache_update() expects kv_cache[0/1] to
            # be K/V tensors. Match model_runner_v1.py regular AttentionSpec
            # branch: K shape is kv_cache_shape[1:], V shape is same unless
            # the spec has a dedicated head_size_v.
            k_shape = kv_cache_shape[1:]
            if hasattr(kv_cache_spec, "head_size_v"):
                v_shape = (*kv_cache_shape[1:-1], kv_cache_spec.head_size_v)
            else:
                v_shape = k_shape

            k_cache = torch.empty(k_shape, dtype=kv_cache_spec.dtype, device=self.device)
            v_cache = torch.empty(v_shape, dtype=kv_cache_spec.dtype, device=self.device)

            attn_layer.kv_cache = (k_cache, v_cache)
            # Reset cached pointers inside AscendAttentionImpl, otherwise it may
            # keep the old global cache reference from an earlier initialization.
            if hasattr(attn_layer.impl, "key_cache"):
                attn_layer.impl.key_cache = None
            if hasattr(attn_layer.impl, "value_cache"):
                attn_layer.impl.value_cache = None

            self.private_kv_caches[layer_name] = (k_cache, v_cache)
```

关键解释：

1. 私有 cache 的 block table 是固定映射，不走全局 allocator。
2. 每个 req 固定占用 `private_max_blocks_per_req` 个 block，避免运行时分配。
3. 私有 cache 只服务 draft head，不影响 target model KV block 数。
4. 首版不覆盖 sparse/compress/MLA，是为了避免 KV layout 不一致导致静默错误。

### 2.5 修改 `set_inputs_first_pass()` 使用私有 block table/slot mapping

当前 `set_inputs_first_pass()` 中 kernel 调用大致是：

```python
        copy_and_expand_dflash_inputs_kernel_single_grid[1,](
            ...
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            ...
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            ...
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
```

替换成：

```python
        if self.use_private_kv_cache:
            assert self._private_block_table is not None
            assert self._private_context_slot_mapping_buffer is not None
            assert self._private_slot_mapping_buffer is not None
            context_slot_mapping_buffer = self._private_context_slot_mapping_buffer
            query_slot_mapping_buffer = self._private_slot_mapping_buffer
            block_table = self._private_block_table
            block_table_stride = self._private_block_table.stride(0)
        else:
            context_slot_mapping_buffer = self._context_slot_mapping_buffer
            query_slot_mapping_buffer = self._slot_mapping_buffer
            block_table = cad.block_table_tensor
            block_table_stride = cad.block_table_tensor.stride(0)

        copy_and_expand_dflash_inputs_kernel_single_grid[1,](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=query_slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=block_table,
            block_table_stride=block_table_stride,
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
            USE_PRIVATE_BLOCK_TABLE=self.use_private_kv_cache,
            private_max_blocks_per_req=self.private_max_blocks_per_req,
        )

        query_slot_mapping = query_slot_mapping_buffer[:num_query_total]
```

并且在设置 metadata 处：

```python
        cad.slot_mapping = query_slot_mapping
```

保持不变。

如果开启私有 KV，还需要让 attention metadata 使用私有 block table：

```python
        if self.use_private_kv_cache:
            cad.block_table_tensor = self._private_block_table[:batch_size]
```

建议插入位置：

```python
        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        ...
        cad.slot_mapping = query_slot_mapping
        if self.use_private_kv_cache:
            cad.block_table_tensor = self._private_block_table[:batch_size]
```

### 2.6 修改 `build_model_inputs_first_pass()`

当前代码：

```python
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
```

替换为：

```python
        context_slot_mapping = (
            self._private_context_slot_mapping_buffer[:num_context]
            if self.use_private_kv_cache
            else self._context_slot_mapping_buffer[:num_context]
        )

        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            context_slot_mapping,
        )
```

这样 `precompute_and_store_context_kv()` 会把 context K/V 写入 DFlash 私有 KV cache，而不是写进 target 模型的全局 KV cache block。

### 2.7 修改 `dummy_run()` graph capture 路径

当前 `dummy_run()` 的 graph capture metadata 里有：

```python
                block_table_tensor=self.runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor()[
                    :num_reqs
                ],
```

替换为：

```python
                block_table_tensor=(
                    self._private_block_table[:num_reqs]
                    if self.use_private_kv_cache
                    else self.runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor()[:num_reqs]
                ),
```

同时 `slot_mapping` 替换为：

```python
                slot_mapping=(
                    self._private_slot_mapping_buffer[:num_query_total]
                    if self.use_private_kv_cache
                    else self._slot_mapping_buffer[:num_query_total]
                ),
```

注意：graph capture 只关心 shape 稳定。私有 block table 和 slot mapping 都是预分配 buffer，地址稳定，不会破坏图模式；但 private block table 的内容必须在真实运行前由 kernel 写好。

## 3. 修改 utils.py 的 Triton kernel

文件：

```text
D:\workspace\speculative\vllm-ascend-0.20.2rc1\vllm_ascend\ops\triton\spec_decode\utils.py
```

当前 kernel：

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(
    ...
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
):
```

把函数签名末尾改为：

```python
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
    USE_PRIVATE_BLOCK_TABLE: tl.constexpr = False,
    private_max_blocks_per_req: tl.constexpr = 0,
):
```

### 3.1 context slot mapping 逻辑替换

当前 context 部分：

```python
            block_num = pos // block_size
            block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_num).to(tl.int64)
            slot = block_id * block_size + (pos % block_size)
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)
```

替换为：

```python
            block_num = pos // block_size
            if USE_PRIVATE_BLOCK_TABLE:
                block_id = req_idx * private_max_blocks_per_req + block_num
                tl.store(block_table_ptr + req_idx * block_table_stride + block_num, block_id)
            else:
                block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_num).to(tl.int64)
            slot = block_id * block_size + (pos % block_size)
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)
```

### 3.2 query slot mapping 逻辑替换

当前 query 部分：

```python
            block_num_q = query_pos // block_size
            block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q).to(tl.int64)
            slot_q = block_id_q * block_size + (query_pos % block_size)
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)
```

替换为：

```python
            block_num_q = query_pos // block_size
            if USE_PRIVATE_BLOCK_TABLE:
                block_id_q = req_idx * private_max_blocks_per_req + block_num_q
                tl.store(block_table_ptr + req_idx * block_table_stride + block_num_q, block_id_q)
            else:
                block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q).to(tl.int64)
            slot_q = block_id_q * block_size + (query_pos % block_size)
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)
```

为什么这样改：

1. 非私有路径保持原逻辑，从全局 `cad.block_table_tensor` 查 block id。
2. 私有路径不再依赖全局 allocator，而是用固定公式生成 block id。
3. 同时写 `private_block_table`，让后续 attention metadata 的 block table 与 slot mapping 一致。

## 4. 完整 kernel 修改后片段

下面是修改后的完整 DFlash expand kernel，方便直接对照替换。

```python
@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
    USE_PRIVATE_BLOCK_TABLE: tl.constexpr = False,
    private_max_blocks_per_req: tl.constexpr = 0,
):
    for req_idx in range(0, batch_size):
        ctx_start = tl.load(query_start_loc_ptr + req_idx)
        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
        num_ctx = ctx_end - ctx_start

        for j in range(0, num_ctx):
            ctx_pos_idx = ctx_start + j
            pos = tl.load(target_positions_ptr + ctx_pos_idx)
            tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

            block_num = pos // block_size
            if USE_PRIVATE_BLOCK_TABLE:
                block_id = req_idx * private_max_blocks_per_req + block_num
                tl.store(block_table_ptr + req_idx * block_table_stride + block_num, block_id)
            else:
                block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_num).to(tl.int64)
            slot = block_id * block_size + (pos % block_size)
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

        if HAS_NUM_REJECTED:
            num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
            valid_ctx_end = ctx_end - num_rejected
        else:
            valid_ctx_end = ctx_end

        last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)

        for q_idx in range(0, num_query_per_req):
            query_pos = last_pos + 1 + q_idx
            query_out_idx = req_idx * num_query_per_req + q_idx

            tl.store(out_query_positions_ptr + query_out_idx, query_pos)

            block_num_q = query_pos // block_size
            if USE_PRIVATE_BLOCK_TABLE:
                block_id_q = req_idx * private_max_blocks_per_req + block_num_q
                tl.store(block_table_ptr + req_idx * block_table_stride + block_num_q, block_id_q)
            else:
                block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q).to(tl.int64)
            slot_q = block_id_q * block_size + (query_pos % block_size)
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

            if q_idx == 0:
                bonus_token = tl.load(next_token_ids_ptr + req_idx)
                tl.store(out_input_ids_ptr + query_out_idx, bonus_token)
            else:
                tl.store(out_input_ids_ptr + query_out_idx, parallel_drafting_token_id)

                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)
```

## 5. 不建议首先改 Mamba/GDN speculative state

不要优先改这里：

```text
vllm/model_executor/layers/mamba/abstract.py
```

也不要先把：

```python
num_speculative_blocks=(
    vllm_config.speculative_config.num_speculative_tokens
    if vllm_config.speculative_config
    else 0
)
```

强行改成 0。

原因：

Qwen3Next 的 target model 有 GatedDeltaNet/Mamba 类 linear attention state。spec decode 验证目标 token 时，需要 speculative token 对应的状态空间支持，尤其涉及 accepted token 更新/rollback。把 target linear attention 的 `num_speculative_blocks` 直接清零，可能提升 KV block 数，但有正确性风险。

优先级应该是：

```text
第一优先级：把 DFlash draft attention KV 从全局 KV manager 中剥离
第二优先级：确认 target GDN/Mamba speculative state 的真实必要大小
第三优先级：再考虑按 full_decode_only、固定接受长度等模式做更激进裁剪
```

## 6. 预期收益与验证点

### 6.1 预期收益

开启私有 KV 后，日志里 “Available KV cache memory” 可能变化不大，但真正关键的是最终 `num_blocks`。

需要观察：

```text
不开 DFlash:
  num_blocks = A

原始 DFlash:
  num_blocks = B，可能比 A 小很多

私有 KV DFlash:
  num_blocks = C，应该明显接近 A
```

如果 C 接近 A，则说明 DFlash draft KV 的全局规划确实吃掉了 target 可用 KV blocks。

### 6.2 必做正确性验证

1. 单请求短 prompt：对比开启/关闭私有 KV 的输出 token。
2. 单请求长 prompt：覆盖 block table 跨多个 block。
3. 高并发短 prompt：验证每个 req 的私有 block range 不互相覆盖。
4. 高并发长 prompt：验证 `private_max_blocks_per_req` 足够。
5. DFlash 接受率：确认接受率没有异常跌为 0。
6. 首 token 时延：确认 TTFT 不再因为全局 KV block 数下降而恶化。
7. TPOT：确认 decode 阶段没有额外明显回退。

### 6.3 建议加日志

在 `model_runner_v1.py:get_kv_cache_spec()` 跳过 draft layer 时打印：

```python
logger.info("DFlash private KV skip layer: %s", layer_name)
```

在 `AscendDflashProposer.initialize_private_kv_cache()` 结束处打印：

```python
logger.info(
    "Initialized DFlash private KV cache: layers=%d, blocks=%d, block_size=%d, "
    "max_blocks_per_req=%d",
    len(self.private_kv_caches),
    self.private_num_blocks,
    self.private_block_size,
    self.private_max_blocks_per_req,
)
```

在 KV config 生成后观察最终 block 数。可以在全局 KV manager 或 worker 初始化处临时打印：

```python
logger.info("Final global KV num_blocks=%s", kv_cache_config.num_blocks)
for group in kv_cache_config.kv_cache_groups:
    logger.info(
        "KV group layers=%d spec=%s page_size=%s",
        len(group.layer_names),
        type(group.kv_cache_spec).__name__,
        getattr(group.kv_cache_spec, "page_size_bytes", None),
    )
```

## 7. 回滚方式

最小回滚：

```bash
unset VLLM_ASCEND_DFLASH_PRIVATE_KV_CACHE
```

或者：

```bash
export VLLM_ASCEND_DFLASH_PRIVATE_KV_CACHE=0
```

源码可以保留，因为默认不生效。

## 8. 风险与后续补齐

### 8.1 Graph 模式风险

这个补丁理论上不破坏 graph shape，因为：

1. 私有 block table 是预分配固定 tensor。
2. 私有 context/query slot mapping 是预分配固定 tensor。
3. graph capture 中传入的是相同地址的 buffer slice。

但仍需要实测，因为 Ascend graph 对 metadata builder、block table 地址、slice shape 可能有隐性约束。

### 8.2 KV layout 风险

首版只建议覆盖普通 DFlash attention KV。

这些路径先不要混用：

```text
use_sparse=True
use_compress=True
MLAAttentionSpec draft head
SlidingWindowMLASpec draft head
```

如果后续要支持，需要把 `model_runner_v1.py:_allocate_kv_cache_tensors()` 和 `_reshape_kv_cache_tensors()` 中对应 layout 逻辑复制到 `initialize_private_kv_cache()`，保证私有 KV cache shape 与 attention backend 完全一致。

### 8.3 私有 cache 容量风险

当前估算：

```python
max_tokens_per_req = max_model_len + num_speculative_tokens + 1
private_max_blocks_per_req = cdiv(max_tokens_per_req, block_size)
private_num_blocks = max_batch_size * private_max_blocks_per_req
```

这个最稳，但私有 cache 可能偏大。后续可以改成：

```text
private_max_blocks_per_req = cdiv(max_num_batched_tokens + num_speculative_tokens + 1, block_size)
```

或增加环境变量：

```bash
export VLLM_ASCEND_DFLASH_PRIVATE_MAX_TOKENS_PER_REQ=8192
```

但这样必须保证长 prompt 不越界。

## 9. 为什么这个方案比调参更直接

调 `max_num_batched_tokens`、`gpu_memory_utilization`、`block_size` 可以缓解调度预算或显存预算问题，但如果 DFlash draft KV 已经被纳入全局 KV group，根因仍在：

```text
全局 KV planner 看到更多 layer / 更大的 page_size / hybrid mamba padding
=> final num_blocks 下降
=> 高并发下更容易等待或调度碎片化
=> DFlash 性能收益被吞掉
```

源码改动的目标是把 DFlash draft KV 从全局容量模型中拿出去，让全局 KV block 只服务 target model。这样更适合 Qwen3.6-35B-A3B 这类 hybrid linear-attention 模型。
