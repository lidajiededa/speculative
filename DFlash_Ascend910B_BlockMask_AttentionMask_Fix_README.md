# DFlash on Ascend 910B: BlockMask 与 eager/sdpa Attention Mask 冲突修复指南

本文档说明在昇腾 910B NPU 上训练 AngelSlim DFlash 时，遇到如下错误的原因和详细修复方法：

```text
speculative/train/models/draft/qwen_dflash.py", line 165, in forward
attn_output, attn_weights = attn_fn(...)

transformers/models/qwen3/modeling_qwen3.py", line 211, in eager_attention_forward
attn_weights = attn_weights + attention_mask

TypeError: unsupported operand type(s) for +: 'Tensor' and 'BlockMask'
```

## 1. 问题结论

该错误不是 hidden cache、数据集或 Qwen3.6 模型权重本身导致的，而是 **DFlash trainer 构造的 attention mask 类型与实际 attention backend 不匹配**：

```text
DFlash trainer 固定创建 torch.nn.attention.flex_attention.BlockMask
             ↓
QwenDFlashDraftModel 实际走 eager_attention_forward 或 sdpa
             ↓
eager/sdpa 期望 Tensor mask，不支持 BlockMask
             ↓
attn_weights + attention_mask 报 TypeError
```

在 910B 上训练时，推荐使用：

```bash
--attention_backend sdpa
```

因此不能继续把 `BlockMask` 传给 DFlash draft model。需要为 `sdpa/eager` 生成普通 dense additive Tensor mask。

## 2. 涉及文件

主要修改两个文件：

```text
angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
angelslim/compressor/speculative/train/models/draft/qwen_dflash.py
```

可选同步检查：

```text
tools/train_dflash_offline.py
tools/train_dflash_online.py
configs/qwen3_6_35b_a3b_dflash_npu.json
scripts/speculative/*.sh
```

## 3. 当前源码中的根因

### 3.1 trainer 固定创建 BlockMask

文件：

```text
angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
```

当前逻辑：

```python
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

def create_dflash_block_mask(...):
    ...
    return create_block_mask(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )
```

后续训练时：

```python
dflash_attn_mask = create_dflash_block_mask(...)

output_hidden = model(
    noise_embedding=noise_embedding,
    target_hidden=hidden_states,
    attention_mask=dflash_attn_mask,
    position_ids=full_position_ids,
)
```

这里 `dflash_attn_mask` 的类型是 `BlockMask`。

### 3.2 draft model 根据 `_attn_implementation` 选择 attention backend

文件：

```text
angelslim/compressor/speculative/train/models/draft/qwen_dflash.py
```

当前逻辑：

```python
attn_fn: Callable = eager_attention_forward
if self.config._attn_implementation != "eager":
    attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

attn_output, attn_weights = attn_fn(
    self,
    q,
    k,
    v,
    attention_mask,
    ...
)
```

当 `_attn_implementation == "eager"` 时，会进入 Transformers 的 `eager_attention_forward`。

该函数内部会执行：

```python
attn_weights = attn_weights + attention_mask
```

所以它要求 `attention_mask` 是 Tensor，而不是 `BlockMask`。

## 4. 修复原则

根据 attention backend 生成不同类型的 mask：

```text
flex_attention -> BlockMask
sdpa/eager     -> dense additive Tensor mask
```

在 910B 上：

```text
默认使用 sdpa
不要使用 flex_attention
不要使用 flash_attention_2
```

## 5. 修改 1：新增 dense additive mask 生成函数

文件：

```text
angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
```

在 `create_dflash_block_mask` 后新增：

```python
def create_dflash_dense_attention_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct dense additive attention mask for eager/sdpa DFlash training.

    Returns:
        Tensor with shape [B, 1, Q_LEN, KV_LEN].

    Values:
        0.0 for visible positions.
        torch.finfo(dtype).min for masked positions.

    KV layout:
        [context tokens S | draft block tokens N * block_size]

    Q layout:
        [draft block tokens N * block_size]
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_idx = torch.arange(Q_LEN, device=device)
    kv_idx = torch.arange(KV_LEN, device=device)

    q_block_id = q_idx // block_size                          # [Q]
    anchors_for_q = anchor_positions[:, q_block_id]           # [B, Q]
    valid_for_q = block_keep_mask[:, q_block_id]              # [B, Q]

    q_block_id_3d = q_block_id.view(1, Q_LEN, 1)               # [1, Q, 1]
    kv_idx_3d = kv_idx.view(1, 1, KV_LEN)                      # [1, 1, K]

    is_context = kv_idx_3d < S
    mask_context = is_context & (kv_idx_3d < anchors_for_q.unsqueeze(-1))

    is_draft = kv_idx_3d >= S
    kv_block_id = (kv_idx_3d - S) // block_size
    mask_draft = is_draft & (kv_block_id == q_block_id_3d)

    visible = (mask_context | mask_draft) & valid_for_q.unsqueeze(-1)
    visible = visible.unsqueeze(1)                             # [B, 1, Q, K]

    mask = torch.full(
        (B, 1, Q_LEN, KV_LEN),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask = mask.masked_fill(visible, 0.0)
    return mask
```

说明：

- `visible=True` 的位置填 `0.0`。
- 被 mask 的位置填 `torch.finfo(dtype).min`。
- 返回 shape 是 `[B, 1, Q_LEN, KV_LEN]`，可以广播到 `[B, num_heads, Q_LEN, KV_LEN]`。
- 该 mask 可用于 `eager_attention_forward` 和 `sdpa`。

## 6. 修改 2：按 backend 创建 mask

文件：

```text
angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
```

找到 `_compute_dflash_loss_and_accuracy()` 里的：

```python
dflash_attn_mask = create_dflash_block_mask(
    anchor_positions=anchor_positions,
    block_keep_mask=block_keep_mask,
    S=seq_len,
    block_size=self.block_size,
    device=device,
)
```

替换为：

```python
model_dtype = next(model.parameters()).dtype

if self.attention_backend == "flex_attention":
    dflash_attn_mask = create_dflash_block_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        S=seq_len,
        block_size=self.block_size,
        device=device,
    )
else:
    dflash_attn_mask = create_dflash_dense_attention_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        S=seq_len,
        block_size=self.block_size,
        device=device,
        dtype=model_dtype,
    )
```

然后保留后面的：

```python
noise_embedding = noise_embedding.to(model_dtype)
hidden_states = hidden_states.to(model_dtype)
```

注意不要重复定义 `model_dtype`。如果原代码在后面已经有：

```python
model_dtype = next(model.parameters()).dtype
```

需要移动到创建 mask 之前。

## 7. 修改 3：在 910B 上强制使用 sdpa

### 7.1 命令行参数

离线训练时必须显式传入：

```bash
--attention_backend sdpa
```

示例：

```bash
torchrun --standalone --nproc_per_node=8 tools/train_dflash_offline.py \
  --target_model_name_or_path "$TARGET_MODEL_PATH" \
  --draft_model_config_path "$DRAFT_CONFIG_PATH" \
  --train_hidden_path "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --attention_backend sdpa \
  --per_device_train_batch_size 1 \
  --num_anchors 64 \
  --block_size 16 \
  --bf16 \
  --fsdp "shard_grad_op auto_wrap" \
  --fsdp_config configs/fsdp_config.json \
  --dataloader_drop_last \
  --report_to none
```

### 7.2 config 默认值

在 NPU 训练 config 中设置：

```json
{
  "attention_backend": "sdpa"
}
```

不要设置为：

```json
{
  "attention_backend": "flex_attention"
}
```

### 7.3 脚本环境变量

建议：

```bash
export ATTENTION_BACKEND=sdpa
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
```

`ATTENTION_BACKEND` 控制 draft DFlash attention；`ANGELSLIM_TARGET_ATTN_IMPL` 控制 target model forward。两者在 910B 上都建议为 `sdpa`。

## 8. 修改 4：增加防御性检查

为避免后续再次把 `BlockMask` 传给 eager/sdpa，建议在 `_compute_dflash_loss_and_accuracy()` 创建 mask 后加入：

```python
if self.attention_backend != "flex_attention" and BlockMask is not None:
    if isinstance(dflash_attn_mask, BlockMask):
        raise TypeError(
            f"attention_backend={self.attention_backend} requires dense Tensor mask, "
            f"but got BlockMask. Use create_dflash_dense_attention_mask instead."
        )
```

也可以打印一次：

```python
if not hasattr(self, "_printed_mask_debug"):
    print(
        f"[DFlash] attention_backend={self.attention_backend}, "
        f"_attn_implementation={getattr(model.config, '_attn_implementation', None)}, "
        f"mask_type={type(dflash_attn_mask)}"
    )
    self._printed_mask_debug = True
```

预期输出：

```text
[DFlash] attention_backend=sdpa, _attn_implementation=sdpa, mask_type=<class 'torch.Tensor'>
```

不应该看到：

```text
attention_backend=sdpa ... mask_type=<class 'torch.nn.attention.flex_attention.BlockMask'>
```

## 9. 修改 5：qwen_dflash.py 中可选增强检查

文件：

```text
angelslim/compressor/speculative/train/models/draft/qwen_dflash.py
```

在调用 `attn_fn` 前可加入：

```python
if self.config._attn_implementation != "flex_attention":
    if attention_mask is not None and attention_mask.__class__.__name__ == "BlockMask":
        raise TypeError(
            f"{self.config._attn_implementation} attention does not accept BlockMask. "
            "Use dense additive Tensor mask for eager/sdpa."
        )
```

这个检查不是必须，但能让错误更早、更可读。

## 10. 为什么不能简单把 backend 改成 flex_attention

在 CUDA 机器上，DFlash 原实现使用 `flex_attention + BlockMask` 是合理的。

但在 910B 上：

- `torch.nn.attention.flex_attention` 的内核路径不一定适配 NPU。
- 即使 Python 侧能 import，底层 kernel 也可能不可用或性能不可控。
- Transformers 的 Qwen3 attention 在 `eager/sdpa` 路径下不会消费 `BlockMask`。

因此 910B 推荐路线是：

```text
sdpa + dense additive Tensor mask
```

而不是：

```text
flex_attention + BlockMask
```

## 11. dense mask 的显存影响

dense additive mask 的 shape 是：

```text
[B, 1, Q_LEN, KV_LEN]
```

其中：

```text
Q_LEN = num_anchors * block_size
KV_LEN = seq_len + num_anchors * block_size
```

例如：

```text
B = 1
seq_len = 2048
num_anchors = 64
block_size = 16

Q_LEN = 1024
KV_LEN = 3072
mask elements = 1 * 1 * 1024 * 3072 = 3,145,728
bf16/fp16 mask 约 6 MB
fp32 mask 约 12 MB
```

对于 910B smoke test，建议从：

```text
model_max_length = 2048
num_anchors = 32 或 64
block_size = 16
per_device_train_batch_size = 1
```

开始。

稳定后再扩大：

```text
model_max_length = 3072
num_anchors = 128
```

## 12. 修复后的验证步骤

### 12.0 本地准确性验证脚本

已提供一个独立脚本：

```text
tools/verify_dflash_dense_mask.py
```

该脚本不依赖完整 AngelSlim 训练流程，只依赖 `torch`。它会验证三件事：

1. dense mask 的可见性是否与 DFlash 规则完全一致。
2. dense mask 是否可以执行 eager attention 中的 `scores + attention_mask`。
3. dense mask attention 输出是否与逐 query 朴素 reference attention 一致。

CPU 环境：

```bash
python tools/verify_dflash_dense_mask.py --device cpu --dtype float32 --cases 100
```

910B/NPU 环境：

```bash
python tools/verify_dflash_dense_mask.py --device npu --dtype bfloat16 --cases 200
```

如果只想验证 mask 规则，不跑 attention 数值对比：

```bash
python tools/verify_dflash_dense_mask.py \
  --device npu \
  --dtype bfloat16 \
  --cases 500 \
  --skip-attention-check
```

通过时会输出：

```text
OK: dense DFlash mask matches reference rules and eager-style Tensor addition ...
```

当前已在 WSL Docker 容器 `optimistic_galileo` 中完成验证。容器环境：

```text
torch 2.10.0+cu130
cuda available: True
cuda count: 1
npu available: False
```

已通过的验证项：

```text
CPU float32:   100 randomized cases, with attention output comparison
CPU bfloat16:  50 randomized cases, with attention output comparison
CUDA float32:  100 randomized cases, with attention output comparison
CUDA float16:  100 randomized cases, with attention output comparison
CUDA bfloat16: 100 randomized cases, with attention output comparison
CPU float32:   200 larger randomized cases, mask rule only
CUDA bfloat16: 200 larger randomized cases, mask rule only
```

说明：该容器没有 NPU/torch-npu，因此 NPU 实机仍需执行：

```bash
python tools/verify_dflash_dense_mask.py --device npu --dtype bfloat16 --cases 200
```

### 12.1 单步 mask 类型验证

启动训练前，可以临时在 `_compute_dflash_loss_and_accuracy()` 中打印：

```python
print(
    "[DFlash mask debug]",
    self.attention_backend,
    getattr(model.config, "_attn_implementation", None),
    type(dflash_attn_mask),
    getattr(dflash_attn_mask, "shape", None),
    getattr(dflash_attn_mask, "dtype", None),
)
```

910B 上期望：

```text
[DFlash mask debug] sdpa sdpa <class 'torch.Tensor'> torch.Size([B, 1, Q_LEN, KV_LEN]) torch.bfloat16
```

### 12.2 运行 smoke test

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ATTENTION_BACKEND=sdpa
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa

torchrun --standalone --nproc_per_node=8 tools/train_dflash_offline.py \
  --target_model_name_or_path "$TARGET_MODEL_PATH" \
  --draft_model_config_path "$DRAFT_CONFIG_PATH" \
  --train_hidden_path "$CACHE_DIR/smoke" \
  --output_dir "$OUTPUT_DIR/smoke" \
  --attention_backend sdpa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --num_anchors 32 \
  --block_size 16 \
  --learning_rate 6e-4 \
  --warmup_ratio 0.04 \
  --bf16 \
  --fsdp "shard_grad_op auto_wrap" \
  --fsdp_config configs/fsdp_config.json \
  --dataloader_drop_last \
  --report_to none
```

### 12.3 通过标准

应满足：

- 不再出现 `Tensor + BlockMask` 报错。
- 日志中 `_attn_implementation=sdpa`。
- mask 类型为 `torch.Tensor`。
- 前 10 到 100 step loss 非 NaN。
- FSDP 不因为最后 batch shape 不一致报错。

## 13. 常见错误

### 13.1 仍然出现 Tensor + BlockMask

说明某处仍然调用了：

```python
create_dflash_block_mask(...)
```

而当前 backend 不是 `flex_attention`。

检查：

```text
self.attention_backend
model.config._attn_implementation
type(dflash_attn_mask)
```

### 13.2 `ALL_ATTENTION_FUNCTIONS["sdpa"]` 找不到

说明当前 Transformers 版本的 attention registry 和代码不匹配。处理方式：

- 检查 `transformers.models.qwen3.modeling_qwen3.ALL_ATTENTION_FUNCTIONS` 中有哪些 key。
- 如果没有 `sdpa`，先用 `eager + dense Tensor mask` 做功能验证：

```bash
--attention_backend eager
```

### 13.3 sdpa 报 mask shape 错误

检查 dense mask shape：

```text
q shape:    [B, H, Q_LEN, D]
k shape:    [B, H_kv, KV_LEN, D]
mask shape: [B, 1, Q_LEN, KV_LEN]
```

mask 的 `Q_LEN` 必须等于 `noise_embedding.shape[1]`，`KV_LEN` 必须等于 `hidden_states.shape[1] + noise_embedding.shape[1]`。

### 13.4 显存上涨明显

dense mask 比 BlockMask 占显存。先降低：

```text
num_anchors
model_max_length
per_device_train_batch_size
```

推荐顺序：

```text
num_anchors: 128 -> 64 -> 32
model_max_length: 4096 -> 3072 -> 2048
batch_size: 固定 1
```

## 14. 推荐最终代码行为

最终应形成如下行为：

```text
CUDA + flex_attention:
  attention_backend=flex_attention
  _attn_implementation=flex_attention
  mask_type=BlockMask

Ascend 910B + sdpa:
  attention_backend=sdpa
  _attn_implementation=sdpa
  mask_type=torch.Tensor

debug/eager:
  attention_backend=eager
  _attn_implementation=eager
  mask_type=torch.Tensor
```

## 15. 一句话总结

这次错误的根因是 **AngelSlim DFlash trainer 无条件生成了 Flex Attention 专用的 `BlockMask`，但 910B 上实际使用的是 `eager/sdpa` attention；修复方式是在 `sdpa/eager` 路径下生成普通 dense additive Tensor mask，并在启动参数和 config 中统一使用 `attention_backend=sdpa`。**
