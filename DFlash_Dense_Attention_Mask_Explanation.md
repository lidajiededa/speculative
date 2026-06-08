# DFlash Dense Attention Mask 代码解释

本文档解释 `create_dflash_dense_attention_mask()` 是怎么写出来的、参考了什么、每段代码对应 DFlash 的哪条注意力可见性规则。

## 1. 它参考了什么

`create_dflash_dense_attention_mask()` 不是重新设计的新算法，而是把 AngelSlim 原有的 `create_dflash_block_mask()` 规则，等价翻译成 eager/sdpa 可以使用的普通 Tensor mask。

原始参考函数在：

```text
angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
```

原始核心逻辑是：

```python
def dflash_mask_mod(b, h, q_idx, kv_idx):
    q_block_id = q_idx // block_size
    anchor_pos = anchor_positions[b, q_block_id]

    is_context = kv_idx < S
    mask_context = is_context & (kv_idx < anchor_pos)

    is_draft = kv_idx >= S
    kv_block_id = (kv_idx - S) // block_size
    mask_draft = is_draft & (q_block_id == kv_block_id)

    is_valid_block = block_keep_mask[b, q_block_id]
    return (mask_context | mask_draft) & is_valid_block
```

这个函数原本是给 PyTorch `flex_attention` 用的，它返回的是 `BlockMask`。但是在 910B 上我们使用 `eager/sdpa`，它们不能消费 `BlockMask`，只能消费普通 Tensor additive mask。

因此我们要把同一套规则翻译成：

```text
visible position   -> 0
invisible position -> torch.finfo(dtype).min
```

## 2. DFlash Attention 的 Q/KV 布局

DFlash 训练时，attention 的 Query 和 Key/Value 不是普通自回归的 `[S -> S]`。

它的布局是：

```text
Q  = 所有 draft block token
KV = 原始 context token + 所有 draft block token
```

设：

```text
S          = 原始上下文序列长度
N          = 采样到的 anchor/block 数量
block_size = 每个 draft block 的 token 数
```

则：

```text
Q_LEN  = N * block_size
KV_LEN = S + N * block_size
```

KV 的布局如下：

```text
[ context_0, context_1, ..., context_{S-1},
  block0_token0, block0_token1, ..., block0_token{block_size-1},
  block1_token0, block1_token1, ..., block1_token{block_size-1},
  ...
]
```

Q 的布局如下：

```text
[ block0_token0, block0_token1, ..., block0_token{block_size-1},
  block1_token0, block1_token1, ..., block1_token{block_size-1},
  ...
]
```

也就是说：

```text
Q 中只有 draft block token
KV 中既有 context token，也有 draft block token
```

## 3. DFlash 的可见性规则

对每个 draft block 来说，它能看见两类 token。

第一类：anchor 之前的 context token。

```python
kv_idx < anchor_pos
```

注意这里是严格小于。也就是说，block 不能看见 anchor 本身及其之后的 context token。

第二类：它自己 block 里的 draft token。

```python
q_block_id == kv_block_id
```

除此之外：

```text
不同 draft block 之间互相不可见
无效 block 什么都不可见
```

所以总规则是：

```python
visible = (mask_context | mask_draft) & is_valid_block
```

对应含义：

```text
能看见 anchor 之前的 context
或者能看见同一个 draft block
并且这个 block 是有效 block
```

## 4. 为什么 dense mask 要填 0 和负无穷

Transformers 的 eager attention 会执行：

```python
attn_weights = attn_weights + attention_mask
```

所以 `attention_mask` 必须是 additive mask。

规则是：

```text
可见位置填 0
不可见位置填非常大的负数
```

这样 softmax 后：

```text
score + 0                    -> 正常参与 attention
score + torch.finfo(dtype).min -> softmax 后约等于 0
```

所以 dense mask 最后是：

```python
mask = torch.full(
    (B, 1, Q_LEN, KV_LEN),
    torch.finfo(dtype).min,
    dtype=dtype,
    device=device,
)
mask = mask.masked_fill(visible, 0.0)
```

## 5. 完整 dense mask 代码

```python
def create_dflash_dense_attention_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_idx = torch.arange(Q_LEN, device=device)
    kv_idx = torch.arange(KV_LEN, device=device)

    q_block_id = q_idx // block_size
    anchors_for_q = anchor_positions[:, q_block_id]
    valid_for_q = block_keep_mask[:, q_block_id]

    q_block_id_3d = q_block_id.view(1, Q_LEN, 1)
    kv_idx_3d = kv_idx.view(1, 1, KV_LEN)

    is_context = kv_idx_3d < S
    mask_context = is_context & (kv_idx_3d < anchors_for_q.unsqueeze(-1))

    is_draft = kv_idx_3d >= S
    kv_block_id = (kv_idx_3d - S) // block_size
    mask_draft = is_draft & (kv_block_id == q_block_id_3d)

    visible = (mask_context | mask_draft) & valid_for_q.unsqueeze(-1)
    visible = visible.unsqueeze(1)

    mask = torch.full(
        (B, 1, Q_LEN, KV_LEN),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return mask.masked_fill(visible, 0.0)
```

## 6. 逐段解释

### 6.1 计算 Q/KV 长度

```python
B, N = anchor_positions.shape
Q_LEN = N * block_size
KV_LEN = S + N * block_size
```

含义：

```text
B: batch size
N: 每条样本采样出来的 anchor/block 数量
Q_LEN: 所有 draft block 的总 token 数
KV_LEN: context token 数 + draft block token 数
```

### 6.2 构造 q_idx 和 kv_idx

```python
q_idx = torch.arange(Q_LEN, device=device)
kv_idx = torch.arange(KV_LEN, device=device)
```

含义：

```text
q_idx:  0 到 Q_LEN-1，每个位置代表一个 query token
kv_idx: 0 到 KV_LEN-1，每个位置代表一个 key/value token
```

### 6.3 计算每个 query 属于哪个 block

原始逐元素逻辑：

```python
q_block_id = q_idx // block_size
```

dense 版本：

```python
q_block_id = q_idx // block_size
```

举例：

```text
block_size = 4
q_idx      = [0, 1, 2, 3, 4, 5, 6, 7]
q_block_id = [0, 0, 0, 0, 1, 1, 1, 1]
```

### 6.4 找到每个 query 对应的 anchor

原始逐元素逻辑：

```python
anchor_pos = anchor_positions[b, q_block_id]
```

dense 版本：

```python
anchors_for_q = anchor_positions[:, q_block_id]
```

如果：

```text
anchor_positions = [[5, 12]]
block_size = 4
```

那么：

```text
anchors_for_q = [[5, 5, 5, 5, 12, 12, 12, 12]]
```

也就是每个 query token 都知道自己所在 block 的 anchor 位置。

### 6.5 找到每个 query 是否属于有效 block

原始逐元素逻辑：

```python
is_valid_block = block_keep_mask[b, q_block_id]
```

dense 版本：

```python
valid_for_q = block_keep_mask[:, q_block_id]
```

如果某个 block 无效，那么该 block 里的所有 query token 都不可见任何 KV token。

### 6.6 计算 context 可见性

原始逐元素逻辑：

```python
is_context = kv_idx < S
mask_context = is_context & (kv_idx < anchor_pos)
```

dense 版本：

```python
kv_idx_3d = kv_idx.view(1, 1, KV_LEN)

is_context = kv_idx_3d < S
mask_context = is_context & (kv_idx_3d < anchors_for_q.unsqueeze(-1))
```

这里的 broadcasting 会生成：

```text
[B, Q_LEN, KV_LEN]
```

含义是：

```text
每个 batch、每个 query、每个 kv 位置是否满足 context 可见规则
```

### 6.7 计算 draft block 可见性

原始逐元素逻辑：

```python
is_draft = kv_idx >= S
kv_block_id = (kv_idx - S) // block_size
mask_draft = is_draft & (q_block_id == kv_block_id)
```

dense 版本：

```python
is_draft = kv_idx_3d >= S
kv_block_id = (kv_idx_3d - S) // block_size
mask_draft = is_draft & (kv_block_id == q_block_id_3d)
```

含义：

```text
如果 KV token 来自 draft 区域，则只有同一个 block 的 query 能看见它
```

也就是说：

```text
block0 的 query 能看 block0 的 draft KV
block1 的 query 能看 block1 的 draft KV
block0 不能看 block1
block1 不能看 block0
```

### 6.8 合并可见性规则

```python
visible = (mask_context | mask_draft) & valid_for_q.unsqueeze(-1)
```

含义：

```text
可见 = 能看 context 或能看自己 block 的 draft
       并且当前 block 是有效 block
```

### 6.9 增加 head 维度

```python
visible = visible.unsqueeze(1)
```

从：

```text
[B, Q_LEN, KV_LEN]
```

变成：

```text
[B, 1, Q_LEN, KV_LEN]
```

中间的 `1` 是 head 维度，可以广播到所有 attention heads。

### 6.10 生成 additive Tensor mask

```python
mask = torch.full(
    (B, 1, Q_LEN, KV_LEN),
    torch.finfo(dtype).min,
    dtype=dtype,
    device=device,
)
return mask.masked_fill(visible, 0.0)
```

含义：

```text
先把所有位置都设为不可见
再把 visible=True 的位置改成 0
```

最终得到 eager/sdpa 可以直接使用的 Tensor mask。

## 7. 一个小例子

假设：

```text
S = 6
block_size = 2
N = 2
anchor_positions = [[2, 5]]
block_keep_mask = [[True, True]]
```

则：

```text
Q_LEN = 2 * 2 = 4
KV_LEN = 6 + 4 = 10
```

KV 布局：

```text
0  1  2  3  4  5  | 6  7 | 8  9
c0 c1 c2 c3 c4 c5 | b0 b0 | b1 b1
```

Q 布局：

```text
0  1 | 2  3
b0 b0 | b1 b1
```

block0 的 anchor 是 2，所以 block0 的 query 能看：

```text
context: kv 0, 1
draft:   kv 6, 7
```

block1 的 anchor 是 5，所以 block1 的 query 能看：

```text
context: kv 0, 1, 2, 3, 4
draft:   kv 8, 9
```

不能看：

```text
block0 不能看 kv 8, 9
block1 不能看 kv 6, 7
任何 block 都不能看 anchor 及之后的 context
```

## 8. 如何验证它是正确的

验证脚本：

```text
tools/verify_dflash_dense_mask.py
```

这个脚本做了三层验证：

1. 用慢速 Python for-loop 实现一份 reference visible mask。
2. 比较 dense mask 的可见位置是否与 reference 完全一致。
3. 随机生成 `q/k/v`，比较 dense mask attention 和逐 query reference attention 的输出。

运行：

```bash
python tools/verify_dflash_dense_mask.py --device cpu --dtype float32 --cases 100
```

910B 上运行：

```bash
python tools/verify_dflash_dense_mask.py --device npu --dtype bfloat16 --cases 200
```

通过时输出：

```text
OK: dense DFlash mask matches reference rules and eager-style Tensor addition ...
```

## 9. 已完成的验证

已在 WSL Docker 容器 `optimistic_galileo` 中验证：

```text
torch 2.10.0+cu130
cuda available: True
cuda count: 1
npu available: False
```

通过项：

```text
CPU float32:   100 cases，包含 attention 输出对比
CPU bfloat16:  50 cases，包含 attention 输出对比
CUDA float32:  100 cases，包含 attention 输出对比
CUDA float16:  100 cases，包含 attention 输出对比
CUDA bfloat16: 100 cases，包含 attention 输出对比

CPU float32:   200 个更大随机形状，仅 mask 规则校验
CUDA bfloat16: 200 个更大随机形状，仅 mask 规则校验
```

由于该容器没有 NPU，还需要在 910B 机器上补充：

```bash
python tools/verify_dflash_dense_mask.py --device npu --dtype bfloat16 --cases 200
```

## 10. 一句话总结

`create_dflash_dense_attention_mask()` 本质上就是把原始 DFlash `BlockMask` 的逐元素规则：

```text
能看 anchor 之前的 context
或能看同一个 draft block
并且 block 有效
```

翻译成 eager/sdpa 可以使用的普通 Tensor additive mask：

```text
可见位置填 0
不可见位置填 torch.finfo(dtype).min
```

