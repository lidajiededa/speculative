# Qwen3-30B-A3B DFlash 投机头在昇腾 910B 8x64G NPU 上的训练适配文档

## 目标

在 `Tencent/AngelSlim` 的 DFlash 训练框架基础上，适配昇腾 910B NPU 机器，使用 8 张 64G NPU 训练出 Qwen3-30B-A3B 的 DFlash speculative draft head。

本文默认本地仓库路径为：

```text
D:\workspace\speculative\AngelSlim
```

我已经补了一份 starter config：

```text
AngelSlim/configs/qwen3_30b_a3b_dflash_npu.json
```

这份 config 是面向 Qwen3-30B-A3B 的 DFlash 头起步配置，训练前仍建议用你的目标模型目录里的 `config.json` 再核对 `hidden_size / num_hidden_layers / num_attention_heads / num_key_value_heads / vocab_size / tie_word_embeddings`。

## 总体建议

不要优先走当前 AngelSlim 的 online DFlash 路线。

原因是现有 online 代码会在每个 `torchrun` rank 上加载一整份 target model：

```text
tools/train_dflash_online.py
  -> create_target_model(...)
  -> AutoModelForCausalLM.from_pretrained(..., device_map=f"cuda:{local_rank}")
```

Qwen3-30B-A3B 虽然 active 参数约 3B，但完整权重仍接近 30B。bf16 权重大约 60GB，单卡 64G 再叠加 activation、hidden states、draft model、optimizer 和框架开销，基本不可控。

推荐路线是：

1. 先用 NPU 分片 target model 生成 DFlash hidden cache。
2. 再用 offline DFlash trainer 训练 draft head。

也就是：

```text
Raw conversation data
  -> target model hidden cache (.ckpt)
  -> offline DFlash training
  -> DFlash draft checkpoint
```

## 需要适配的核心问题

### 1. CUDA/NCCL 要换成 NPU/HCCL

当前代码里有几类硬编码：

- `torch.cuda.set_device(...)`
- `.to(f"cuda:{local_rank}")`
- `dist.init_process_group(backend="nccl")`
- `device="cuda"`
- shell 脚本里的 `NCCL_*` 和 `CUDA_DEVICE_MAX_CONNECTIONS`

NPU 上应改为：

- `import torch_npu`
- `torch.npu.set_device(f"npu:{local_rank}")`
- `dist.init_process_group(backend="hccl")`
- `.to(f"npu:{local_rank}")`
- shell 使用 `ASCEND_RT_VISIBLE_DEVICES`，不要设置 CUDA/NCCL 专用环境变量。

### 2. target model 不能使用 `flash_attention_2`

`target_model_wrapper.py` 里 HuggingFace target backend 默认写死：

```python
"attn_implementation": "flash_attention_2"
```

NPU 上通常应改为：

```python
"attn_implementation": "sdpa"
```

如果你当前 CANN/torch_npu/transformers 组合下 SDPA 不通，再降级到：

```python
"attn_implementation": "eager"
```

### 3. DFlash 默认 `flex_attention` 不适合直接搬到 NPU

DFlash trainer 当前用 `torch.nn.attention.flex_attention.create_block_mask` 构造 `BlockMask`。这个路径主要面向 PyTorch flex attention，NPU 上大概率不支持或不稳定。

NPU 适配建议：

- 首选 `attention_backend=sdpa`，但把 DFlash BlockMask 改成 dense additive attention mask。
- 如果 SDPA 不接受该 mask 或报算子错误，降级到 `attention_backend=eager`。
- 如果用 eager，先把 `num_anchors` 降到 64 或 128，避免 attention matrix 太大。

### 4. Qwen3-30B-A3B 是 MoE target，需要配置 `target_model_type=qwen3_moe`

我新增的 starter config 已包含：

```json
"target_model_type": "qwen3_moe"
```

并使用：

```json
"hidden_size": 2048,
"num_target_layers": 48,
"dflash_config": {
  "target_layer_ids": [1, 12, 23, 34, 45]
}
```

这对应 5 层 DFlash draft head，每层从 target 的不同深度取 hidden states。若你的 Qwen3-30B-A3B checkpoint 的层数不是 48，需要重新生成 `target_layer_ids`。

## 建议代码修改

下面是按文件组织的修改建议。建议先在新分支上改：

```bash
git checkout -b npu-dflash-qwen3-30b-a3b
```

如果你当前目录是 zip 解压包，没有 `.git`，就直接改文件即可。

### 修改 1：增加设备抽象

文件：

```text
AngelSlim/angelslim/utils/utils.py
```

增加几个 helper：

```python
def is_npu_available():
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def get_accelerator_type():
    forced = os.environ.get("ANGELSLIM_DEVICE_TYPE")
    if forced:
        return forced.lower()
    if is_npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_accelerator_device_from_env():
    device_type = get_accelerator_type()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_type == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(f"npu:{local_rank}")
    elif device_type == "cuda":
        torch.cuda.set_device(local_rank)


def get_dist_backend():
    return "hccl" if get_accelerator_type() == "npu" else "nccl"
```

把 `get_best_device()` 改为优先 NPU：

```python
def get_best_device():
    if is_npu_available():
        return "npu:0"
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    elif torch.xpu.is_available():
        return "xpu:0"
    else:
        return "cpu"
```

把 `decide_device_for_distributed()` 改成：

```python
def decide_device_for_distributed():
    rank, _, local_rank = _get_distributed_info()
    device_type = get_accelerator_type()

    if device_type == "cpu":
        return "cpu"
    if local_rank >= 0:
        return f"{device_type}:{local_rank}"
    if rank >= 0:
        return f"{device_type}:{rank}"
    return f"{device_type}:0"
```

### 修改 2：target model 支持 NPU attention backend

文件：

```text
AngelSlim/angelslim/compressor/speculative/train/models/target/target_model_wrapper.py
```

在 `TransformersBackend.load_model()` 开头调用：

```python
from angelslim.utils import set_accelerator_device_from_env

set_accelerator_device_from_env()
```

把 `_prepare_model_kwargs()` 里的 attention 改成可配置：

```python
attn_impl = os.environ.get("ANGELSLIM_TARGET_ATTN_IMPL")
if attn_impl is None:
    attn_impl = "sdpa" if str(device).startswith("npu") else "flash_attention_2"

default_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": device,
    "trust_remote_code": True,
    "attn_implementation": attn_impl,
}
```

如果你要做 Qwen3-30B-A3B hidden cache 生成，单卡放不下 target model。需要另写 sharded hidden 生成脚本，让 target model 使用 `device_map="auto"` 或手写 layer-to-NPU map，而不是 `device_map=f"npu:{local_rank}"`。

### 修改 3：DFlash trainer 的 target embedding/head 加载设备改成当前设备

文件：

```text
AngelSlim/angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
```

当前代码：

```python
target_components = TargetEmbeddingsAndHead.from_pretrained(
    target_model_path,
    embed_key=embed_weight_key,
    lm_head_key=lm_head_key,
    device="cuda",
    trust_remote_code=trust_remote_code,
)
```

改为：

```python
from angelslim.utils import decide_device_for_distributed

target_components = TargetEmbeddingsAndHead.from_pretrained(
    target_model_path,
    embed_key=embed_weight_key,
    lm_head_key=lm_head_key,
    device=decide_device_for_distributed(),
    trust_remote_code=trust_remote_code,
)
```

`TargetEmbeddingsAndHead.from_pretrained()` 的默认参数也建议从：

```python
device: str = "cuda"
```

改成：

```python
device: str = None
```

并在函数内：

```python
if device is None:
    device = decide_device_for_distributed()
```

### 修改 4：为 NPU 增加 dense DFlash attention mask

文件：

```text
AngelSlim/angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
```

保留原来的 `create_dflash_block_mask()` 给 `flex_attention` 用，新增 dense mask：

```python
def create_dflash_dense_attention_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_idx = torch.arange(Q_LEN, device=device)
    kv_idx = torch.arange(KV_LEN, device=device)

    q_block_id = (q_idx // block_size).view(1, Q_LEN)
    q_block_id_b = q_block_id.expand(B, -1)
    anchor_for_q = torch.gather(anchor_positions, 1, q_block_id_b)
    valid_for_q = torch.gather(block_keep_mask, 1, q_block_id_b)

    is_context = kv_idx.view(1, 1, KV_LEN) < S
    mask_context = is_context & (kv_idx.view(1, 1, KV_LEN) < anchor_for_q.unsqueeze(-1))

    is_draft = kv_idx.view(1, 1, KV_LEN) >= S
    kv_block_id = ((kv_idx - S).clamp(min=0) // block_size).view(1, 1, KV_LEN)
    mask_draft = is_draft & (kv_block_id == q_block_id_b.unsqueeze(-1))

    allowed = valid_for_q.unsqueeze(-1) & (mask_context | mask_draft)

    attn_mask = torch.zeros((B, 1, Q_LEN, KV_LEN), device=device, dtype=dtype)
    attn_mask = attn_mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return attn_mask
```

然后在 `_compute_dflash_loss_and_accuracy()` 里，把原来无条件调用 `create_dflash_block_mask()` 的地方改成：

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

注意：dense mask 会明显增加显存压力。Qwen3-30B-A3B + 910B 起步建议：

```text
num_anchors=64 或 128
block_size=16
attention_backend=sdpa
```

跑通后再尝试 `num_anchors=256`。不要一开始就用 512。

### 修改 5：generate_dflash_data.py 的 NPU 后端

文件：

```text
AngelSlim/tools/generate_dflash_data.py
```

修改分布式初始化：

```python
from angelslim.utils import get_dist_backend, set_accelerator_device_from_env, decide_device_for_distributed

def init_distributed():
    if get_world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend=get_dist_backend())
    set_accelerator_device_from_env()
```

把：

```python
input_ids = input_ids.to(f"cuda:{local_rank}")
attention_mask = attention_mask.to(f"cuda:{local_rank}")
```

改成：

```python
device = decide_device_for_distributed()
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
```

但是再次强调：这个脚本即使改成 NPU，默认仍然是“每个 rank 加载完整 target model”。对 Qwen3-30B-A3B 不推荐直接 8 rank 跑。建议另写 sharded hidden cache 生成脚本。

### 修改 6：train_dflash_online/offline 顶部导入 torch_npu

文件：

```text
AngelSlim/tools/train_dflash_online.py
AngelSlim/tools/train_dflash_offline.py
AngelSlim/tools/generate_dflash_data.py
```

在 `import torch` 后加入：

```python
if os.environ.get("ANGELSLIM_DEVICE_TYPE", "").lower() == "npu":
    import torch_npu  # noqa: F401
```

如果你希望少改代码，也可以在训练入口最前面加入：

```python
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
```

### 修改 7：Qwen3 MoE 参数映射

文件：

```text
AngelSlim/angelslim/compressor/speculative/train/models/model_utils.py
```

给 `MODEL_TYPE_PARAM_MAP` 增加：

```python
"qwen3_moe": (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "qwen3",
),
```

虽然 DFlash 当前入口主要从 config/default 取 `lm_head_key/embed_weight_key`，但加上这个映射能避免后续自动推断或工具脚本遇到 `qwen3_moe` 时退化。

## Qwen3-30B-A3B DFlash config

已新增：

```text
AngelSlim/configs/qwen3_30b_a3b_dflash_npu.json
```

关键内容：

```json
{
  "architectures": ["QwenDFlashDraftModel"],
  "target_model_type": "qwen3_moe",
  "block_size": 16,
  "hidden_size": 2048,
  "intermediate_size": 6144,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "num_hidden_layers": 5,
  "num_target_layers": 48,
  "dflash_config": {
    "mask_token_id": 151669,
    "target_layer_ids": [1, 12, 23, 34, 45]
  },
  "vocab_size": 151936,
  "lm_head_key": "lm_head.weight",
  "embed_weight_key": "model.embed_tokens.weight",
  "num_anchors": 128,
  "loss_decay_gamma": 7.0,
  "attention_backend": "sdpa"
}
```

核对方法：

```bash
python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/path/to/Qwen3-30B-A3B", trust_remote_code=True)
for k in [
    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
    "tie_word_embeddings", "max_position_embeddings", "rope_theta"
]:
    print(k, getattr(cfg, k, None))
PY
```

如果 `num_hidden_layers` 不是 48，按下面方式重算 5 个 target layer：

```python
num_target_layers = cfg.num_hidden_layers
num_draft_layers = 5
start = 1
end = num_target_layers - 3
target_layer_ids = [
    int(round(start + (i * (end - start)) / (num_draft_layers - 1)))
    for i in range(num_draft_layers)
]
print(target_layer_ids)
```

## NPU 环境准备

在 910B 机器上确认：

```bash
npu-smi info
python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
PY
```

建议环境变量：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ANGELSLIM_DEVICE_TYPE=npu
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

不要设置原 CUDA 脚本里的：

```bash
NCCL_*
CUDA_DEVICE_MAX_CONNECTIONS
CUDA_VISIBLE_DEVICES
```

除非你的容器/调度系统有特殊兼容层。

## 推荐训练流程

### Step 0：准备数据

输入数据使用 conversation JSON/JSONL：

```json
{
  "id": "0",
  "conversations": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
  ]
}
```

DFlash 会只在 assistant response 区域计算 loss，并过滤 loss token 少于 `2 * block_size` 的样本。

建议先用 1K 到 5K 样本做 smoke test。

### Step 1：生成 target hidden cache

对 Qwen3-30B-A3B，建议新建一个 `tools/generate_dflash_data_npu_sharded.py`，从 `tools/generate_dflash_data.py` 拷贝后做两点变化：

1. 不用 `torchrun` 多进程分数据。
2. target model 用 sharded device map 放到 8 张 NPU。

核心加载逻辑示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu

model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map="auto",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH, trust_remote_code=True)
```

如果 `device_map="auto"` 在你的 transformers/accelerate/torch_npu 组合下不能正确识别 NPU，就改成手写 map，把 embedding、layers、norm、lm_head 分配到 `npu:0..npu:7`。

hidden cache 保存格式必须和现有 offline trainer 一致：

```python
ckpt = {
    "input_ids": input_ids.cpu(),
    "hidden_states": hidden_states.cpu().to(torch.bfloat16),
    "loss_mask": loss_mask.cpu(),
    "attention_mask": attention_mask.cpu(),
}
torch.save(ckpt, ckpt_path)
```

先用小样本：

```bash
export TARGET_MODEL_PATH=/models/Qwen3-30B-A3B
export TRAIN_DATA_PATH=/data/dflash_train_1k.jsonl
export OUTPUT_DIR=/data/qwen3_30b_a3b_dflash_hidden_smoke

python tools/generate_dflash_data_npu_sharded.py \
  --target_model_name_or_path $TARGET_MODEL_PATH \
  --draft_model_config_path configs/qwen3_30b_a3b_dflash_npu.json \
  --train_data_path $TRAIN_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --model_max_length 2048 \
  --chat_template_type qwen3 \
  --sample_num 1000 \
  --shard_size 10000
```

如果 hidden cache 生成阶段 OOM：

- 先把 `model_max_length` 从 3072 降到 2048 或 1536。
- batch size 保持 1。
- 使用 CPU offload 或更细的 device map。
- 关闭 logits 保存；DFlash offline 只需要 `hidden_states`，不需要 logits。

### Step 2：offline 训练 smoke test

建议先不用 FSDP，只跑 DDP，降低适配变量：

```bash
cd /path/to/AngelSlim

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ANGELSLIM_DEVICE_TYPE=npu
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
export TARGET_MODEL_PATH=/models/Qwen3-30B-A3B
export TRAIN_HIDDEN_PATH=/data/qwen3_30b_a3b_dflash_hidden_smoke
export OUTPUT_DIR=/data/qwen3_30b_a3b_dflash_head_smoke

torchrun --standalone --nproc_per_node 8 \
  tools/train_dflash_offline.py \
  --target_model_name_or_path $TARGET_MODEL_PATH \
  --draft_model_config_path configs/qwen3_30b_a3b_dflash_npu.json \
  --train_hidden_path $TRAIN_HIDDEN_PATH \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-4 \
  --warmup_ratio 0.04 \
  --max_grad_norm 1.0 \
  --model_max_length 2048 \
  --chat_template_type qwen3 \
  --attention_backend sdpa \
  --block_size 16 \
  --num_anchors 64 \
  --loss_decay_gamma 7 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 100 \
  --bf16 \
  --lr_scheduler_type cosine \
  --dataloader_drop_last \
  --report_to none
```

如果 `sdpa` 报算子或 mask 错误，改成：

```bash
--attention_backend eager
--num_anchors 32
```

如果 DDP 跑通，再考虑加 FSDP：

```bash
--fsdp "shard_grad_op auto_wrap" \
--fsdp_config configs/fsdp_config.json
```

但因为 DFlash head 本身相对 target 小很多，NPU 上优先保证 DDP 稳定性。

### Step 3：扩大训练

跑通 smoke test 后逐步增加：

```text
model_max_length: 2048 -> 3072
num_anchors: 64 -> 128 -> 256
sample_num: 1K -> 10K -> full data
num_train_epochs: 1 -> 6/12
```

推荐正式 offline 训练命令：

```bash
torchrun --standalone --nproc_per_node 8 \
  tools/train_dflash_offline.py \
  --target_model_name_or_path $TARGET_MODEL_PATH \
  --draft_model_config_path configs/qwen3_30b_a3b_dflash_npu.json \
  --train_hidden_path $TRAIN_HIDDEN_PATH \
  --eval_hidden_path $EVAL_HIDDEN_PATH \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 6e-4 \
  --warmup_ratio 0.04 \
  --max_grad_norm 1.0 \
  --model_max_length 3072 \
  --chat_template_type qwen3 \
  --attention_backend sdpa \
  --block_size 16 \
  --num_anchors 128 \
  --loss_decay_gamma 7 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 1000 \
  --bf16 \
  --lr_scheduler_type cosine \
  --dataloader_drop_last \
  --report_to wandb \
  --wandb_project angelslim-qwen3-30b-a3b-dflash-npu \
  --wandb_run_name qwen3-30b-a3b-dflash-offline-npu
```

如果显存很稳，可以尝试：

```bash
--num_anchors 256
--gradient_accumulation_steps 2
```

不建议第一轮就用 `num_anchors=512`。dense mask 在 NPU 上的显存和算子压力会明显高于原始 CUDA flex attention 路径。

## Online 训练可选改造

只有在你完成 target model sharding 后，才建议尝试 online。否则 8 个 rank 每个 rank 各加载一份 Qwen3-30B-A3B target，几乎一定 OOM。

online 要改的核心点：

- `train_dflash_online.py` 加 `--target_device_map auto` 或 `--target_device_map_json`。
- `create_target_model()` / `TransformersBackend._prepare_model_kwargs()` 支持传入 sharded device map。
- `OnlineDFlashTrainer.prepare_data_for_draft_model()` 里 target forward 的输入需要放到 target embedding 所在设备，输出 hidden states 再移回当前 rank 的训练设备。
- 这种跨设备拷贝会比较复杂，也会影响吞吐。

因此，Qwen3-30B-A3B + 8x64G NPU 的第一版建议只做 offline。

## 验证训练是否正常

训练早期看三个信号：

1. `train/loss` 能稳定下降。
2. `train/accuracy` 高于随机，并逐步上升。
3. 没有 `nan/inf`，没有 HCCL timeout。

如果 loss 不降：

- 检查 `loss_mask` 是否全 0。
- 检查 target hidden cache 的 `target_layer_ids` 是否和 DFlash config 一致。
- 检查 `mask_token_id=151669` 是否存在于 tokenizer vocab。
- 检查 `lm_head_key/embed_weight_key` 是否和 target checkpoint 权重名一致。

## 常见问题

### HCCL 初始化失败

排查：

```bash
echo $ASCEND_RT_VISIBLE_DEVICES
npu-smi info
```

确认没有 CUDA/NCCL 环境变量污染。单机 8 卡先用：

```bash
torchrun --standalone --nproc_per_node 8 ...
```

### `flash_attention_2` 报错

说明 target 仍在走 CUDA flash-attn 路径。确认：

```bash
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
```

并确认 `target_model_wrapper.py` 已改成读取该环境变量。

### `BlockMask` 或 `flex_attention` 报错

说明 DFlash 仍在走 CUDA/PyTorch flex attention 路径。确认：

```bash
--attention_backend sdpa
```

并确认 `online_dflash_trainer.py` 已实现 dense mask 分支。

### NPU OOM

按顺序降：

1. `num_anchors`: 128 -> 64 -> 32
2. `model_max_length`: 3072 -> 2048 -> 1536
3. `per_device_train_batch_size`: 保持 1
4. 增大 `gradient_accumulation_steps`
5. `attention_backend`: sdpa 不行再 eager，但 eager 可能更吃显存

### hidden cache 太大

DFlash cache 很大。以 `S=3072`、`hidden_size=2048`、5 个 target layer、bf16 估算：

```text
3072 * 2048 * 5 * 2 bytes ~= 60 MB / sample
```

10 万样本大约 6 TB。正式训练前要规划本地 NVMe 或共享存储。可以先用 2048 长度、抽样数据验证收益，再扩大。

## 最小改造清单

必须改：

- `angelslim/utils/utils.py`
  - NPU device detection
  - `decide_device_for_distributed()`
  - `get_dist_backend()`
- `target_model_wrapper.py`
  - target attention 从 `flash_attention_2` 改为 NPU 可配置 `sdpa/eager`
- `online_dflash_trainer.py`
  - `device="cuda"` 改为当前设备
  - 增加 dense DFlash attention mask，供 `sdpa/eager` 使用
- `generate_dflash_data.py`
  - `nccl/cuda` 改为 `hccl/npu`
  - 但 Qwen3-30B-A3B 推荐另写 sharded generator
- `model_utils.py`
  - 增加 `qwen3_moe` 参数映射

已经新增：

- `configs/qwen3_30b_a3b_dflash_npu.json`

推荐新增：

- `tools/generate_dflash_data_npu_sharded.py`
  - 用 8 张 NPU sharded 加载 Qwen3-30B-A3B
  - 生成 offline `.ckpt` hidden cache

## 推荐最终路线

```text
1. 改 NPU device / HCCL / target sdpa / dense DFlash mask
2. 用 qwen3_30b_a3b_dflash_npu.json 核对 target config
3. sharded target 生成 1K hidden cache
4. offline DDP 训练 1 epoch smoke test
5. 调大 num_anchors 到 128/256，调大 max_length 到 3072
6. 生成正式 hidden cache
7. offline 训练 6 到 12 epoch
8. 用 dflash_benchmark.py 的 NPU 适配版本评估 accepted length 和 speedup
```

