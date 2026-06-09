# Qwen3-30B-A3B DFlash Target Hidden Cache 生成指南

本文说明如何在原始 AngelSlim 框架下，为 `Qwen3-30B-A3B` 生成 DFlash 离线训练所需的 target hidden cache。该模型是纯文本 Causal LM + MoE，不是 VLM，因此不要复用 Qwen3.6-35B-A3B VLM 的 `AutoProcessor`、`pixel_values`、`image_grid_thw` 链路。

## 1. 模型关键参数

Hugging Face 官方配置中，`Qwen3-30B-A3B` 的结构信息如下：

```text
architectures = Qwen3MoeForCausalLM
model_type = qwen3_moe
hidden_size = 2048
num_hidden_layers = 48
num_attention_heads = 32
num_key_value_heads = 4
head_dim = 128
num_experts = 128
num_experts_per_tok = 8
vocab_size = 151936
max_position_embeddings = 40960
torch_dtype = bfloat16
```

官方 model card 也说明该模型是 30.5B 总参数、3.3B 激活参数、48 层、128 experts、每 token 激活 8 experts，并建议 `transformers>=4.51.0`，否则可能遇到：

```text
KeyError: 'qwen3_moe'
```

## 2. 这一步产出的 cache

每条样本保存一个 `.ckpt`，至少包含：

```python
{
    "input_ids": Tensor[1, seq_len],
    "attention_mask": Tensor[1, seq_len],
    "loss_mask": Tensor[1, seq_len],
    "hidden_states": Tensor[1, seq_len, 2048 * len(target_layer_ids)],
}
```

例如：

```python
target_layer_ids = [20, 24, 28, 32, 36, 40, 44, 47]
```

则：

```text
hidden_states.shape[-1] = 2048 * 8 = 16384
```

后续离线训练 DFlash 投机头时，会用这些 target hidden states 作为监督信号，不再每步加载完整 `Qwen3-30B-A3B` target model。

## 3. 与 Qwen3.6 VLM 版本的区别

`Qwen3-30B-A3B` 是 LLM-MoE，不需要：

```text
AutoProcessor
VLMDataCollatorWithPadding
pixel_values
image_grid_thw
video_grid_thw
M-RoPE 三维 position_ids
```

它应该使用：

```python
AutoTokenizer.from_pretrained(...)
modal_type = "LLM"
chat_template_type = "qwen3"
target_model_type = "qwen3_moe"
```

数据经过 tokenizer 和普通 LLM dataset builder 后，生成：

```python
input_ids
attention_mask
loss_mask
```

再前向 target model，抽取指定层 hidden states。

## 4. AngelSlim 需要补的 qwen3_moe 映射

当前仓库里 `qwen3` 已经支持普通 Qwen3 dense LLM，但 `Qwen3-30B-A3B` 的 `model_type` 是 `qwen3_moe`。建议显式补上映射。

文件：

```text
angelslim/compressor/speculative/train/models/model_utils.py
```

在 `MODEL_TYPE_PARAM_MAP` 中增加：

```python
"qwen3_moe": (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "qwen3",
),
```

原因：

1. `Qwen3MoeForCausalLM` 仍然是纯 LLM 结构。
2. embedding 权重路径仍是：

```text
model.embed_tokens.weight
```

3. chat template 应复用 Qwen3：

```text
qwen3
```

## 5. target model 加载方式

原始生成脚本：

```text
tools/generate_dflash_data.py
```

已经是 LLM cache 生成脚本，主要流程可以复用：

1. 读取 draft config 中的 `dflash_config.target_layer_ids`。
2. 加载 target model。
3. 用 `DatasetManager` 构造 online LLM dataset。
4. 前向 target model。
5. 保存 `input_ids`、`attention_mask`、`loss_mask`、`hidden_states`。

但在 910B 上需要改两类内容：

1. CUDA/NCCL 改成 NPU/HCCL。
2. `qwen3_moe` 的模型类型和 attention backend 要显式适配。

### 5.1 NPU 初始化

把脚本里的：

```python
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
```

改成设备自适应：

```python
def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        backend = "hccl"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return local_rank
```

再写一个统一 device 函数：

```python
def get_device(local_rank):
    if hasattr(torch, "npu") and torch.npu.is_available():
        return f"npu:{local_rank}"
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"
```

原脚本中：

```python
input_ids = input_ids.to(f"cuda:{local_rank}")
attention_mask = attention_mask.to(f"cuda:{local_rank}")
```

改成：

```python
device = get_device(local_rank)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
```

### 5.2 target model type

建议给 `tools/generate_dflash_data.py` 增加参数：

```python
parser.add_argument(
    "--target_model_type",
    type=str,
    default="qwen3_moe",
)
```

创建 target model 时传进去：

```python
target_model = create_target_model(
    backend=args.target_backend,
    model_path=args.target_model_name_or_path,
    modal_type="LLM",
    target_model_type=args.target_model_type,
    torch_dtype=torch_dtype,
    trust_remote_code=args.trust_remote_code,
)
```

### 5.3 attention implementation

原始 `TransformersBackend._prepare_model_kwargs` 默认：

```python
"attn_implementation": "flash_attention_2"
```

910B 上通常不能直接使用 CUDA 版 `flash_attention_2`。建议改成可配置：

```python
attn_impl = os.environ.get("ANGELSLIM_TARGET_ATTN_IMPL", "eager")

default_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": device,
    "trust_remote_code": True,
    "attn_implementation": attn_impl,
}
```

910B cache 生成阶段推荐先用：

```bash
export ANGELSLIM_TARGET_ATTN_IMPL=eager
```

如果当前环境已验证 `sdpa` 在 torch-npu 上可用，也可以尝试：

```bash
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
```

## 6. 推荐生成脚本最小改法

基于 `tools/generate_dflash_data.py`，需要改的核心片段如下。

### 6.1 参数新增

```python
parser.add_argument("--target_model_type", type=str, default="qwen3_moe")
parser.add_argument("--num_workers", type=int, default=4)
```

### 6.2 分布式初始化替换

```python
def init_distributed():
    local_rank = get_local_rank()
    world_size = get_world_size()

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        backend = "hccl"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
```

### 6.3 target model 创建

```python
target_model = create_target_model(
    backend=args.target_backend,
    model_path=args.target_model_name_or_path,
    modal_type="LLM",
    target_model_type=args.target_model_type,
    torch_dtype=torch_dtype,
    trust_remote_code=args.trust_remote_code,
)
```

### 6.4 DataLoader num_workers 参数化

```python
dataloader = DataLoader(
    online_train_dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    num_workers=args.num_workers,
    pin_memory=False,
    collate_fn=collate_fn,
)
```

910B / NPU 上建议先关闭 `pin_memory`。

### 6.5 前向保存

原逻辑可以保留：

```python
hidden_states, _ = target_model.get_hidden_states_and_logits(
    input_ids=input_ids,
    attention_mask=attention_mask,
    aux_hidden_states_layer_ids=target_layer_ids,
)
```

保存：

```python
ckpt = {
    "input_ids": input_ids[i : i + 1].cpu(),
    "hidden_states": hidden_states[i : i + 1].cpu().to(torch.bfloat16),
    "loss_mask": loss_mask[i : i + 1].cpu(),
    "attention_mask": attention_mask[i : i + 1].cpu(),
}
torch.save(ckpt, ckpt_path)
```

## 7. draft config 里的 target_layer_ids

Qwen3-30B-A3B 有 48 层。DFlash 训练可以先取中后层，例如：

```json
{
  "model_type": "qwen3",
  "hidden_size": 2048,
  "num_hidden_layers": 48,
  "dflash_config": {
    "target_layer_ids": [20, 24, 28, 32, 36, 40, 44, 47]
  }
}
```

注意：

1. `target_layer_ids` 是 decoder layer id，不包含 embedding 层。
2. AngelSlim 的 `_extract_auxiliary_hidden_states` 会自动加 `embed_offset=1`，所以配置里写真实 layer id。
3. 最大 layer id 是 `47`，不要写 `48`。

## 8. 910B 上推荐执行方式

### 8.0 使用 generate_dflash_data.sh

推荐直接使用仓库脚本：

```text
scripts/speculative/generate_dflash_data.sh
```

该脚本会调用：

```text
tools/generate_dflash_data.py
```

完整链路是：

```text
JSONL 训练数据
  -> DatasetManager online LLM builder
  -> AutoTokenizer + qwen3 chat template
  -> input_ids / attention_mask / loss_mask
  -> Qwen3-30B-A3B target model forward
  -> 抽取 dflash_config.target_layer_ids 对应 hidden states
  -> 保存 .ckpt hidden cache
```

如果要用该脚本在 910B 上生成 Qwen3-30B-A3B hidden cache，需要修改点汇总如下：

```text
scripts/speculative/generate_dflash_data.sh
  - 把 TARGET_MODEL_PATH / TRAIN_DATA_PATH / OUTPUT_DIR 改成环境变量可传入。
  - 把 draft config 从 configs/qwen3_dflash.json 换成 configs/qwen3_30b_a3b_dflash_npu.json。
  - 默认 NUM_GPUS 改成 1，避免 8 个 rank 各加载一份 30B target。
  - 增加 TARGET_MODEL_TYPE=qwen3_moe、CHAT_TEMPLATE_TYPE=qwen3。
  - SAMPLE_NUM 改成可选参数，smoke test 显式设置，全量生成时不传。

tools/generate_dflash_data.py
  - 增加 --target_model_type 参数，并传给 create_target_model。
  - 增加 --num_workers 参数，替代 DataLoader 中写死的 num_workers=4。
  - init_distributed 支持 torch-npu：NPU 用 hccl，CUDA 用 nccl。
  - input_ids / attention_mask 移动设备时支持 npu:{local_rank}。

angelslim/compressor/speculative/train/models/model_utils.py
  - 在 MODEL_TYPE_PARAM_MAP 增加 qwen3_moe -> qwen3 映射。

angelslim/compressor/speculative/train/models/target/target_model_wrapper.py
  - target attention implementation 改成环境变量可控。
  - 910B 上建议 ANGELSLIM_TARGET_ATTN_IMPL=eager 或 sdpa。
  - 如需单进程跨卡加载 target，可增加 ANGELSLIM_TARGET_DEVICE_MAP=auto。

angelslim/utils/utils.py
  - decide_device_for_distributed 如需支持 910B，增加 torch.npu 优先判断。
```

注意：仓库原始脚本目前还没有这些环境变量入口，也没有 `--target_model_type`、`--num_workers`、NPU/HCCL 适配。不要直接按下面命令跑原始脚本，需要先按本文第 4、5、6 节的建议修改代码，或者把这些改动整理成单独 patch 后再应用。

建议把脚本默认值改成：

```bash
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-"${ROOT_DIR}/configs/qwen3_30b_a3b_dflash_npu.json"}
TARGET_MODEL_TYPE=${TARGET_MODEL_TYPE:-qwen3_moe}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-qwen3}
NUM_GPUS=${NUM_GPUS:-${1:-1}}
```

同时建议把 `SAMPLE_NUM` 做成可选参数：

```bash
SAMPLE_NUM=${SAMPLE_NUM:-}
SAMPLE_NUM_ARGS=()
if [[ -n "$SAMPLE_NUM" ]]; then
    SAMPLE_NUM_ARGS=(--sample_num "$SAMPLE_NUM")
fi
```

Qwen3-30B-A3B 不要使用默认的 `configs/qwen3_dflash.json`，应使用：

```text
configs/qwen3_30b_a3b_dflash_npu.json
```

这个 config 中已经包含：

```json
{
  "target_model_type": "qwen3_moe",
  "hidden_size": 2048,
  "num_target_layers": 48,
  "dflash_config": {
    "target_layer_ids": [1, 12, 23, 34, 45]
  }
}
```

对应生成的 hidden shape 是：

```text
hidden_states.shape[-1] = 2048 * 5 = 10240
```

### 8.1 环境变量

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=3600
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ANGELSLIM_TARGET_ATTN_IMPL=eager
export ANGELSLIM_TARGET_DEVICE_MAP=auto
```

`NUM_GPUS=1` 配合 `ANGELSLIM_TARGET_DEVICE_MAP=auto` 的含义是：只启动一个 cache 生成进程，但让 target model 加载阶段尝试跨可见 NPU 放置权重。不要直接 `NUM_GPUS=8`，否则脚本会启动 8 个进程，每个进程都尝试加载一份完整 Qwen3-30B-A3B target model。

### 8.2 先跑 10 条 smoke test

完成上述建议改动后，用脚本跑 smoke test：

```bash
cd /path/to/AngelSlim

export TARGET_MODEL_PATH=/models/Qwen3-30B-A3B
export TRAIN_DATA_PATH=/data/qwen3_30b_train.jsonl
export OUTPUT_DIR=/data/qwen3_30b_dflash_hidden_cache_smoke
export DRAFT_MODEL_CONFIG_PATH=$PWD/configs/qwen3_30b_a3b_dflash_npu.json

export NUM_GPUS=1
export MODEL_MAX_LENGTH=2048
export SAMPLE_NUM=10
export BATCH_SIZE=1
export NUM_PROC=8
export NUM_WORKERS=2
export SHARD_SIZE=10000
export TARGET_MODEL_TYPE=qwen3_moe
export CHAT_TEMPLATE_TYPE=qwen3
export ANGELSLIM_TARGET_ATTN_IMPL=eager
export ANGELSLIM_TARGET_DEVICE_MAP=auto

bash scripts/speculative/generate_dflash_data.sh
```

等价的 Python 入口是：

```bash
torchrun --nproc_per_node=1 tools/generate_dflash_data.py \
  --target_model_name_or_path /models/Qwen3-30B-A3B \
  --draft_model_config_path configs/qwen3_30b_a3b_dflash_npu.json \
  --train_data_path /data/qwen3_30b_train.jsonl \
  --output_dir /data/qwen3_30b_dflash_hidden_cache_smoke \
  --model_max_length 2048 \
  --chat_template_type qwen3 \
  --target_model_type qwen3_moe \
  --batch_size 1 \
  --sample_num 10 \
  --num_workers 2 \
  --shard_size 10000
```

先用 `--nproc_per_node=1` 的原因是避免每张卡都加载一份完整 target model。30B MoE 的 bf16 权重很大，8 进程 data parallel cache 生成不一定划算。

### 8.3 全量生成

完成上述建议改动后，用脚本跑全量：

```bash
cd /path/to/AngelSlim

export TARGET_MODEL_PATH=/models/Qwen3-30B-A3B
export TRAIN_DATA_PATH=/data/qwen3_30b_train.jsonl
export OUTPUT_DIR=/data/qwen3_30b_dflash_hidden_cache
export DRAFT_MODEL_CONFIG_PATH=$PWD/configs/qwen3_30b_a3b_dflash_npu.json

export NUM_GPUS=1
export MODEL_MAX_LENGTH=4096
unset SAMPLE_NUM
export BATCH_SIZE=1
export NUM_PROC=16
export NUM_WORKERS=4
export SHARD_SIZE=10000
export TARGET_MODEL_TYPE=qwen3_moe
export CHAT_TEMPLATE_TYPE=qwen3
export ANGELSLIM_TARGET_ATTN_IMPL=eager
export ANGELSLIM_TARGET_DEVICE_MAP=auto

bash scripts/speculative/generate_dflash_data.sh
```

建议改动后的脚本默认不限制样本数。只有显式设置：

```bash
export SAMPLE_NUM=10
```

时才会只生成前 10 条，用于 smoke test。全量生成时执行 `unset SAMPLE_NUM` 即可。原始脚本默认写死 `--sample_num 128`，不改的话只会生成 128 条。

如果单进程模型可以稳定加载：

```bash
torchrun --nproc_per_node=1 tools/generate_dflash_data.py \
  --target_model_name_or_path /models/Qwen3-30B-A3B \
  --draft_model_config_path configs/qwen3_30b_a3b_dflash_npu.json \
  --train_data_path /data/qwen3_30b_train.jsonl \
  --output_dir /data/qwen3_30b_dflash_hidden_cache \
  --model_max_length 4096 \
  --chat_template_type qwen3 \
  --target_model_type qwen3_moe \
  --batch_size 1 \
  --num_workers 4 \
  --shard_size 10000
```

如果要并行提速，更推荐按数据 shard 启动多个独立任务，而不是一个 torchrun 里 8 rank 各自复制完整模型。例如：

```bash
python tools/split_jsonl.py \
  --input /data/qwen3_30b_train.jsonl \
  --output_dir /data/qwen3_30b_train_shards \
  --num_shards 8
```

然后每个任务处理一个 shard，并写入不同输出目录：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/generate_dflash_data.py \
  --target_model_name_or_path /models/Qwen3-30B-A3B \
  --draft_model_config_path configs/qwen3_30b_a3b_dflash_npu.json \
  --train_data_path /data/qwen3_30b_train_shards/shard_000.jsonl \
  --output_dir /data/qwen3_30b_dflash_hidden_cache/shard_000 \
  --model_max_length 4096 \
  --chat_template_type qwen3 \
  --target_model_type qwen3_moe \
  --batch_size 1
```

但只有在确认单个进程不会独占多卡、不会和其他任务抢 HBM 时，才建议这样做。

## 9. cache 正确性验证

随机检查一个 `.ckpt`：

```python
import glob
import torch

files = sorted(glob.glob("/data/qwen3_30b_dflash_hidden_cache/**/*.ckpt", recursive=True))
assert files, "no ckpt files found"

x = torch.load(files[0], map_location="cpu")

for key in ["input_ids", "attention_mask", "loss_mask", "hidden_states"]:
    assert key in x, f"missing key: {key}"

input_ids = x["input_ids"]
attention_mask = x["attention_mask"]
loss_mask = x["loss_mask"]
hidden_states = x["hidden_states"]

assert input_ids.ndim == 2
assert attention_mask.shape == input_ids.shape
assert loss_mask.shape == input_ids.shape
assert hidden_states.ndim == 3
assert hidden_states.shape[:2] == input_ids.shape

hidden_size = 2048
target_layer_num = 8
assert hidden_states.shape[-1] == hidden_size * target_layer_num
assert loss_mask.sum() > 0
assert torch.isfinite(hidden_states.float()).all()

print("file:", files[0])
print("input_ids:", tuple(input_ids.shape), input_ids.dtype)
print("attention_mask:", tuple(attention_mask.shape), attention_mask.dtype)
print("loss_mask:", tuple(loss_mask.shape), loss_mask.dtype, int(loss_mask.sum()))
print("hidden_states:", tuple(hidden_states.shape), hidden_states.dtype)
print("hidden mean/std:", hidden_states.float().mean().item(), hidden_states.float().std().item())
```

如果使用上面的 8 个 target layers，期望：

```text
hidden_states.shape[-1] = 16384
```

## 10. 常见问题

### 10.1 KeyError: qwen3_moe

原因通常是 `transformers` 版本过低。官方建议至少使用支持 Qwen3-MoE 的版本，model card 中明确提到 `transformers<4.51.0` 会遇到 `KeyError: 'qwen3_moe'`。

### 10.2 找不到 qwen3_moe 的 embedding 映射

给 `MODEL_TYPE_PARAM_MAP` 增加：

```python
"qwen3_moe": (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "qwen3",
),
```

### 10.3 910B 上 flash_attention_2 不可用

把 target model 的 attention implementation 改为环境变量可控：

```bash
export ANGELSLIM_TARGET_ATTN_IMPL=eager
```

或在验证通过后试：

```bash
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
```

### 10.4 hidden_states shape 不对

检查三件事：

1. `draft_model_config_path` 中的 `dflash_config.target_layer_ids` 数量。
2. `hidden_size` 是否按 Qwen3-30B-A3B 的 `2048` 计算。
3. layer id 是否没有越界，48 层模型最大 layer id 是 `47`。

### 10.5 loss_mask 全 0

说明 chat template 或训练数据格式没有正确标记 assistant answer。DFlash cache 虽然可以生成，但这类样本没有训练价值，应在 dataset builder 阶段过滤掉。

## 11. 离线训练接入

生成完成后，将 cache 目录传给 DFlash offline 训练：

```bash
torchrun --nproc_per_node=8 tools/train_dflash_offline.py \
  --model_name_or_path /models/Qwen3-30B-A3B \
  --hidden_cache_path /data/qwen3_30b_dflash_hidden_cache \
  --chat_template_type qwen3 \
  --model_max_length 4096 \
  --torch_dtype bfloat16 \
  ...
```

离线训练阶段只需要读取：

```text
input_ids
attention_mask
loss_mask
hidden_states
```

不再需要每步前向完整 target model。
