# Qwen3-30B-A3B：2 张 NPU 布 1 个模型，4 个实例并行生成 hidden_states

本文说明如何在 8 张 64G 昇腾 910B 上，为 `Qwen3-30B-A3B` 生成 DFlash 离线训练所需的 target hidden states，并实现：

```text
实例 0：NPU 0,1 -> 加载 1 份 Qwen3-30B-A3B -> 处理数据 shard_0
实例 1：NPU 2,3 -> 加载 1 份 Qwen3-30B-A3B -> 处理数据 shard_1
实例 2：NPU 4,5 -> 加载 1 份 Qwen3-30B-A3B -> 处理数据 shard_2
实例 3：NPU 6,7 -> 加载 1 份 Qwen3-30B-A3B -> 处理数据 shard_3
```

目标是 **2 张卡切 1 个 target model，4 个独立实例并行跑数据**。这不是原始脚本的默认行为，需要按本文修改后再跑。

## 0. 结论和关键修正

这套并行思路是：

```text
外部 4 实例数据并行 + 每个实例内部 2 NPU model dispatch
```

方向是对的，但不能只把 `device_map` 改成 `auto` 就结束。对当前 AngelSlim 代码，还必须补下面两个关键点：

1. `generate_dflash_data.py` 不能固定把 `input_ids` 放到 `npu:0` / `cuda:0`，而应放到 target model embedding 层所在设备。
2. `TransformersBackend._extract_auxiliary_hidden_states()` 不能直接 `torch.cat()` 多层 hidden states；`device_map=auto` 后，不同 layer 的 hidden states 可能在不同 NPU 上，必须先搬到同一个设备再拼接。

另外，`model_utils.py` 里的 `qwen3_moe` 映射对 hidden states 生成本身不是硬依赖。它主要影响自动推断 `lm_head_key / embed_weight_key / chat_template_type` 的链路，建议补，但不是这一步能否前向生成 hidden cache 的核心。

## 1. 先明确：不要用 torchrun --nproc_per_node=2

原始 `tools/generate_dflash_data.py` 的分布式逻辑是 data parallel，不是 tensor/model parallel。

如果执行：

```bash
torchrun --nproc_per_node=2 tools/generate_dflash_data.py ...
```

实际含义是：

```text
rank0 -> npu/cuda:0 -> 加载一份完整 target model
rank1 -> npu/cuda:1 -> 加载一份完整 target model
```

这不是“2 张卡布一个模型”，而是“2 张卡各自布一份模型”。

本文推荐的方式是：

```text
每个实例只启动 1 个 Python 进程。
每个进程只看到 2 张 NPU。
进程内部通过 device_map=auto 尝试把 Qwen3-30B-A3B 切到这 2 张 NPU 上。
```

也就是：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 NUM_GPUS=1 bash scripts/speculative/generate_dflash_data.sh
ASCEND_RT_VISIBLE_DEVICES=2,3 NUM_GPUS=1 bash scripts/speculative/generate_dflash_data.sh
ASCEND_RT_VISIBLE_DEVICES=4,5 NUM_GPUS=1 bash scripts/speculative/generate_dflash_data.sh
ASCEND_RT_VISIBLE_DEVICES=6,7 NUM_GPUS=1 bash scripts/speculative/generate_dflash_data.sh
```

四个实例分别处理四份不同 JSONL 数据。

## 2. 总体链路

```text
原始训练 JSONL
  -> split_jsonl 拆成 4 份
  -> 启动 4 个独立 hidden generation 实例
  -> 每个实例只可见 2 张 NPU
  -> 每个实例加载 1 份 Qwen3-30B-A3B target model
  -> target model 在本实例的 2 张 NPU 内部切分
  -> 每个实例写自己的 hidden cache 输出目录
  -> 后续 offline DFlash 训练读取总 cache 根目录
```

输出目录建议：

```text
/data/qwen3_30b_dflash_hidden_cache/
  instance_0/
    shard_00000/sample_00000000_rank0.ckpt
  instance_1/
    shard_00000/sample_00000000_rank0.ckpt
  instance_2/
    shard_00000/sample_00000000_rank0.ckpt
  instance_3/
    shard_00000/sample_00000000_rank0.ckpt
```

每个实例内部文件名可以重复，因为输出目录不同。

## 3. Qwen3-30B-A3B 配置要求

该模型是 LLM-MoE，不是 VLM。生成 hidden states 时使用：

```text
modal_type = LLM
chat_template_type = qwen3
target_model_type = qwen3_moe
tokenizer = AutoTokenizer
```

建议使用已有配置：

```text
configs/qwen3_30b_a3b_dflash_npu.json
```

其中关键字段是：

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

因此生成出来的 hidden states 末维应为：

```text
2048 * 5 = 10240
```

## 4. 需要修改哪些文件

只列修改建议，不在本文直接改源码。

### 4.1 model_utils.py：可选增加 qwen3_moe 映射

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

```text
Qwen3-30B-A3B 的 model_type 是 qwen3_moe。
它仍然是纯 LLM，embedding 路径是 model.embed_tokens.weight。
chat template 复用 qwen3。
```

注意：这不是 `tools/generate_dflash_data.py` 生成 hidden states 的硬依赖。当前生成脚本不会调用 `infer_model_params()`，target model 也是通过 `AutoModelForCausalLM.from_pretrained()` 直接加载。该映射更偏向后续训练、自动推断参数名、统一配置管理。

### 4.2 generate_dflash_data.py：支持 NPU 和 target_model_type

文件：

```text
tools/generate_dflash_data.py
```

新增参数：

```python
parser.add_argument("--target_model_type", type=str, default="qwen3_moe")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--target_device_map", type=str, default=None)
parser.add_argument("--target_attn_implementation", type=str, default="eager")
```

其中：

```text
--target_device_map auto
```

用于让单个实例内的 target model 尝试切到两张可见 NPU 上。

```text
--target_attn_implementation eager
```

用于避免原始默认的 CUDA `flash_attention_2` 路径。

注意：`target_model_type` 不要传给 `DatasetManager`。当前 LLM dataset builder 只注册了：

```text
("online", "LLM", None)
```

如果把 `target_model_type="qwen3_moe"` 传给 `DatasetManager(..., target_model_type=...)`，会找不到 dataset builder。它只应传给 `create_target_model()`，或者仅作为日志和配置一致性字段。

### 4.3 generate_dflash_data.py：修改设备初始化

原始代码是：

```python
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
```

建议改成：

```python
def is_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


def init_distributed():
    local_rank = get_local_rank()
    world_size = get_world_size()

    if is_npu_available():
        torch.npu.set_device(local_rank)
        backend = "hccl"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)


def get_current_device(local_rank: int):
    if is_npu_available():
        return f"npu:{local_rank}"
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"
```

主函数中：

```python
local_rank = get_local_rank()
device = get_current_device(local_rank)
```

后续把：

```python
input_ids = input_ids.to(f"cuda:{local_rank}")
attention_mask = attention_mask.to(f"cuda:{local_rank}")
```

改成：

```python
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
```

如果 target 使用 `device_map="auto"`，上面这个 `device` 还不够严谨。更稳妥的做法是 target model 加载后取 embedding 层所在设备：

```python
def get_target_input_device(target_model):
    embeddings = target_model.model.get_input_embeddings()
    return next(embeddings.parameters()).device
```

然后在主循环中：

```python
target_input_device = get_target_input_device(target_model)

input_ids = input_ids.to(target_input_device)
attention_mask = attention_mask.to(target_input_device)
```

原因是 `device_map=auto` 下 embedding 层不一定等于 `local_rank` 对应设备；输入应该进入 embedding 层所在设备，由 HF/Accelerate dispatch hook 继续把中间激活搬到后续 layer 所在设备。

### 4.4 generate_dflash_data.py：创建 target model 时传入切分参数

原始：

```python
target_model = create_target_model(
    backend=args.target_backend,
    model_path=args.target_model_name_or_path,
    modal_type="LLM",
    torch_dtype=torch_dtype,
    trust_remote_code=args.trust_remote_code,
)
```

建议改成：

```python
extra_model_kwargs = {
    "attn_implementation": args.target_attn_implementation,
}

if args.target_device_map:
    extra_model_kwargs["device_map"] = args.target_device_map

target_model = create_target_model(
    backend=args.target_backend,
    model_path=args.target_model_name_or_path,
    modal_type="LLM",
    target_model_type=args.target_model_type,
    torch_dtype=torch_dtype,
    trust_remote_code=args.trust_remote_code,
    **extra_model_kwargs,
)
```

说明：

```text
create_target_model 会把 extra kwargs 传给 TransformersBackend。
TransformersBackend._prepare_model_kwargs 里 default_kwargs.update(filtered) 会覆盖默认 device_map 和 attn_implementation。
这一步可以通过 `generate_dflash_data.py` 把 `device_map="auto"` 和 `attn_implementation="eager"` 传入 target wrapper；但这还不够，下一节仍需要修改 `target_model_wrapper.py` 中 hidden states 拼接逻辑，否则跨设备 layer hidden 可能无法直接 `torch.cat()`。
```

### 4.5 target_model_wrapper.py：处理跨设备 hidden_states 拼接

文件：

```text
angelslim/compressor/speculative/train/models/target/target_model_wrapper.py
```

当前 `_extract_auxiliary_hidden_states()` 逻辑是：

```python
selected_hiddens = [hidden_states[layer_id + embed_offset] for layer_id in aux_layer_ids]
return torch.cat(selected_hiddens, dim=-1)
```

这在单卡或每 rank 一份完整模型时没问题。但在 `device_map=auto` 两卡切一个模型时，`outputs.hidden_states` 中不同 layer 的 tensor 可能落在不同 NPU 上，直接 `torch.cat()` 会报类似：

```text
Expected all tensors to be on the same device
```

建议改成：

```python
selected_hiddens = [hidden_states[layer_id + embed_offset] for layer_id in aux_layer_ids]

concat_device = selected_hiddens[0].device
selected_hiddens = [h.to(concat_device) for h in selected_hiddens]

return torch.cat(selected_hiddens, dim=-1)
```

这一步会带来少量跨卡拷贝，但 DFlash hidden cache 生成是离线预处理，正确性优先。

### 4.6 generate_dflash_data.py：打印模型切分结果

为了确认模型真的布在两张 NPU 上，target model 加载后建议打印：

```python
hf_device_map = getattr(target_model.model, "hf_device_map", None)
if hf_device_map is not None:
    rank0_print(f"Target model hf_device_map: {hf_device_map}")
else:
    rank0_print("Target model has no hf_device_map; check whether device_map took effect.")
```

验收标准：

```text
hf_device_map 中应该同时出现 npu:0 和 npu:1，或者至少显示不同 layer 被分配到两张可见设备。
```

如果所有 layer 都在 `npu:0`，说明没有实现“2 张卡布一个模型”，不要继续全量生成。

### 4.7 generate_dflash_data.py：DataLoader 参数化

原始代码写死：

```python
num_workers=4
pin_memory=True
```

建议改成：

```python
dataloader = DataLoader(
    online_train_dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    num_workers=args.num_workers,
    pin_memory=device.startswith("cuda"),
    collate_fn=collate_fn,
)
```

NPU 上不建议开 CUDA 的 `pin_memory=True`。

### 4.8 generate_dflash_data.py：可选避免保存/搬运 logits

当前 `get_hidden_states_and_logits()` 会返回 logits，生成脚本虽然只保存 hidden states，但仍然计算并把 logits 搬回 `input_ids.device`：

```python
return hidden_states, logits.to(input_ids.device)
```

Qwen3-30B-A3B 的 vocab 较大，长序列下 logits 占用很高。这不是正确性问题，但会增加显存和耗时。生成 hidden cache 时可以新增 `return_logits=False` 分支，或者新增一个 hidden-only 方法，只返回：

```python
hidden_states
```

如果不改，smoke test 也能验证链路，但全量生成时吞吐和显存压力会更大。

### 4.9 generate_dflash_data.sh：改成环境变量驱动

文件：

```text
scripts/speculative/generate_dflash_data.sh
```

原始脚本中这些变量是空值或固定值：

```bash
NUM_GPUS=${1:-8}
TARGET_MODEL_PATH=""
TRAIN_DATA_PATH=""
OUTPUT_DIR="${ROOT_DIR}/outputs/"
--draft_model_config_path $ROOT_DIR/configs/qwen3_dflash.json
--sample_num 128
```

建议改成：

```bash
NUM_GPUS=${NUM_GPUS:-${1:-1}}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:?TARGET_MODEL_PATH is required}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:?TRAIN_DATA_PATH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-"${ROOT_DIR}/configs/qwen3_30b_a3b_dflash_npu.json"}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-qwen3}
TARGET_MODEL_TYPE=${TARGET_MODEL_TYPE:-qwen3_moe}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_PROC=${NUM_PROC:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
SHARD_SIZE=${SHARD_SIZE:-10000}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
TARGET_DEVICE_MAP=${TARGET_DEVICE_MAP:-auto}
TARGET_ATTN_IMPLEMENTATION=${TARGET_ATTN_IMPLEMENTATION:-eager}

SAMPLE_NUM_ARGS=()
if [[ -n "${SAMPLE_NUM:-}" ]]; then
    SAMPLE_NUM_ARGS=(--sample_num "$SAMPLE_NUM")
fi
```

torchrun 参数中增加：

```bash
--target_model_type $TARGET_MODEL_TYPE \
--target_device_map $TARGET_DEVICE_MAP \
--target_attn_implementation $TARGET_ATTN_IMPLEMENTATION \
--num_workers $NUM_WORKERS \
"${SAMPLE_NUM_ARGS[@]}" \
```

并把 config 改成：

```bash
--draft_model_config_path $DRAFT_MODEL_CONFIG_PATH
```

不要继续写死：

```bash
--sample_num 128
```

否则全量生成时只会生成 128 条。

## 5. 数据拆成 4 份

建议新增一个简单脚本：

```text
tools/split_jsonl.py
```

逻辑：

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = [
        open(output_dir / f"shard_{i:02d}.jsonl", "w", encoding="utf-8")
        for i in range(args.num_shards)
    ]
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                writers[idx % args.num_shards].write(line)
    finally:
        for w in writers:
            w.close()


if __name__ == "__main__":
    main()
```

执行：

```bash
python tools/split_jsonl.py \
  --input /data/qwen3_30b_train.jsonl \
  --output_dir /data/qwen3_30b_train_shards \
  --num_shards 4
```

得到：

```text
/data/qwen3_30b_train_shards/shard_00.jsonl
/data/qwen3_30b_train_shards/shard_01.jsonl
/data/qwen3_30b_train_shards/shard_02.jsonl
/data/qwen3_30b_train_shards/shard_03.jsonl
```

## 6. 四实例启动脚本

建议新增：

```text
scripts/speculative/generate_qwen3_30b_hidden_2npu_x4.sh
```

内容：

```bash
#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:?TARGET_MODEL_PATH is required}
SHARD_DIR=${SHARD_DIR:?SHARD_DIR is required}
CACHE_DIR=${CACHE_DIR:?CACHE_DIR is required}

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

export DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-"$ROOT_DIR/configs/qwen3_30b_a3b_dflash_npu.json"}
export MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
export BATCH_SIZE=${BATCH_SIZE:-1}
export NUM_PROC=${NUM_PROC:-16}
export NUM_WORKERS=${NUM_WORKERS:-4}
export SHARD_SIZE=${SHARD_SIZE:-10000}
export TARGET_MODEL_TYPE=${TARGET_MODEL_TYPE:-qwen3_moe}
export CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-qwen3}
export TARGET_DEVICE_MAP=${TARGET_DEVICE_MAP:-auto}
export TARGET_ATTN_IMPLEMENTATION=${TARGET_ATTN_IMPLEMENTATION:-eager}
export NUM_GPUS=1

mkdir -p "$CACHE_DIR" "$CACHE_DIR/logs"

PAIRS=("0,1" "2,3" "4,5" "6,7")

for i in 0 1 2 3; do
  export ASCEND_RT_VISIBLE_DEVICES=${PAIRS[$i]}
  export TRAIN_DATA_PATH="$SHARD_DIR/shard_0${i}.jsonl"
  export OUTPUT_DIR="$CACHE_DIR/instance_${i}"

  echo "[INFO] launch instance_${i}: devices=${ASCEND_RT_VISIBLE_DEVICES}, data=${TRAIN_DATA_PATH}, output=${OUTPUT_DIR}"

  bash scripts/speculative/generate_dflash_data.sh \
    > "$CACHE_DIR/logs/instance_${i}.log" 2>&1 &
done

wait
echo "[INFO] all hidden_states generation instances finished"
```

启动：

```bash
export TARGET_MODEL_PATH=/models/Qwen3-30B-A3B
export SHARD_DIR=/data/qwen3_30b_train_shards
export CACHE_DIR=/data/qwen3_30b_dflash_hidden_cache

bash scripts/speculative/generate_qwen3_30b_hidden_2npu_x4.sh
```

## 7. smoke test 流程

先不要跑全量。建议每个 shard 只取少量样本，或设置：

```bash
export SAMPLE_NUM=2
```

然后启动：

```bash
bash scripts/speculative/generate_qwen3_30b_hidden_2npu_x4.sh
```

检查每个实例日志：

```bash
tail -n 100 /data/qwen3_30b_dflash_hidden_cache/logs/instance_0.log
tail -n 100 /data/qwen3_30b_dflash_hidden_cache/logs/instance_1.log
tail -n 100 /data/qwen3_30b_dflash_hidden_cache/logs/instance_2.log
tail -n 100 /data/qwen3_30b_dflash_hidden_cache/logs/instance_3.log
```

重点看：

```text
Target model loaded successfully
Target model hf_device_map
Data generation complete
```

如果 `hf_device_map` 没有显示模型被放到两张可见 NPU 上，需要先解决模型切分问题。

## 8. 全量生成流程

smoke test 通过后：

```bash
unset SAMPLE_NUM
export MODEL_MAX_LENGTH=4096
export NUM_PROC=16
export NUM_WORKERS=4

bash scripts/speculative/generate_qwen3_30b_hidden_2npu_x4.sh
```

此时 4 个实例会同时运行：

```text
instance_0 -> NPU 0,1 -> shard_00
instance_1 -> NPU 2,3 -> shard_01
instance_2 -> NPU 4,5 -> shard_02
instance_3 -> NPU 6,7 -> shard_03
```

## 9. hidden cache 验证

随机检查：

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
target_layer_num = 5
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

期望：

```text
hidden_states.shape[-1] = 10240
```

如果你修改了 `target_layer_ids` 数量，需要同步修改 `target_layer_num`。

## 10. 常见问题

### 10.1 device_map=auto 没有切到两张 NPU

表现：

```text
hf_device_map 为空，或者所有 layer 都在 npu:0
```

处理：

1. 确认每个实例只看到两张卡：

```bash
echo $ASCEND_RT_VISIBLE_DEVICES
```

2. 确认传入了：

```bash
--target_device_map auto
```

3. 确认 transformers / accelerate / torch-npu 版本支持当前 NPU device map。Ascend NPU 上 `device_map=auto`  historically 有兼容性问题，需要以本地环境实测为准。

### 10.2 误用 NUM_GPUS=2

不要这样跑：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/speculative/generate_dflash_data.sh
```

这会启动两个 rank，通常是两份 target model，而不是两卡一份 target model。

正确方式：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 NUM_GPUS=1 TARGET_DEVICE_MAP=auto bash scripts/speculative/generate_dflash_data.sh
```

### 10.3 原始脚本只生成 128 条

原始脚本写死：

```bash
--sample_num 128
```

全量生成前必须改成可选参数。smoke test 时显式设置 `SAMPLE_NUM=2` 或 `SAMPLE_NUM=10`，全量时 `unset SAMPLE_NUM`。

### 10.4 flash_attention_2 报错

原始 target wrapper 默认：

```python
attn_implementation = "flash_attention_2"
```

910B 上建议从：

```bash
TARGET_ATTN_IMPLEMENTATION=eager
```

开始验证。若本地 torch-npu / transformers 组合确认支持，再尝试 `sdpa`。

### 10.5 OOM

处理顺序：

1. 降低 `MODEL_MAX_LENGTH`，例如从 `4096` 降到 `2048`。
2. 保持 `BATCH_SIZE=1`。
3. 确认 `device_map=auto` 真的把模型放到了两张 NPU。
4. 检查是否误启动了 `NUM_GPUS=2` 或更多 rank。
5. 检查是否四个实例的 `ASCEND_RT_VISIBLE_DEVICES` 有重叠。

## 11. 后续离线训练接入

hidden cache 根目录：

```text
/data/qwen3_30b_dflash_hidden_cache
```

下面包含四个实例的输出：

```text
instance_0/
instance_1/
instance_2/
instance_3/
```

离线训练时让 offline dataset 递归读取该根目录下的 `.ckpt`。如果当前 offline dataset 只支持读取单层目录，则需要把四个 instance 目录下的 `.ckpt` 汇总成统一目录，或修改 offline dataset 的 glob 为递归：

```python
glob.glob(os.path.join(hidden_cache_path, "**", "*.ckpt"), recursive=True)
```

## 12. 参考说明

- Qwen3-30B-A3B 是 `qwen3_moe`，需要 transformers 版本支持该模型类型。
- Hugging Face Accelerate Big Model Inference 支持通过 `device_map="auto"` 自动放置大模型权重，也可以通过模型的 `hf_device_map` 查看最终放置结果。
- Transformers `from_pretrained(..., device_map="auto")` 会接入 Accelerate 的大模型加载能力。
- Ascend 迁移文档中，CUDA 设备设置接口需要替换为 NPU 对应接口，例如 `torch.cuda.set_device()` 对应 `torch_npu.npu.set_device()` / `torch.npu.set_device()`。
- Hugging Face / Accelerate 的 `device_map=auto` 在 Ascend NPU 上需要以本地 transformers / accelerate / torch-npu 版本实测为准，必须通过 `hf_device_map` 和实际显存占用确认。

参考链接：

- [Hugging Face Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/v1.9.0/usage_guides/big_modeling)
- [Hugging Face Accelerate: Handling big models for inference](https://huggingface.co/docs/accelerate/v0.34.1/en/concept_guides/big_model_inference)
- [Transformers Big Models](https://huggingface.co/docs/transformers/main/big_models)
- [Ascend Extension for PyTorch 迁移调优指南](https://www.hiascend.com/doc_center/source/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/Ascend%20Extension%20for%20PyTorch%206.0.RC3%20%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E8%BF%81%E7%A7%BB%E8%B0%83%E4%BC%98%E6%8C%87%E5%8D%97%2001.pdf)
