# Qwen3.6-35B-A3B DFlash on Ascend 910B Adaptation Guide

本文档面向原始 `Tencent/AngelSlim` 工程，说明如何适配 `Qwen/Qwen3.6-35B-A3B` 的 DFlash 离线训练链路。目标环境为单机 8 张昇腾 910B 64G NPU。

> 结论先行：不要把它当作旧的 `qwen3_vl` 或纯 LLM 来适配。Hugging Face 官方配置中该模型的顶层 `model_type` 是 `qwen3_5_moe`，架构是 `Qwen3_5MoeForConditionalGeneration`，是带视觉编码器的 MoE 视觉语言模型。AngelSlim 当前 DFlash 脚本把 target 和 dataset 都写成 LLM 路径，需要补齐 VLM cache 生成、`qwen3_5_moe` target 类型、M-RoPE position 处理和 910B/HCCL/sdpa 运行适配。

## 1. 官方模型关键信息

模型地址：

- Hugging Face: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Config: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json>

官方模型卡说明该模型是 `Causal Language Model with Vision Encoder`，35B 总参数、约 3B 激活参数、支持图像输入，推荐最新 Transformers/vLLM/SGLang 等推理框架。

从官方 `config.json` 读取到的关键字段如下：

```json
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "vision_start_token_id": 248053,
  "vision_end_token_id": 248054,
  "image_token_id": 248056,
  "video_token_id": 248057,
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    }
  },
  "vision_config": {
    "hidden_size": 1152,
    "depth": 27,
    "num_heads": 16,
    "patch_size": 16,
    "spatial_merge_size": 2
  }
}
```

生成配置：

```json
{
  "bos_token_id": 248044,
  "eos_token_id": [248046, 248044],
  "pad_token_id": 248044,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95
}
```

图像预处理配置：

```json
{
  "processor_class": "Qwen3VLProcessor",
  "image_processor_type": "Qwen2VLImageProcessorFast",
  "size": {
    "longest_edge": 16777216,
    "shortest_edge": 65536
  },
  "patch_size": 16,
  "temporal_patch_size": 2,
  "merge_size": 2
}
```

## 2. AngelSlim 当前可复用部分

原工程已经有以下 VLM 能力，可以复用：

```text
angelslim/compressor/speculative/train/data/dataset_builder/online_dataset_builder.py
  - OnlineVLMDatasetBuilder
  - qwen3_vl / qwen2_5_vl 数据处理
  - 图像路径收集、chat template、loss_mask 构造

angelslim/compressor/speculative/train/data/data_utils.py
  - VLMDataCollatorWithPadding
  - image_paths -> pixel_values / image_grid_thw
  - inputs_embeds / position_ids padding

angelslim/compressor/speculative/train/models/target/target_model_wrapper.py
  - VLMTransformersBackend
  - AutoModelForImageTextToText
  - AutoProcessor
  - get_aux_and_target_hiddens 返回 hidden_states / inputs_embeds / position_ids

angelslim/compressor/speculative/train/models/model_utils.py
  - qwen3_vl 映射
  - apply_rotary_pos_emb_mrope

tools/train_dflash_offline.py
angelslim/compressor/speculative/train/trainer/offline_dflash_trainer.py
angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
angelslim/compressor/speculative/train/models/draft/qwen_dflash.py
  - DFlash 离线训练主体逻辑
```

当前缺口：

- `tools/generate_dflash_data.py` 写死 `modal_type="LLM"`，并写死 CUDA/NCCL。
- `tools/train_dflash_online.py` 写死 DFlash target 使用 `modal_type="LLM"`，不适合 VLM。
- `VLMTransformersBackend.SUPPORT_MODEL_TYPE` 没有 `qwen3_5_moe`。
- `MODEL_TYPE_PARAM_MAP` 没有 `qwen3_5_moe / qwen3_5_moe_text`。
- DFlash offline cache 目前只要求 `input_ids / hidden_states / loss_mask`，没有把 VLM `position_ids` 传到训练 loss。
- `QwenDFlashDraftModel` 当前按 Qwen3 普通 RoPE 写，未完整适配 Qwen3.6 的 `rope_parameters.mrope_interleaved` 与 `partial_rotary_factor`。

## 3. 推荐适配策略

只做离线训练，不建议先做在线训练。流程如下：

```text
VLM JSONL 数据
  -> AutoProcessor + OnlineVLMDatasetBuilder
  -> Qwen3.6-35B-A3B target forward
  -> 保存 DFlash VLM hidden cache
  -> train_dflash_offline.py 训练 DFlash head
  -> checkpoint 加载和 VLM speculative decoding 验证
```

离线 cache 生成阶段需要完整 target VLM。离线训练阶段不再跑完整 target，只训练 draft head。

## 4. 必须修改 1：支持 qwen3_5_moe 类型

文件：

```text
angelslim/compressor/speculative/train/models/target/target_model_wrapper.py
```

将 `VLMTransformersBackend` 的支持列表扩展：

```python
SUPPORT_MODEL_TYPE = ["hunyuan_vl", "qwen3_vl", "qwen3_5_moe", "qwen2_5_vl"]
```

把所有：

```python
self.target_model_type in ("qwen3_vl", "qwen2_5_vl")
```

改为：

```python
self.target_model_type in ("qwen3_vl", "qwen3_5_moe", "qwen2_5_vl")
```

文件：

```text
angelslim/compressor/speculative/train/models/model_utils.py
```

在 `MODEL_TYPE_PARAM_MAP` 中增加：

```python
"qwen3_5_moe": (
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "qwen3_vl",
),
"qwen3_5_moe_text": (
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "qwen3_vl",
),
```

说明：

- 官方顶层 `model_type` 是 `qwen3_5_moe`。
- 文本子配置 `text_config.model_type` 是 `qwen3_5_moe_text`。
- `embed_weight_key` 应优先使用 `model.language_model.embed_tokens.weight`。
- `lm_head_key` 需要用本地 checkpoint 的 `model.safetensors.index.json` 最终确认；若不存在 `lm_head.weight`，需要按实际 key 修正。

## 5. 必须修改 2：新增 VLM DFlash cache 生成脚本

建议新增文件：

```text
tools/generate_dflash_vlm_data.py
```

不要直接覆盖 `generate_dflash_data.py`，避免破坏现有 LLM DFlash 路径。

核心差异如下。

### 5.1 参数新增

```python
parser.add_argument("--target_model_type", type=str, default="qwen3_5_moe")
parser.add_argument("--max_pixels", type=int, default=None)
parser.add_argument("--min_pixels", type=int, default=1024)
parser.add_argument("--device_map", type=str, default=None)
```

### 5.2 使用 VLM target backend

```python
target_model = create_target_model(
    backend=args.target_backend,
    model_path=args.target_model_name_or_path,
    modal_type="VLM",
    torch_dtype=torch_dtype,
    trust_remote_code=args.trust_remote_code,
    target_model_type=args.target_model_type,
)
```

### 5.3 使用 AutoProcessor 与 VLM dataset builder

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    args.target_model_name_or_path,
    trust_remote_code=True,
)

args.modal_type = "VLM"
args.training_mode = "online"
args.target_model_type = args.target_model_type

dataset_manager = DatasetManager(
    data_args=args,
    tokenizer=processor,
    model_max_length=args.model_max_length,
    chat_template_type="qwen3_vl",
    target_model_type=args.target_model_type,
)
```

### 5.4 forward 时传入 VLM 字段

```python
outputs = target_model.get_aux_and_target_hiddens(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=batch.get("pixel_values"),
    image_grid_thw=batch.get("image_grid_thw"),
    pixel_values_videos=batch.get("pixel_values_videos"),
    video_grid_thw=batch.get("video_grid_thw"),
    input_position_ids=batch.get("input_position_ids"),
    aux_hidden_states_layer_ids=target_layer_ids,
)
```

### 5.5 cache 保存字段

VLM DFlash cache 建议保存：

```python
ckpt = {
    "input_ids": input_ids[i : i + 1].cpu(),
    "attention_mask": attention_mask[i : i + 1].cpu(),
    "loss_mask": loss_mask[i : i + 1].cpu(),
    "hidden_states": outputs["hidden_states"][i : i + 1].cpu().to(torch.bfloat16),
}

if outputs.get("position_ids") is not None:
    ckpt["position_ids"] = outputs["position_ids"][:, i : i + 1, :].cpu()

if outputs.get("inputs_embeds") is not None:
    ckpt["inputs_embeds"] = outputs["inputs_embeds"][i : i + 1].cpu().to(torch.bfloat16)
```

注意：

- `hidden_states` 是多层 target hidden concat，形状应为 `[1, S, 2048 * len(target_layer_ids)]`。
- `position_ids` 对 Qwen3.6 很重要，因为官方 `rope_parameters` 是 M-RoPE。
- `inputs_embeds` 当前 DFlash loss 不一定直接使用，但建议保存，便于后续对齐和 debug。

## 6. 必须修改 3：NPU/HCCL 设备适配

原始 `generate_dflash_data.py` 写死：

```python
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
input_ids = input_ids.to(f"cuda:{local_rank}")
```

910B 需要改成：

```python
def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        backend = "hccl"
        device = f"npu:{local_rank}"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"

    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return device
```

训练和 cache 生成脚本里统一使用：

```python
device = init_distributed()
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
```

环境变量建议：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
export ATTENTION_BACKEND=sdpa
```

不要在 910B 上默认使用 CUDA 的 `flash_attention_2` 或 `flex_attention`。

## 7. 必须修改 4：DFlash offline dataset 读取 VLM position_ids

文件：

```text
tools/train_dflash_offline.py
```

`OfflineDFlashDataset` 当前只要求：

```python
REQUIRED_KEYS = ["input_ids", "hidden_states", "loss_mask"]
```

保持 required 不变，但增加 optional 校验：

```python
OPTIONAL_KEYS = ["attention_mask", "position_ids", "inputs_embeds"]
```

`DataCollatorWithPadding` 当前不会 padding `position_ids / inputs_embeds`。有两种选择：

### 推荐方式

新增一个 `DFlashVLMDataCollatorWithPadding`，支持：

- `input_ids`
- `attention_mask`
- `loss_mask`
- `hidden_states`
- `position_ids`
- `inputs_embeds`

其中 `position_ids` 可能是 `[3, 1, S]` 或 `[1, S]`，需要按实际 cache shape 分支处理。

### 快速方式

复用 `VLMDataCollatorWithPadding` 的 offline padding 逻辑，但要避免重新从 `image_paths` 读取图片。

## 8. 必须修改 5：OfflineDFlashTrainer 传递 position_ids

文件：

```text
angelslim/compressor/speculative/train/trainer/offline_dflash_trainer.py
```

改为：

```python
def prepare_data_for_draft_model(self, inputs):
    return {
        "input_ids": inputs["input_ids"],
        "hidden_states": inputs["hidden_states"],
        "loss_mask": inputs["loss_mask"],
        "attention_mask": inputs["attention_mask"],
        "position_ids": inputs.get("position_ids", None),
        "inputs_embeds": inputs.get("inputs_embeds", None),
    }
```

文件：

```text
angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py
```

将 `_compute_dflash_loss_and_accuracy` 签名扩展为：

```python
def _compute_dflash_loss_and_accuracy(
    self,
    model,
    input_ids,
    hidden_states,
    loss_mask,
    position_ids=None,
):
```

在 `compute_loss` 和 `prediction_step` 中传入：

```python
position_ids=data.get("position_ids", None)
```

对于 VLM，如果 `position_ids` 存在，不应再简单使用：

```python
context_position_ids = torch.arange(seq_len, device=device)
```

而应优先使用 cache 中的 context position。

## 9. 必须修改 6：QwenDFlashDraftModel 支持 Qwen3.6 M-RoPE

文件：

```text
angelslim/compressor/speculative/train/models/draft/qwen_dflash.py
```

当前 `Qwen3DFlashAttention` 使用本文件内的普通 `apply_rotary_pos_emb`。Qwen3.6 的官方配置是：

```json
"rope_parameters": {
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "partial_rotary_factor": 0.25,
  "rope_theta": 10000000,
  "rope_type": "default"
}
```

适配要求：

1. draft config 中保留 `rope_parameters`。
2. `Qwen3DFlashAttention` 根据 `config.rope_parameters.mrope_interleaved` 判断是否走 M-RoPE。
3. 引入 `model_utils.apply_rotary_pos_emb_mrope`。
4. 处理 `position_ids` 为 `[3, B, S]` 的情况。
5. 注意 `partial_rotary_factor=0.25`，不要假设整个 `head_dim` 都参与 RoPE。

最低可用实现路线：

- 先让 VLM cache 生成和 offline 训练跑通。
- 使用 `attention_backend=sdpa`。
- `position_ids` 从 target forward hook 保存。
- draft forward 中使用同一套 position 语义。

更严格的实现路线：

- 新增 `Qwen36DFlashDraftModel`，不要复用 `Qwen3PreTrainedModel`。
- 参考 Transformers 的 `Qwen3_5MoeForConditionalGeneration` 文本模块实现 rotary embedding 和 attention。
- DFlash draft 仍保持小模型层数，但 RoPE、head_dim、position 语义与 Qwen3.6 text_config 对齐。

## 10. Draft config 模板

建议新增：

```text
configs/qwen3_6_35b_a3b_dflash_npu.json
```

初始模板：

```json
{
  "architectures": ["QwenDFlashDraftModel"],
  "model_type": "qwen3",
  "target_model_type": "qwen3_5_moe",
  "attention_bias": false,
  "attention_dropout": 0.0,
  "block_size": 16,
  "bos_token_id": 248044,
  "eos_token_id": 248044,
  "pad_token_id": 248044,
  "dflash_config": {
    "mask_token_id": 248058,
    "target_layer_ids": [1, 10, 20, 30, 39]
  },
  "dtype": "bfloat16",
  "torch_dtype": "bfloat16",
  "hidden_act": "silu",
  "hidden_size": 2048,
  "head_dim": 256,
  "initializer_range": 0.02,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "num_hidden_layers": 5,
  "num_target_layers": 40,
  "intermediate_size": 6144,
  "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000000,
  "rope_parameters": {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "use_cache": true,
  "vocab_size": 248320,
  "tie_word_embeddings": false,
  "image_token_id": 248056,
  "video_token_id": 248057,
  "vision_start_token_id": 248053,
  "vision_end_token_id": 248054,
  "lm_head_key": "lm_head.weight",
  "embed_weight_key": "model.language_model.embed_tokens.weight",
  "num_anchors": 64,
  "loss_decay_gamma": 7.0,
  "attention_backend": "sdpa"
}
```

注意事项：

- `mask_token_id` 必须从 tokenizer special tokens 实际确认，上面的 `248058` 只是占位示例，不能盲用。
- `target_layer_ids` 建议先用 `[1, 10, 20, 30, 39]` 做 smoke test，后续可按 loss/接受率调整。
- `model_type` 暂时保留 `qwen3` 是为了让现有 `QwenDFlashDraftModel` 能用 `Qwen3Config` 解析；如果实现了专用 `Qwen36DFlashDraftModel`，再改成新 config class。

## 11. 35B-A3B 在 8x64G 910B 上的加载策略

cache 生成阶段要跑完整 target VLM，不能简单照搬 LLM DFlash 的：

```bash
torchrun --nproc_per_node=8 tools/generate_dflash_data.py
```

因为这会让每个 rank 都加载一份完整 35B+vision target。即使 MoE 每 token 激活约 3B，权重本身仍是 35B 级别，bf16 权重大约 70GB 以上，加上视觉模块和运行时缓存，单张 64G NPU 很容易 OOM。

推荐优先级：

1. **优先方案：单进程跨 8 NPU 切分 target model 生成 cache**
   - 使用 Ascend/Transformers/Accelerate 支持的 device map 或模型并行能力。
   - batch size 从 1 开始。
   - 优先保证 cache 正确性。

2. **备选方案：少进程，每进程绑定一组 NPU**
   - 例如 2 个进程，每个进程 4 张 NPU。
   - 每个进程处理不同数据 shard。

3. **不建议：8 rank 每 rank 一份完整 target**
   - 高概率 OOM。
   - 即使能启动，显存和吞吐也不可控。

离线训练阶段不需要完整 target forward，只训练 DFlash draft head，可以使用 8 卡 FSDP。

## 12. 数据格式

JSONL 每行建议：

```json
{
  "id": "sample_000001",
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/abs/path/image.jpg"},
        {"type": "text", "text": "请描述图中的主要内容。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "图中展示了..."}
      ]
    }
  ]
}
```

检查要求：

- 图像路径必须在 cache 生成机器可访问。
- 视觉 token 区域 `loss_mask=0`。
- assistant 文本 token 区域 `loss_mask=1`。
- `input_ids` 长度不要一开始就拉满，先用 `model_max_length=2048` 做 smoke test。

## 13. 推荐执行流程

### 13.1 环境

```bash
cd /path/to/AngelSlim

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ANGELSLIM_TARGET_ATTN_IMPL=sdpa
export ATTENTION_BACKEND=sdpa

export TARGET_MODEL_PATH=/models/Qwen3.6-35B-A3B
export TRAIN_DATA_PATH=/data/qwen36_vlm_train.jsonl
export DRAFT_CONFIG_PATH=configs/qwen3_6_35b_a3b_dflash_npu.json
export CACHE_DIR=/data/qwen36_dflash_hidden_cache
export OUTPUT_DIR=/output/qwen36_dflash_head
```

### 13.2 cache 生成 smoke test

先跑 10 到 100 条数据：

```bash
python tools/generate_dflash_vlm_data.py \
  --target_model_name_or_path "$TARGET_MODEL_PATH" \
  --draft_model_config_path "$DRAFT_CONFIG_PATH" \
  --target_model_type qwen3_5_moe \
  --train_data_path "$TRAIN_DATA_PATH" \
  --output_dir "$CACHE_DIR/smoke" \
  --chat_template_type qwen3_vl \
  --model_max_length 2048 \
  --sample_num 100 \
  --batch_size 1 \
  --torch_dtype bfloat16
```

检查每个 `.ckpt`：

```python
import torch
p = "/data/qwen36_dflash_hidden_cache/smoke/sample_00000000.ckpt"
x = torch.load(p, map_location="cpu")
for k, v in x.items():
    print(k, None if v is None else (v.shape, v.dtype))
```

期望：

```text
input_ids      [1, S] int64
attention_mask [1, S]
loss_mask      [1, S]
hidden_states  [1, S, 2048 * len(target_layer_ids)] bf16
position_ids   optional, usually related to M-RoPE
inputs_embeds  optional [1, S, 2048] bf16
```

### 13.3 离线训练 smoke test

```bash
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

### 13.4 扩大训练

smoke test 通过后再逐步扩大：

```text
model_max_length: 2048 -> 3072 -> 4096
num_anchors:      32   -> 64   -> 128
sample_num:       100  -> 10k  -> full
```

## 14. 验证清单

cache 生成验证：

- target forward 能在 910B 上稳定跑。
- `hidden_states.shape[-1] == 2048 * len(target_layer_ids)`。
- `loss_mask.sum() > 0`。
- 视觉 token 不参与 loss。
- `position_ids` 与 target VLM forward hook 捕获一致。

训练验证：

- 首 100 step loss 非 NaN。
- accuracy 非长期为 0。
- FSDP 不因最后 batch shape 不一致报错。
- checkpoint 能恢复。
- 保存后的 draft config 与 tokenizer/vocab 对齐。

效果验证：

- 加载 target VLM + DFlash draft head。
- 图文 prompt 下做 speculative decoding。
- 统计 acceptance length、acceptance rate、tokens/s。
- 对比不开 DFlash 的原始 target 输出质量。
- 对视觉问答、OCR、图表理解、纯文本问题分别评估。

## 15. 主要风险和处理建议

### 风险 1：target cache 生成 OOM

原因：35B target 不适合 8 rank 全副本加载。

处理：

- 改成单进程跨 8 NPU 切分 target。
- 降低 `MAX_PIXELS`。
- 降低 `model_max_length`。
- batch size 固定为 1。

### 风险 2：M-RoPE 没对齐

现象：

- 训练 loss 能下降，但 speculative acceptance 很低。
- 图文任务质量明显退化。

处理：

- cache 保存 target 捕获到的 `position_ids`。
- DFlash draft 使用 Qwen3.6 的 `rope_parameters`。
- 不要把 VLM position 简化成普通 `arange`。

### 风险 3：权重 key 不一致

现象：

- `TargetEmbeddingsAndHead.from_pretrained` 找不到 key。
- 或 logits 投影异常。

处理：

- 检查 `model.safetensors.index.json`。
- 确认：
  - `lm_head.weight`
  - `model.language_model.embed_tokens.weight`

### 风险 4：token id 错误

现象：

- mask token 不存在。
- 特殊 token 被当作普通 token 训练。

处理：

- 用 tokenizer 打印 special tokens。
- 确认 `mask_token_id`，不要直接沿用 Qwen3/Qwen3-VL 旧 id。

### 风险 5：Qwen3.6 text backbone 与 Qwen3 draft 差异

Qwen3.6 文本侧是 `qwen3_5_moe_text`，包含 Gated DeltaNet、Gated Attention、MoE 等结构；当前 `QwenDFlashDraftModel` 是基于 Qwen3 attention/MLP 的轻量 draft。短期可以作为 DFlash head 训练基线，但要把 RoPE、vocab、hidden size、lm head 对齐。若基线 acceptance 不理想，应实现更贴近 Qwen3.6 的专用 `Qwen36DFlashDraftModel`。

## 16. 推荐落地顺序

1. 加 `qwen3_5_moe` 类型映射和 VLM target backend 支持。
2. 新增 `configs/qwen3_6_35b_a3b_dflash_npu.json`。
3. 新增 `tools/generate_dflash_vlm_data.py`。
4. 完成 NPU/HCCL/sdpa 设备适配。
5. 生成 100 条 VLM cache，并检查 tensor schema。
6. 扩展 offline collator/trainer 读取 `position_ids`。
7. 跑 8 卡 offline smoke training。
8. 补 M-RoPE draft 对齐。
9. 全量 cache 生成和离线训练。
10. 做图文 speculative decoding 评估。

## 17. 最小交付标准

一版可接受的 910B 适配交付应满足：

- 能在 910B 上成功加载 Qwen3.6-35B-A3B target VLM 做 cache 生成。
- 能生成包含 `hidden_states / loss_mask / input_ids / attention_mask` 的 VLM DFlash cache。
- 能用 8 卡 FSDP 训练 DFlash draft head。
- 能恢复 checkpoint。
- 能加载 draft head 做至少图文输入的 speculative decoding smoke test。
- 有 acceptance rate、tokens/s、质量回归样例。

