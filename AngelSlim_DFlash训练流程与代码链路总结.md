# AngelSlim 中 DFlash 训练流程与代码链路总结

## 仓库位置

- 本地目录：`D:\workspace\speculative\AngelSlim`
- 下载方式：GitHub zipball 下载并解压。最初 `git clone` 因网络超时未完成，已清理失败的半成品目录。
- 关键配置：`AngelSlim/configs/qwen3_dflash.json`

## 一句话总览

AngelSlim 的 DFlash 训练是一个 block-parallel speculative draft model 训练流程：target model 提供多层 hidden states，DFlash draft model 在随机 anchor 位置上并行预测后续一个 block 的 token，用带位置衰减的 cross-entropy 训练 draft model，使其在推理时一次提出一个 block，再由 target model 并行验证最长正确前缀。

## 两条训练路线

### 1. Online training：推荐入口

入口脚本：

- `AngelSlim/scripts/speculative/run_dflash_online.sh`
- Python 入口：`AngelSlim/tools/train_dflash_online.py`

运行前需要设置：

```bash
export TARGET_MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_DATA_PATH=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/output

bash scripts/speculative/run_dflash_online.sh 8 flex_attention
```

流程：

1. 读取 draft config：`configs/qwen3_dflash.json`。
2. 加载 target model，target 使用 HuggingFace backend。
3. 根据 `dflash_config.target_layer_ids` 指定 target 需要导出的层。
4. 创建 `QwenDFlashDraftModel`。
5. 用 `DatasetManager` 将 conversation JSON/JSONL 转为 `input_ids / attention_mask / loss_mask`。
6. 创建 `OnlineDFlashTrainer`。
7. 每个训练 step 现场跑 target model，取多层 hidden states。
8. DFlash 在随机 anchor 上并行预测 block token，计算加权 CE loss。
9. 用 HF Trainer/FSDP/FP32 master optimizer 训练并保存 checkpoint。

主要默认参数：

- `block_size=16`
- `num_anchors=512`
- `loss_decay_gamma=7`
- `model_max_length=3072`
- `per_device_train_batch_size=2`
- `learning_rate=6e-4`
- `warmup_ratio=0.04`
- `num_train_epochs=6`
- `bf16`
- `fsdp="shard_grad_op auto_wrap"`
- `attention_backend=flex_attention`

### 2. Offline training：先缓存 hidden states，再训练

第一步生成缓存：

- Shell：`AngelSlim/scripts/speculative/generate_dflash_data.sh`
- Python：`AngelSlim/tools/generate_dflash_data.py`

```bash
export TARGET_MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_DATA_PATH=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/hidden_cache

bash scripts/speculative/generate_dflash_data.sh 8
```

生成的每个 `.ckpt` 样本包含：

- `input_ids`: `[1, S]`
- `hidden_states`: `[1, S, D * num_target_layers]`
- `loss_mask`: `[1, S]`
- `attention_mask`: `[1, S]`

第二步训练：

- Shell：`AngelSlim/scripts/speculative/run_dflash_offline.sh`
- Python：`AngelSlim/tools/train_dflash_offline.py`

```bash
export TARGET_MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_HIDDEN_PATH=/path/to/hidden_cache
export OUTPUT_DIR=/path/to/output

bash scripts/speculative/run_dflash_offline.sh 8 flex_attention
```

Offline 和 online 的训练 loss 逻辑相同；区别只是 hidden states 来源不同：

- online：`OnlineDFlashTrainer.prepare_data_for_draft_model()` 现场调用 target model；
- offline：`OfflineDFlashTrainer.prepare_data_for_draft_model()` 直接从 batch 取 `.ckpt` 中的 hidden states。

Offline 默认训练 epoch 是 12，online 默认是 6。

## 配置链路

核心配置文件是 `configs/qwen3_dflash.json`：

- `architectures=["QwenDFlashDraftModel"]`：由 factory 路由到 DFlash draft 模型。
- `block_size=16`：每次并行预测 16 个 token 位置，其中 anchor 位置本身不计 loss。
- `num_hidden_layers=5`：draft model 有 5 层。
- `num_target_layers=36`：target model 层数。
- `dflash_config.target_layer_ids=[1,9,17,25,33]`：从 target model 抽这 5 层 hidden states。
- `num_anchors=512`：每条样本最多采样 512 个 anchor。
- `loss_decay_gamma=7.0`：block 内越靠后的 token loss 权重按指数衰减。
- `attention_backend=flex_attention`：draft 训练 attention backend。

## 数据链路

训练数据是 conversation JSON/JSONL。基础格式兼容：

```json
{
  "id": "0",
  "conversations": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
  ]
}
```

也兼容 content 为 list 的多模态风格，但 DFlash LLM 训练会走 LLM tokenization 路径。

数据处理代码：

- `DatasetManager.create_online_datasets()`：选择 online dataset builder。
- DFlash 会额外设置 `min_loss_tokens = 2 * block_size`，过滤 assistant loss token 太少的样本。
- `OnlineLLMDatasetBuilder`/`OnlineDatasetBuilder`：apply chat template，tokenize，并生成 `loss_mask`。
- `loss_mask=1` 的 token 是 assistant response 区域，训练只在这些 token 上算 loss。
- `DataCollatorWithPadding`：按 batch 内最大长度 padding `input_ids / attention_mask / loss_mask`，offline 时也会拼接 `hidden_states`。

## 模型链路

### Draft model 创建

入口：

- `create_draft_model(draft_model_config)`
- `DraftModelFactory.from_config()`
- 根据 config 的 `architectures` 找到 `QwenDFlashDraftModel`

模型类：

- `AngelSlim/angelslim/compressor/speculative/train/models/draft/qwen_dflash.py`

关键结构：

- `QwenDFlashDraftModel`
- `Qwen3DFlashDecoderLayer`
- `Qwen3DFlashAttention`

DFlash 的 attention 不是普通 causal self-attention：

- Query 来自 draft block 的 hidden states。
- Key/Value 来自 `[target context hidden | draft block hidden]` 的拼接。
- target 多层 hidden states 先 concat，再通过 `fc` 投影回 draft hidden size。
- 每个 draft layer 都对 target context 与自身 block 做 cross-attention。

### Target model 角色

target model 在训练中有两个作用：

1. 输出指定层的 hidden states，作为 DFlash 的 context feature。
2. 提供 embedding 和 lm_head：trainer 中的 `TargetEmbeddingsAndHead` 只加载 target 的 `embed_tokens` 和 `lm_head`，用于构造 mask/noise embedding 和把 draft hidden 投到 vocab logits。

online 模式下 target model 每步前向；offline 模式下 target hidden 已经预先缓存，但仍需要 target embedding/lm_head。

## Trainer 与 loss 链路

核心 trainer：

- `OnlineDFlashTrainer`
- `OfflineDFlashTrainer`

注册方式：

- online: `@Eagle3TrainerFactory.register("online", "DFlash")`
- offline: `@Eagle3TrainerFactory.register("offline", "DFlash")`

核心 loss 函数：

- `OnlineDFlashTrainer.compute_loss()`
- `OnlineDFlashTrainer._compute_dflash_loss_and_accuracy()`

训练 step 的详细流程：

1. 准备数据：
   - online 跑 target model 得到 `hidden_states`
   - offline 直接解包 batch 中的 `hidden_states`
2. 随机采样 anchor：
   - 从 `loss_mask` 有效区域抽样。
   - 每条序列最多 `num_anchors` 个。
   - anchor 必须保证后面有足够 block 范围可预测。
3. 构造 noise block：
   - 每个 block 第 0 个位置放 anchor token 的真实 embedding。
   - block 其他位置放 `mask_token_id` 的 embedding。
4. 构造 position ids：
   - context 使用原序列位置。
   - draft block 使用 `anchor + offset` 的绝对位置。
5. 构造 DFlash attention mask：
   - block 可以看 anchor 之前的 context。
   - block 内部 bidirectional 可见。
   - 不同 block 之间互不可见。
   - invalid block 不可见。
6. 前向：
   - `QwenDFlashDraftModel(noise_embedding, target_hidden, attention_mask, position_ids)`
   - 输出 hidden 后接 target `lm_head` 得 logits。
7. 构造 label：
   - block 内第 k 个位置预测 `input_ids[anchor + k]`。
   - k=0 是 anchor 自身，跳过不算 loss。
8. 加权 CE loss：
   - 权重 = valid block * in-bounds * loss_mask * position decay。
   - decay 为 `exp(-(k-1)/loss_decay_gamma)`，k=1 权重为 1。
9. 记录 accuracy：
   - accuracy 使用未 decay 的二值 mask。

## Optimizer 与分布式训练

DFlash trainer 自定义了 FP32 master weights optimizer，原因是 bf16 momentum/variance 精度不足，训练数千步后可能损害质量。

路径：

- FSDP：`_FP32StateAdamW`
- DDP/single GPU：`_FP32MasterWeightOptimizer`

更新逻辑：

1. bf16 gradient cast 到 fp32。
2. fp32 grad norm clipping。
3. 在 fp32 master weights 上做 AdamW。
4. 把 fp32 master weights copy 回 bf16 模型参数。

官方脚本默认使用 FSDP：

```bash
--fsdp "shard_grad_op auto_wrap"
--fsdp_config configs/fsdp_config.json
```

## 推理/验证链路

DFlash draft model 自带 `spec_generate()`：

1. target model prefill 输入 prompt。
2. 抽取 target hidden states。
3. draft model 一次预测 `block_size - 1` 个后续 token。
4. target model 并行验证整个 block。
5. 计算最长连续匹配前缀 acceptance length。
6. 接受匹配 token，并用 target posterior 修正第一个不匹配位置。
7. 重复直到达到最大长度或 stop token。

benchmark 入口：

- `AngelSlim/tools/dflash_benchmark.py`

DFlare 文档里说明该 benchmark 同时支持 DFlash/DFlare，通过 `--draft-arch dflash|dflare` 切换。

## 关键代码链路图

```text
run_dflash_online.sh
  -> tools/train_dflash_online.py
    -> DraftModelConfig.from_file(configs/qwen3_dflash.json)
    -> create_target_model(...)
    -> create_draft_model(...)
      -> DraftModelFactory.from_config(...)
      -> QwenDFlashDraftModel
    -> DatasetManager.create_online_datasets()
      -> OnlineLLMDatasetBuilder / OnlineDatasetBuilder
      -> DataCollatorWithPadding
    -> Eagle3TrainerFactory.create("online", "DFlash")
      -> OnlineDFlashTrainer
      -> Trainer.train()
        -> compute_loss()
          -> prepare_data_for_draft_model()
            -> target_model.get_hidden_states_and_logits(...)
          -> _sample_anchor_positions()
          -> _create_noise_embed()
          -> create_dflash_block_mask()
          -> QwenDFlashDraftModel.forward()
          -> target_lm_head(...)
          -> weighted CE loss
```

```text
generate_dflash_data.sh
  -> tools/generate_dflash_data.py
    -> create_target_model(...)
    -> DatasetManager.create_all_datasets()
    -> target_model.get_hidden_states_and_logits(...)
    -> save .ckpt(input_ids, hidden_states, loss_mask, attention_mask)

run_dflash_offline.sh
  -> tools/train_dflash_offline.py
    -> OfflineDFlashDataset(load .ckpt)
    -> create_draft_model(...)
    -> Eagle3TrainerFactory.create("offline", "DFlash")
      -> OfflineDFlashTrainer extends OnlineDFlashTrainer
      -> same _compute_dflash_loss_and_accuracy()
```

## 重要文件索引

- `AngelSlim/scripts/speculative/run_dflash_online.sh`：online 官方启动脚本。
- `AngelSlim/scripts/speculative/generate_dflash_data.sh`：offline hidden cache 生成脚本。
- `AngelSlim/scripts/speculative/run_dflash_offline.sh`：offline 官方启动脚本。
- `AngelSlim/tools/train_dflash_online.py`：online 训练主入口。
- `AngelSlim/tools/train_dflash_offline.py`：offline 训练主入口。
- `AngelSlim/tools/generate_dflash_data.py`：预生成 `.ckpt` hidden states。
- `AngelSlim/configs/qwen3_dflash.json`：Qwen3 DFlash draft 配置。
- `AngelSlim/angelslim/compressor/speculative/train/models/draft/qwen_dflash.py`：DFlash draft model。
- `AngelSlim/angelslim/compressor/speculative/train/trainer/online_dflash_trainer.py`：DFlash loss、anchor、attention mask、optimizer。
- `AngelSlim/angelslim/compressor/speculative/train/trainer/offline_dflash_trainer.py`：offline trainer。
- `AngelSlim/angelslim/compressor/speculative/train/data/dataset.py`：DatasetManager。
- `AngelSlim/angelslim/compressor/speculative/train/data/dataset_builder/base_dataset_builder.py`：conversation tokenization 和 loss mask。

