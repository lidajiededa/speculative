# Standalone DFlash Offline Trainer

这是从 AngelSlim `scripts/speculative/run_dflash_offline.sh` 训练链路中抽出的独立离线训练工程。运行时不导入、不安装、也不要求存在 `angelslim` 包；输入仍兼容原脚本生成的 `.ckpt` hidden-state cache。

当前版本面向 **Qwen3-30B-A3B target + Qwen3 DFlash draft head + 8 张昇腾 910B**，固定使用 Transformers 5.5.3 的 Qwen3 基础层。

## 1. 工程边界

保留的功能：

- 离线 `.ckpt` 数据加载、按 batch 补齐及四个张量同步截断。
- DFlash anchor 随机采样、MASK block 构造及绝对位置编码。
- `context < anchor`、block 内双向、block 间隔离的注意力规则。
- Qwen3 DFlash cross-attention draft model。
- 指数衰减加权 CE loss 和不带衰减的 top-1 accuracy。
- 只从 target 权重中加载 `embed_tokens` 和 `lm_head`。
- bf16 训练、FP32 master/state AdamW、FSDP、断点恢复和 HF checkpoint 保存。
- Ascend NPU 的 SDPA/dense-mask 路径。

没有包含的功能：hidden states 生成、在线 target forward、EAGLE3、VLM、多后端推理及 speculative decoding benchmark。这些都不是 `run_dflash_offline.sh` 完成一次离线训练所必需的部分。

## 2. 代码链路

```text
scripts/run_qwen3_30b_a3b_8npu.sh
  -> train.py
     -> configuration.py              读取 draft JSON
     -> data.py                       加载/校验/截断/拼 batch
     -> modeling_qwen3_dflash.py      Qwen3 DFlash draft forward
     -> trainer.py
        -> target_weights.py          只加载 target embedding + lm_head
        -> objective.py               anchor、label、weight
        -> masks.py                   dense bool mask 或 BlockMask
        -> optim.py                   FP32 optimizer
        -> transformers.Trainer       FSDP、日志、保存和恢复
```

原 AngelSlim 链路中的工厂类和 Eagle3 父类已全部消除。可以执行以下命令确认源码没有框架导入：

```bash
rg 'from angelslim|import angelslim' .
```

预期无输出。

## 3. 环境准备

先按昇腾软件栈的版本配套关系安装 CANN、PyTorch 和 `torch_npu`。不要让 `pip` 用 CUDA/CPU 版 torch 覆盖 NPU 版本。随后在本目录安装 Python 依赖：

```bash
cd /path/to/dflash_offline_standalone
pip install -r requirements.txt
```

`requirements.txt` 不包含 `torch` 和 `torch_npu`，并固定 `transformers==5.5.3`。启动前检查：

```bash
python - <<'PY'
import torch
import torch_npu
import transformers
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('transformers:', transformers.__version__)
print('npu count:', torch.npu.device_count())
PY
```

## 4. 离线数据要求

每个 `.ckpt` 表示一个样本，至少包含：

| 键 | 形状 | 类型 | 含义 |
|---|---:|---|---|
| `input_ids` | `[1, S]` | `torch.long` | target tokenizer 的 token IDs |
| `attention_mask` | `[1, S]` | 整数/bool | 可选；缺失时自动补全 1 |
| `loss_mask` | `[1, S]` | 整数/float | assistant/有效监督 token 为 1 |
| `hidden_states` | `[1, S, 10240]` | 建议 bf16 | 5 个 target layer hidden 拼接结果 |

Qwen3-30B-A3B 配置使用 `target_layer_ids=[1,12,23,34,45]`，target hidden size 为 2048，所以最后一维必须为 `5 × 2048 = 10240`，拼接顺序也必须完全一致。

训练前先检查数据：

```bash
python tools/validate_hidden_cache.py \
  --hidden_path /data/qwen3_30b_hidden_cache \
  --draft_config configs/qwen3_30b_a3b_dflash_npu.json
```

先用 `--limit 100` 快速抽检，正式训练前再去掉 `--limit` 全量检查。

## 5. 8×910B 启动

```bash
export TARGET_MODEL_PATH=/models/Qwen3-30B-A3B
export TRAIN_HIDDEN_PATH=/data/qwen3_30b_hidden_cache/train
export EVAL_HIDDEN_PATH=/data/qwen3_30b_hidden_cache/eval   # 可选
export OUTPUT_DIR=/data/outputs/qwen3_30b_a3b_dflash

bash scripts/run_qwen3_30b_a3b_8npu.sh 8
```

默认关键参数为：单卡 batch 1、梯度累积 2、`block_size=16`、`num_anchors=128`、`model_max_length=3072`、bf16、SDPA、FP32 master/state optimizer，以及 FSDP `SHARD_GRAD_OP`。

可通过环境变量覆盖高频参数：

```bash
MODEL_MAX_LENGTH=8192 NUM_ANCHORS=64 GRAD_ACC_STEPS=4 \
bash scripts/run_qwen3_30b_a3b_8npu.sh 8
```

32K 样本建议从 `MODEL_MAX_LENGTH=32768 NUM_ANCHORS=64 BLOCK_SIZE=16` 起步。dense mask 的 query 长度是 `num_anchors × block_size`，所以长上下文 OOM 时应优先降低 `NUM_ANCHORS`，其次降低 `BLOCK_SIZE`。

## 6. 为什么 NPU 不再出现 BlockMask 报错

AngelSlim 原实现创建 Flex Attention `BlockMask`，然后 Qwen3 attention 根据 `_attn_implementation` 分派。当实际落到 Transformers eager attention 时，源码执行 `Tensor + attention_mask`，于是得到 `Tensor + BlockMask` 的类型错误。

独立版不再调用 Transformers 的 Qwen3 attention dispatcher：

- `sdpa/eager` 创建 `[B,1,N×block_size,S+N×block_size]` 的 bool 可见性 mask。
- 自有 eager 路径用 `masked_fill`，SDPA 路径直接把 bool mask 交给 `scaled_dot_product_attention`。
- 只有显式选择 `flex_attention` 时才创建并消费 `BlockMask`。

因此 `BlockMask` 不会进入 eager/SDPA，也不需要修改 Transformers site-packages。

## 7. Loss 与准确率

对每个 anchor 建立一个长度为 `block_size` 的 draft block。block 第 0 位放真实 anchor token，其余位置放 MASK；第 `k` 位对应标签 `input_ids[anchor+k]`。

- `k=0` 只提供条件，不计 loss。
- 标签越界、无效 block、`loss_mask=0` 的位置不计 loss。
- 若 `loss_decay_gamma=7`，位置权重为 `exp(-(k-1)/7)`。
- 总 loss 是所有有效 token 的加权 CE 均值。
- accuracy 使用相同有效位置，但不乘指数衰减，是 token top-1 accuracy。

日志中的 `dflash_loss` 和 `dflash_accuracy` 是一个 logging interval 内各 micro-batch 的均值；HF 的 `loss` 是 Trainer 按梯度累积处理后的训练 loss。

## 8. 截断、保存与恢复

`--model_max_length` 会同步截断 `input_ids/attention_mask/loss_mask/hidden_states`。这一点与原入口不同：原脚本虽然传了 `model_max_length`，离线 Dataset 实际没有使用它。

启动脚本默认 `--resume_from_checkpoint auto`：输出目录中存在 `checkpoint-*` 时恢复最新 checkpoint，否则从头训练。训练结束会在 `OUTPUT_DIR` 保存 draft model config、权重、Trainer 状态及指标。

加载训练结果时需要继续使用本工程的模型类：

```python
from dflash_offline.configuration import load_draft_config
from dflash_offline.modeling_qwen3_dflash import QwenDFlashDraftModel

config = load_draft_config('/data/outputs/qwen3_30b_a3b_dflash/config.json')
model = QwenDFlashDraftModel.from_pretrained(
    '/data/outputs/qwen3_30b_a3b_dflash',
    config=config,
)
```

## 9. 本地验证

```bash
pytest -q
```

测试覆盖 dense mask 的逐元素真值对照、标签与衰减权重、tiny Qwen3 在 eager/SDPA 上的前后向，以及包含本地 target 权重和 `.ckpt` 数据的 Trainer 单步训练。

## 10. 文件来源与许可

Qwen3 DFlash draft model、训练目标和 FP32 optimizer 的数学/实现来源于 Tencent AngelSlim；独立化、dense NPU attention、数据截断/校验、启动脚本和测试是本工程的改造。保留了 Apache License 2.0 的 `LICENSE` 和归属说明 `NOTICE`。
