# vllm-project/speculators 下载下来

> [vllm-project/speculators](https://github.com/vllm-project/speculators) 下载下来

<details><summary>4 previous messages</summary>

> 我会把仓库克隆到当前工作目录，并先确认没有同名目录冲突。
>
> 克隆还在进行中，我在等 Git 完成并会核对下载结果。
>
> 第一次连接 GitHub 时被重置了；我改用浅克隆重试，能减少传输量。
>
> GitHub 的 Git 端口目前连不上。我再尝试 GitHub 官方的源码压缩包接口；下载成功后会解压到同一目录。
</details>

已下载并解压到：

[speculators](./speculators)

由于 GitHub 的 Git 连接失败，本次下载的是 `main` 分支源码压缩包，因此不包含 `.git` 历史。

> 我要为GLM5.2训练一个和https://www.modelscope.cn/models/UCloud-AILab/GLM-5.2-FP8-DFlash 结构相近的DFlash头，如何使用该框架来做？

<details><summary>4 previous messages</summary>

> 我会先对照本地 `speculators` 的训练入口、DFlash 配置与数据格式，再核查你给的 ModelScope 模型结构；最后给出一套针对 GLM‑5.2 的可执行迁移方案和命令。
>
> 本地框架已经确认：DFlash 草稿网络固定采用 Qwen3 风格层，但可挂接不同结构的 verifier。现在关键是核对 GLM‑5.2 的层数、隐藏维度、词表和参考 DFlash 头的 `config.json`，判断能否直接复刻配置，还是需要先补兼容。
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `https://www.modelscope.cn/models/UCloud-AILab/GLM-5.2-FP8-DFlash`
>
> Searched the web for `site:modelscope.cn/models/UCloud-AILab/GLM-5.2-FP8-DFlash config.json ...`
>
> </details>
>
> 这里有一个容易踩坑的编号差异：参考模型配置写的是 `[1,20,38,56,75]`，而 `speculators` 的第三方 DFlash 转换器会把这类层号统一加 1。也就是说，用这个框架从头训练时，应使用 `[2,21,39,57,76]`，并让 vLLM 额外抽取最终层 `78` 作为训练目标 logits 的来源。
</details>

可以。最稳妥的做法是：用 GLM‑5.2‑FP8 作为只负责生成 hidden states 的 verifier，用 `speculators` 从零训练一个 Qwen3-style、5 层、block size 16 的 DFlash drafter。

关键点是不要直接只写 `--num-layers 5`。框架自动继承 GLM 的 attention 参数时会得到 `head_dim=192`，而参考 DFlash 头是 `head_dim=64`，且参考模型五层都是 full attention；因此必须用 `--draft-config` 明确定义草稿网络。

## 1. 目标结构

参考模型的公开配置与训练说明显示：5 层、hidden size 6144、FFN 12288、64 个 attention/KV heads、head dim 64、block size 16、完整 154880 词表，训练时取 GLM‑5.2 的五层 hidden states。[参考模型 config.json](https://huggingface.co/UCloud-org/GLM-5.2-FP8-DFlash/blob/main/config.json)、[模型说明](https://huggingface.co/UCloud-org/GLM-5.2-FP8-DFlash)

| 项目 | 参考模型 | 在 speculators 中 |
|---|---:|---|
| Draft 架构 | Qwen3-style dense | `model_type=qwen3` |
| Draft 层数 | 5 | config 中 `num_hidden_layers=5` |
| Hidden size | 6144 | `hidden_size=6144` |
| FFN size | 12288 | `intermediate_size=12288` |
| Attention heads | 64 | `num_attention_heads=64` |
| KV heads | 64 | `num_key_value_heads=64` |
| Head dim | 64 | `head_dim=64` |
| Attention | 五层 full attention | 五个 `full_attention` |
| Block size | 16 | `--block-size 16` |
| Mask token | 154821 | `--mask-token-id 154821` |
| 原始层号 | 1,20,38,56,75 | speculators 中用 2,21,39,57,76 |
| Loss | CE，gamma 7 | `--loss-fn ce --dflash-decay-gamma 7` |
| 最大 anchors | 512 | `--max-anchors 512` |

层号需要加 1，是因为原始 DFlash 的 `hidden_states[0]` 是 embedding 输出；`speculators` 转换器也明确执行了这个偏移，见 [converter.py](./speculators/src/speculators/convert/dflash/converter.py:117)。

## 2. 创建 Draft 配置

在 Linux 训练机上创建 `configs/glm52_dflash_qwen3.json`：

```json
{
  "model_type": "qwen3",
  "vocab_size": 154880,
  "hidden_size": 6144,
  "intermediate_size": 12288,
  "num_hidden_layers": 5,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 1048576,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-5,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "tie_word_embeddings": false,
  "use_cache": true,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "sliding_window": null,
  "use_sliding_window": false,
  "rope_parameters": {
    "rope_theta": 8000000,
    "rope_type": "default"
  }
}
```

不要同时传以下参数，因为它们与 `--draft-config` 互斥：

```text
--num-layers
--sliding-window
--full-attention-indices
--draft-hidden-act
```

## 3. 环境和硬件布局

建议两个独立环境：

```bash
# 训练环境
uv venv .venv-speculators
source .venv-speculators/bin/activate
uv pip install -e .

# vLLM 环境
uv venv .venv-vllm
source .venv-vllm/bin/activate
uv pip install "vllm==0.23.0" --torch-backend=auto
uv pip install "transformers>=5.9.0"
```

GLM‑5.2‑FP8 本体约 743B，官方 vLLM 配置需要一台 8×H200/H20 节点；在线训练还需要另一组 GPU 放 DFlash 训练进程。[vLLM GLM‑5.2 官方配方](https://recipes.vllm.ai/zai-org/GLM-5.2)

推荐布局：

```text
节点 A：8×H200，运行 GLM-5.2-FP8 + hidden-state extraction
节点 B：4～8×H100/H200，运行 DFlash FSDP2 训练
两节点：挂载相同路径的共享文件系统
```

这里的 file hidden-state connector 会让节点 A 临时写文件、节点 B 读取后删除，所以两台机器必须看到相同的绝对路径，例如 `/shared/glm52_dflash_hs`。

## 4. 准备数据

参考模型使用的是经过 GLM‑5.2 非思考模式重新生成的 Nemotron + CodeAlpaca 数据。公开数据集可以直接被当前框架的 `hf:` loader 读取：

```bash
MODEL=/models/GLM-5.2-FP8
DATA=/data/glm52_dflash

python scripts/prepare_data.py \
  --model "$MODEL" \
  --data "hf:JessieWei/GLM-5.2-FP8-nemotron-codealpaca" \
  --output "$DATA" \
  --seq-length 3072 \
  --trust-remote-code
```

先做一次小规模验证：

```bash
python scripts/prepare_data.py \
  --model "$MODEL" \
  --data "hf:JessieWei/GLM-5.2-FP8-nemotron-codealpaca" \
  --output /data/glm52_dflash_smoke \
  --seq-length 3072 \
  --max-samples 5000 \
  --trust-remote-code
```

不建议加 `--draft-vocab-size 32000`。参考模型使用完整 154880 词表；省略该参数时，框架会使用 verifier 的完整词表。缩减词表能节省内存，但结构就不再与参考模型一致。

## 5. 启动 hidden-state 服务

在节点 A、vLLM 环境中：

```bash
MODEL=/models/GLM-5.2-FP8
SHARED_HS=/shared/glm52_dflash_hs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/launch_vllm.py "$MODEL" \
  --hidden-states-path "$SHARED_HS" \
  --target-layer-ids 2 21 39 57 76 \
  -- \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 3072 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

`launch_vllm.py` 会自动追加最终层 `78`。最终抽取列表应当是：

```text
[2, 21, 39, 57, 76, 78]
```

前五层拼接后输入 DFlash 的 `fc`；最后的第 78 层单独用于计算 verifier logits/训练目标。训练命令中的 `--target-layer-ids` 仍然只写前五个，不写 78。

不要在这个 vLLM 命令里启用 GLM 自带 MTP；脚本已经设置了 `extract_hidden_states` speculative method，两者不能叠加。

## 6. 启动 DFlash 训练

在节点 B、speculators 环境中：

```bash
MODEL=/models/GLM-5.2-FP8
DATA=/data/glm52_dflash
SHARED_HS=/shared/glm52_dflash_hs
OUT=/checkpoints/glm52_dflash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA" \
  --hidden-states-path "$SHARED_HS" \
  --vllm-endpoint "http://NODE_A_IP:8000/v1" \
  --save-path "$OUT" \
  --speculator-type dflash \
  --draft-config ./configs/glm52_dflash_qwen3.json \
  --block-size 16 \
  --max-anchors 512 \
  --target-layer-ids 2 21 39 57 76 \
  --mask-token-id 154821 \
  --no-sample-from-anchor \
  --loss-fn ce \
  --dflash-decay-gamma 7 \
  --epochs 6 \
  --lr 6e-4 \
  --optimizer adamw \
  --scheduler-type cosine \
  --scheduler-warmup-ratio 0.04 \
  --total-seq-len 3072 \
  --hidden-states-dtype bfloat16 \
  --draft-attn-impl simple_flex_attention \
  --on-missing generate \
  --on-generate delete \
  --fsdp-shard \
  --checkpoint-freq 1 \
  --save-best
```

这套参数尽量贴近参考模型公开的训练配方，但优化器实现并非完全一致：参考模型用的是带 update clipping 的 StableAdamW 风格实现，而该框架这里是 AdamW 加全局梯度裁剪。因此能做到结构和主要 recipe 接近，不能保证逐项复现参考训练结果。

`block_size=16` 配合 `sample_from_anchor=false` 时，框架表示为“1 个 anchor/bonus 位置 + 15 个 speculative tokens”。这是标准 DFlash 模式，不应改成 `sample_from_anchor=true`。

## 7. 先运行一次结构验证

全量训练前建议执行：

```bash
python scripts/train.py \
  --verifier-name-or-path "$MODEL" \
  --speculator-type dflash \
  --draft-config ./configs/glm52_dflash_qwen3.json \
  --block-size 16 \
  --target-layer-ids 2 21 39 57 76 \
  --mask-token-id 154821 \
  --no-sample-from-anchor \
  --save-path /checkpoints/glm52_dflash_init \
  --dry-run
```

然后检查生成的 `config.json`：

```bash
python - <<'PY'
import json

p = "/checkpoints/glm52_dflash_init/config.json"
c = json.load(open(p))

t = c["transformer_layer_config"]
assert t["hidden_size"] == 6144
assert t["intermediate_size"] == 12288
assert t["num_hidden_layers"] == 5
assert t["head_dim"] == 64
assert t["layer_types"] == ["full_attention"] * 5
assert c["block_size"] == 16
assert c["draft_vocab_size"] == 154880
assert c["aux_hidden_state_layer_ids"] == [2, 21, 39, 57, 76]
assert c["mask_token_id"] == 154821
print("GLM-5.2 DFlash config OK")
PY
```

## 8. 部署验证

训练完成后：

```bash
vllm serve "$OUT/checkpoint_best" \
  --tensor-parallel-size 8 \
  --max-model-len 131072
```

`speculators` 保存的是它自己的标准化 checkpoint 格式，vLLM 可以根据其中的 `speculators_config` 自动加载 GLM‑5.2 verifier 和 DFlash drafter。相关本地教程见 [train_dflash_online.md](./speculators/docs/user_guide/tutorials/train_dflash_online.md)。

如果只有一台 8 卡 GLM 节点，就只能先离线生成 hidden states、停止 GLM 服务后再训练；但 5 个辅助层加最终层、6144 维、最长 3072 token 的缓存规模会达到数十 TB，通常不如“8 卡 verifier 节点 + 独立训练节点”的纯在线方案实际。

> 启动 hidden-state 服务这一步中的vllm服务只负责提取hidden-state吗？重新生成数据这一步是要我自己先做好吗？

对，这里要区分两个完全不同的阶段。

1. 重新生成回答：生成训练文本
2. Hidden-state 服务：为已经确定的 token 序列提取隐藏状态

### Hidden-state 服务做什么？

`launch_vllm.py` 启动的 vLLM 服务主要负责：

```text
训练数据中的 input_ids
        ↓
GLM‑5.2 前向计算
        ↓
提取第 2、21、39、57、76、78 层 hidden states
        ↓
DFlash 训练进程读取
```

它不会替你重新编写数据集中的 assistant 回答，也不会把原始问题变成新的问答数据。

训练时：

```bash
--on-missing generate
--on-generate delete
```

这里的 `generate` 指“生成缺失的 hidden states”，不是生成新的文本回答。

### 重新生成数据是否要提前做？

分两种情况。

#### 使用参考模型公开数据集

如果使用：

```text
JessieWei/GLM-5.2-FP8-nemotron-codealpaca
```

则不需要重新生成。这个数据集的回答已经由 GLM‑5.2‑FP8 以非思考模式重新生成过。

直接执行预处理即可：

```bash
python scripts/prepare_data.py \
  --model "$MODEL" \
  --data "hf:JessieWei/GLM-5.2-FP8-nemotron-codealpaca" \
  --output "$DATA" \
  --seq-length 3072 \
  --trust-remote-code
```

然后启动 hidden-state 服务并训练。

#### 使用你自己的数据集

建议先让 GLM‑5.2 重新生成 assistant 回答，使数据变成 on-policy 数据：

```text
原始用户问题
    ↓
GLM‑5.2 重新回答
    ↓
得到 on-policy JSONL
    ↓
prepare_data.py 分词
    ↓
hidden-state extraction
    ↓
训练 DFlash
```

完整流程是：

```bash
# 第一步：用普通 vLLM 服务重新生成回答
vllm serve "$MODEL" \
  --tensor-parallel-size 8 \
  --port 8000
```

然后运行：

```bash
python scripts/response_regeneration/script.py \
  --dataset /data/my_original_dataset.jsonl \
  --endpoint http://NODE_A_IP:8000/v1 \
  --outfile /data/my_glm52_regenerated.jsonl
```

具体参数以当前脚本帮助为准：

```bash
python scripts/response_regeneration/script.py --help
```

接着预处理：

```bash
python scripts/prepare_data.py \
  --model "$MODEL" \
  --data /data/my_glm52_regenerated.jsonl \
  --output /data/my_glm52_dflash \
  --seq-length 3072 \
  --trust-remote-code
```

最后停止普通生成服务，改为启动 hidden-state 服务：

```bash
python scripts/launch_vllm.py "$MODEL" \
  --hidden-states-path /shared/glm52_dflash_hs \
  --target-layer-ids 2 21 39 57 76 \
  -- \
  --tensor-parallel-size 8 \
  --max-model-len 3072 \
  --port 8000
```

### 能否跳过重新生成回答？

可以，但通常效果会差一些。

如果原始 assistant 回答不是 GLM‑5.2 生成的，DFlash 实际学的是：

```text
“其他模型/人工回答的 token”
```

而推理时希望它预测的是：

```text
“GLM‑5.2 下一步会输出的 token”
```

两者分布不一致，通常会降低 speculative token 接受长度。

所以建议：

- 首次跑通流程：可以用原始数据，先做 5K 样本 smoke test。
- 正式训练：使用 GLM‑5.2 重新生成的 on-policy 回答。
- 直接使用公开参考数据集：不需要你再次生成。

换句话说，推荐顺序是：

```text
重新生成回答（一次）
→ prepare_data（一次）
→ 启动 hidden-state 服务
→ DFlash 在线训练
```

重新生成回答和 hidden-state 提取可以共用同一套 GLM‑5.2 GPU，但不能同时运行；前一阶段完成后重启成 hidden-state extraction 模式即可。
