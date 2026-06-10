# DFlash 投机解码在行业增训大模型中的落地规划

## 1. 背景

DFlash 是一种面向大模型推理加速的投机解码方案。相比传统 draft model 或 MTP，DFlash 的核心价值在于：利用目标模型中间 hidden states，训练一个轻量投机头，使其能够在不显著增加推理成本的前提下，一次提出多个候选 token，并通过目标模型验证来减少实际 decode 轮数。

在通用大模型场景中，DFlash 的收益受投机接受率影响较大。如果投机头预测分布与目标模型输出分布足够接近，则平均接受长度更高，推理速度收益更明显。

行业增训模型天然更适合 DFlash 落地，原因包括：

- 行业模型输出分布更集中，领域语料、术语、格式、回答风格更稳定。
- 业务场景通常有较强模板化特征，例如客服、问答、报告生成、代码生成、运维助手、金融/政务/医疗等垂直领域。
- 增训后的目标模型相对通用模型更容易被投机头拟合。
- 行业生产环境通常有明确 SLA，对 TTFT、TPOT、吞吐、成本都有刚性诉求。

因此，DFlash 在行业增训大模型领域具备明确落地价值：通过训练领域投机头，在保持目标模型效果不变的前提下，降低 decode 成本、提升吞吐、优化推理成本。

## 2. 当前现状

### 2.1 推理性能摸测已经验证出落地潜力

当前在 `Qwen3.6-35B-A3B` 上开启 DFlash 后，推理性能已经基本持平 MTP。这个结论非常关键，因为当前 DFlash 的平均接受长度仍只有约 `3~4`，还不是高接受率状态。

这说明：

- DFlash 当前实现已经具备可用的推理性能基础。
- 即使接受率不高，DFlash 也已经接近 MTP 性能。
- 如果后续通过领域数据训练提升接受率，DFlash 推理速度还有进一步提升空间。
- 在行业增训模型中，由于输出分布更稳定，DFlash 接受率有更大概率显著高于通用模型。

因此，从推理侧看，DFlash 已经具备进入行业落地验证的条件。

### 2.2 基础训练流程已打通

当前 DFlash 的基本训练链路已经打通，包括：

- DFlash 数据构造。
- 目标模型 hidden states 采集。
- DFlash 投机头训练。
- DFlash 权重加载。
- vLLM / vLLM-Ascend 推理接入。
- NPU 侧推理链路调试。
- 首 token、图模式、`max_num_batched_tokens` 等关键性能问题定位。

当前训练能力边界：

- 现阶段 DFlash 训练流程暂时只支持 LLM 文本模型。
- 多模态 VLM 的 DFlash 训练尚未支持，需要后续单独适配。
- 因此第一阶段落地应优先选择 LLM 行业增训模型，VLM 场景作为专项适配工作推进。

这意味着当前工作重心可以从“是否能跑通”转向“如何训得更好、如何稳定上线、如何获得持续收益”。

## 3. 落地目标

DFlash 落地目标可以分为三个层次。

### 3.1 性能目标

- 在行业增训模型上，DFlash 推理吞吐显著优于不开投机。
- 在平均接受长度提升后，DFlash 性能超过 MTP。
- 在高并发场景下，TTFT 不明显劣化。
- TPOT 随接受长度提升呈稳定下降。
- 单卡/单机吞吐提升能转化为实际服务成本下降。

### 3.2 效果目标

DFlash 不改变目标模型输出分布，最终输出仍由目标模型验证，因此原则上不应损失模型效果。

需要验证：

- 严格投机验证路径下，输出质量与目标模型一致。
- 对业务评测集，回答准确率、格式遵循、幻觉率、拒答率无异常。
- 长文本、RAG、Agent 场景下无明显上下文相关退化。

### 3.3 工程目标

- 形成标准化 DFlash 训练、评测、部署流程。
- 形成不同业务场景的推荐参数。
- 形成线上灰度和回滚机制。
- 支持按模型、场景、并发、上下文长度动态选择是否开启 DFlash。

## 4. 落地流程

## 4.1 阶段一：场景选择与模型选择

优先选择以下行业场景：

- 输出格式稳定的问答类任务。
- RAG 问答。
- 行业客服。
- 报告生成。
- 数据分析解释。
- 代码/SQL/DSL 生成。
- Agent 中的规划、工具调用、结果总结。

优先选择以下模型：

- 已完成行业增训的 MoE 或 Dense 模型。
- 输出风格稳定、业务 prompt 模板稳定的模型。
- 线上请求量大、decode 成本占比较高的模型。
- 当前 MTP 或无投机场景下成本压力明显的模型。

不建议第一批选择：

- 输出极度开放的创作类场景。
- 高随机采样场景。
- 对长链推理中间过程非常敏感但缺少评测集的场景。
- 工具链尚未稳定的场景。
- 多模态 VLM 场景暂不作为第一批主线，因为当前 DFlash 训练流程暂时只支持 LLM，还不支持 VLM 训练，需要单独适配。

## 4.2 阶段二：训练数据集建设

DFlash 投机头训练数据应尽可能贴近线上目标模型的真实输出分布。

### 4.2.1 数据来源

推荐数据来源：

- 行业 SFT 数据。
- 线上真实请求和目标模型高质量回答。
- RAG 场景的检索上下文 + 问题 + 答案。
- Agent 场景的任务规划、工具调用轨迹、工具返回、最终回答。
- 长上下文问答数据。
- 高价值业务模板数据。
- 线上高频问题和长尾问题采样。

数据优先级：

```text
真实线上分布 > 业务评测集 > 行业 SFT 数据 > 通用指令数据
```

### 4.2.2 数据组织方式

建议按场景分桶：

- 短问答。
- 长问答。
- RAG 问答。
- Agent 工具调用。
- 格式化输出。
- 代码/SQL 生成。
- 多轮对话。
- 长上下文总结。

每个桶分别统计：

- prompt 长度分布。
- output 长度分布。
- 领域术语占比。
- 格式化输出占比。
- 工具调用占比。
- 平均接受长度。
- 每位置接受率。

### 4.2.3 数据规模建议

第一阶段可以从小规模验证开始：

| 阶段 | 数据规模 | 目标 |
| --- | --- | --- |
| POC | 1万~5万条 | 验证训练链路和接受率趋势 |
| 小规模业务验证 | 10万~50万条 | 验证特定业务场景收益 |
| 正式训练 | 100万条以上 | 覆盖主业务分布 |
| 持续迭代 | 每周/每月增量 | 跟随线上分布更新 |

### 4.2.4 数据质量要求

需要过滤：

- 错误答案。
- 格式不稳定答案。
- 超长无意义输出。
- 重复模板。
- 低质量合成数据。
- 与线上业务分布偏差过大的通用数据。

对 DFlash 来说，数据质量比数据量更重要。投机头的目标不是学会“更聪明”，而是更贴近目标模型在业务场景下的下一个 token 分布。

## 4.3 阶段三：训练参数选择

训练参数需要围绕“接受率”和“推理收益”优化，而不是只看训练 loss。

### 4.3.1 num_speculative_tokens

建议从以下值开始实验：

```text
4, 8, 15
```

选择原则：

- `K=4`：风险低，训练容易，收益稳定。
- `K=8`：适合多数生产场景，收益和成本平衡。
- `K=15`：适合接受率较高、输出分布稳定的行业场景。

最终选择不只看平均接受长度，还要看：

- TPOT。
- TTFT。
- 高并发吞吐。
- KV cache 压力。
- graph capture 覆盖情况。
- 拒绝后的恢复成本。

### 4.3.2 训练目标

建议同时观察：

- token-level loss。
- position-wise accuracy。
- per-position acceptance rate。
- unconditional acceptance rate。
- expected accepted length。
- 线上 prompt 分桶下的接受率。

需要特别关注每个 speculative 位置的接受率：

```text
accept@1, accept@2, ..., accept@K
```

如果前几位接受率高、后几位快速下降，可以考虑降低 K 或调整训练数据。

### 4.3.3 训练超参

建议实验矩阵：

- learning rate：`1e-5`、`5e-6`、`1e-6`。
- batch size：根据显存和序列长度调整。
- sequence length：覆盖线上主流 prompt 长度。
- warmup ratio：`0.03~0.1`。
- epoch：从 `1~3` 开始，避免过拟合特定模板。
- mixed precision：优先使用生产推理相近 dtype。

### 4.3.4 数据配比

推荐初始配比：

| 数据类型 | 比例 |
| --- | --- |
| 线上真实请求/回答 | 40%~60% |
| 行业 SFT 数据 | 20%~30% |
| RAG/Agent/长文本专项数据 | 20%~30% |
| 通用指令数据 | 0%~10% |

如果目标是行业模型落地，通用数据不宜过多，否则可能拉低领域分布拟合能力。

## 4.4 阶段四：长文本能力建设

长文本场景是 DFlash 行业落地的重要方向，尤其是：

- Agent。
- RAG。
- 长上下文问答。
- 长文档总结。
- 多轮对话。
- 复杂报告生成。

### 4.4.1 长文本训练数据

需要构造不同长度分桶：

| 长度区间 | 场景 |
| --- | --- |
| 0~2K | 普通问答、客服 |
| 2K~8K | RAG、长问答 |
| 8K~32K | 长文档问答、总结 |
| 32K+ | Agent 轨迹、超长上下文 |

训练时不要只堆长文本，也要保留短文本分布，否则短请求性能和接受率可能下降。

### 4.4.2 RAG 场景

RAG 的输出分布通常比开放问答更稳定，非常适合 DFlash。

建议单独构造：

- query。
- retrieved documents。
- citation style。
- final answer。
- 无答案拒答。
- 多文档综合。
- 表格/结构化信息抽取。

评测指标：

- 引用准确率。
- 答案 groundedness。
- 幻觉率。
- 接受长度。
- 不同 context 长度下的 TPOT。

### 4.4.3 Agent 场景

Agent 场景需要覆盖：

- 任务规划。
- 工具选择。
- 工具参数生成。
- 工具返回后的总结。
- 多步调用。
- 错误恢复。

DFlash 训练时要特别关注工具调用格式，因为工具调用通常模板稳定，容易获得较高接受率。

### 4.4.4 长上下文推理优化

长文本下需要同时考虑：

- DFlash context KV precompute 成本。
- 是否需要 DFlash draft window。
- 高并发下是否按长度阈值跳过 DFlash。
- `max_num_batched_tokens` 是否足够。
- graph capture size 是否覆盖常见 decode 并发。

对于超长 prompt，如果 DFlash 首轮 precompute 成本过高，可以采用：

- 首轮跳过 DFlash。
- 长 prompt 阈值跳过 DFlash。
- DFlash sliding window。
- 高并发或长上下文下动态降低 K。

### 4.4.5 多模态 VLM 训练适配

当前 DFlash 训练流程暂时只支持 LLM 文本模型，还不支持多模态 VLM 训练。若后续需要落地到图文问答、OCR、文档理解、视觉 Agent 等 VLM 场景，需要单独增加适配工作。

VLM 适配需要关注：

- 多模态输入格式：图片、视频、OCR 结果、视觉 token、文本 token 的拼接方式。
- processor / tokenizer 对齐：训练侧和推理侧必须保持一致。
- multimodal embeddings：需要明确 DFlash 训练时采集的是文本 hidden states、视觉 hidden states，还是融合后的 hidden states。
- target hidden states 采集：VLM 模型可能有 vision encoder、projector、language model 多段结构，需要确认采集层和采集位置。
- attention mask / position ids：图文混排下 position、mask、slot mapping 与纯文本不同。
- 数据集构造：需要多模态 instruction、图文问答、文档理解、图表理解、OCR、视觉工具调用等数据。
- 效果评测：除文本质量外，还要评估视觉 grounding、OCR 准确性、图文一致性和幻觉率。
- 推理适配：确认 vLLM/vLLM-Ascend 的 VLM 输入、KV cache、DFlash proposer 是否支持多模态 batch。

这部分由 A 主责推进，因为核心难点在于多模态训练目标、数据构造、hidden states 采集位置和效果评测定义；B 负责配合训练/评测脚本工程化和推理加载验证。

建议将 VLM 适配作为 LLM DFlash 落地稳定后的专项阶段，不阻塞第一批 LLM 行业模型上线。

## 4.5 阶段五：推理性能机制继续优化

当前已经验证 DFlash 在 `Qwen3.6-35B-A3B` 上，即使平均接受长度只有 `3~4`，也能基本持平 MTP。后续推理优化重点应该放在让 DFlash 的性能上限充分释放。

### 4.5.1 max_num_batched_tokens

DFlash 部署时必须按有效调度预算配置：

```text
effective_scheduler_budget
  = max_num_batched_tokens
    - max_num_seqs * (num_speculative_tokens - 1)
```

建议：

- 高并发、`K=15` 时，`max_num_batched_tokens` 从 `16384` 起步。
- 确保 `effective_scheduler_budget >= 8192`。
- 不要沿用非投机服务的 `4096` 默认值。

### 4.5.2 图模式

推荐：

```text
FULL_AND_PIECEWISE
```

如果不稳定：

```text
FULL_DECODE_ONLY -> PIECEWISE -> NONE
```

`cudagraph_capture_sizes` 按下面公式配置：

```text
capture_size = 并发请求数 * (num_speculative_tokens + 1)
```

例如：

```text
max_num_seqs = 256
num_speculative_tokens = 15
capture_size = 256 * 16 = 4096
```

建议配置覆盖热点档位：

```json
{
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "max_cudagraph_capture_size": 4096,
  "cudagraph_capture_sizes": [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
}
```

### 4.5.3 DFlash NPU 算子优化

继续优化方向：

- input expand kernel 从 single-grid 向更高并行度实现演进。
- 减少 hidden states copy。
- 优化 `precompute_and_store_context_kv()`。
- 去除不必要 clone。
- 优化 RoPE 调用和返回值处理。
- 优化 per-layer KV cache update。
- 对高并发和长 prompt 做 threshold skip。

### 4.5.4 动态策略

建议加入动态策略：

- 高并发时降低 K。
- 长 prompt 时跳过 DFlash 或使用 window。
- 低接受率场景自动降级到 MTP 或无投机。
- 对不同业务 route 使用不同 DFlash head。
- 对 RAG/Agent/短问答分别配置 DFlash 参数。

## 4.6 阶段六：评测体系

### 4.6.1 性能评测

需要覆盖：

- TTFT。
- TPOT。
- e2e latency。
- 吞吐 tokens/s。
- 请求吞吐 QPS。
- 平均接受长度。
- 每位置接受率。
- 拒绝率。
- NPU 利用率。
- 显存占用。
- graph 命中率。

评测维度：

- 不同并发：`1, 4, 8, 16, 32, 64, 128, 256`。
- 不同 prompt 长度：`512, 2K, 4K, 8K, 16K, 32K`。
- 不同 output 长度：短输出、中输出、长输出。
- 不同业务场景：RAG、Agent、问答、报告生成。

### 4.6.2 效果评测

需要验证：

- 与目标模型输出一致性。
- 业务自动评测指标。
- 人工评测。
- 格式遵循。
- 工具调用正确率。
- RAG 引用正确率。
- 长上下文信息召回。

因为 DFlash 走目标模型验证，理论上最终分布不变，但工程实现仍需要验证：

- 不同 batch 下输出一致。
- graph/eager 下输出一致。
- NPU/GPU 下输出一致。
- 长文本和短文本下输出一致。

### 4.6.3 接受率评测

接受率需要按场景分桶，不只看全局平均：

| 分桶 | 指标 |
| --- | --- |
| 短问答 | 平均接受长度、accept@k |
| RAG | 有检索/无检索接受率 |
| Agent | 工具调用前后接受率 |
| 长文本 | 不同上下文长度接受率 |
| 格式化输出 | JSON/Markdown/表格接受率 |

如果某些场景接受率明显低，可以选择：

- 单独增训该场景数据。
- 使用场景专属 DFlash head。
- 降低 K。
- 对该场景关闭 DFlash。

## 4.7 阶段七：灰度上线

推荐灰度策略：

### P0：离线验证

- 固定模型。
- 固定业务评测集。
- 对比无投机、MTP、DFlash。
- 验证质量一致性。
- 验证性能收益。

### P1：影子流量

- 使用线上真实请求。
- DFlash 只旁路生成，不返回用户。
- 统计接受率和性能。
- 记录异常请求。

### P2：小流量灰度

- 1% 流量开启 DFlash。
- 观察 TTFT、TPOT、错误率、拒绝率。
- 与 MTP 和无投机 A/B。

### P3：扩大灰度

- 10% -> 30% -> 50%。
- 按业务场景分 route 开启。
- 针对低接受率场景动态关闭。

### P4：正式上线

- 默认开启 DFlash。
- 保留 MTP/无投机回退。
- 持续采样线上数据增量训练。

## 5. 风险与应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| 接受率不稳定 | 某些场景速度收益低 | 场景分桶训练、动态开关 |
| 长文本首轮成本高 | TTFT 变差 | threshold skip、draft window、首轮跳过 |
| 图模式不稳定 | capture/replay 报错或性能抖动 | `FULL_DECODE_ONLY` / `PIECEWISE` 回退 |
| max_num_batched_tokens 配置过小 | 高并发 TTFT 异常 | 按 effective budget 配置 |
| DFlash head 泛化不足 | 业务新分布接受率低 | 增量训练、线上数据回流 |
| NPU 算子瓶颈 | 高并发收益不充分 | input expand、RoPE、KV update 优化 |
| VLM 训练暂未支持 | 多模态场景无法直接训练 DFlash head | 由 A 主责推进 VLM 数据、hidden states、训练目标和评测适配 |

## 6. 具体工作拆分

下面将 DFlash 行业落地拆成 10 条工作流，每条工作流都应有明确产出物和验收标准。

## 6.1 工作流 A：目标场景与基线确定

目标：

明确第一批落地业务、目标模型、性能基线和效果基线。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| A1 场景筛选 | 选择 RAG、Agent、客服、报告生成等候选场景 | 候选场景列表 | 明确 1~2 个 POC 场景 |
| A2 模型确认 | 确认目标模型、MTP 模型、DFlash head 基座 | 模型清单 | 模型版本、权重路径、推理参数固定 |
| A3 请求分布分析 | 统计 prompt/output 长度、并发、QPS、业务类型 | 请求分布报告 | 覆盖主要线上流量 |
| A4 基线压测 | 对无投机、MTP、当前 DFlash 做压测 | 性能基线报告 | TTFT、TPOT、吞吐、显存、接受率完整 |
| A5 效果基线 | 固定业务评测集和人工样例 | 效果基线集 | 能对比输出质量和业务指标 |

建议负责人：

```text
算法 + 推理 + 业务方
```

## 6.2 工作流 B：训练数据集建设

目标：

构建贴近行业增训模型真实输出分布的 DFlash 训练数据。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| B1 数据源梳理 | 整理 SFT、线上日志、RAG、Agent、长文本数据 | 数据源清单 | 数据来源、规模、权限明确 |
| B2 数据清洗 | 去重、去低质、去错误答案、去格式异常 | 清洗后数据集 | 抽检质量达标 |
| B3 场景分桶 | 按短问答、RAG、Agent、长文本、格式化输出分桶 | 分桶数据集 | 每桶样本量和长度分布明确 |
| B4 目标模型生成 | 用目标模型生成或重放高质量答案 | 目标输出数据 | 输出风格与线上一致 |
| B5 hidden states 采集 | 采集 DFlash 训练所需 hidden states | hidden states 数据 | shape、dtype、层号正确 |
| B6 数据版本管理 | 固化数据版本、hash、采样比例 | 数据版本说明 | 可复现实验 |

验收指标：

```text
POC 数据量 >= 1万条
业务验证数据量 >= 10万条
正式训练数据量建议 >= 100万条
```

## 6.3 工作流 C：DFlash 训练实验

目标：

训练可用于业务模型的 DFlash head，并找到合适的 K、学习率、数据配比。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| C1 训练配置确认 | 确认 target layers、hidden size、K、dtype | 训练配置文件 | 配置可复现 |
| C2 小规模试训 | 使用 1万~5万数据快速试训 | POC DFlash head | 能正常加载推理 |
| C3 K 值实验 | 对比 K=4、8、15 | K 值对比报告 | 明确推荐 K |
| C4 超参实验 | 对比 lr、batch、epoch、warmup | 超参报告 | loss 和接受率稳定 |
| C5 数据配比实验 | 调整 RAG/Agent/短问答/长文本比例 | 数据配比报告 | 分桶接受率提升 |
| C6 checkpoint 选择 | 按接受率和效果选择 checkpoint | 最优 DFlash head | 离线指标优于当前版本 |

核心指标：

```text
平均接受长度
accept@1..K
分桶接受率
训练 loss
离线推理速度
业务评测效果
```

## 6.4 工作流 D：长文本专项能力

目标：

保证 DFlash 在 RAG、Agent、长上下文问答等长文本场景可用，并避免 TTFT 劣化。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| D1 长度分桶构建 | 构造 2K、8K、32K、32K+ 数据 | 长文本数据集 | 每个长度桶有代表样本 |
| D2 RAG 数据增强 | 构造检索上下文、引用、拒答数据 | RAG 专项数据 | RAG 接受率单独可评估 |
| D3 Agent 数据增强 | 构造工具调用、规划、工具返回总结数据 | Agent 专项数据 | 工具调用格式稳定 |
| D4 长文本训练 | 引入长文本数据继续训练 | 长文本 DFlash head | 长文本接受率提升 |
| D5 长文本推理策略 | 设计 window、首轮跳过、阈值跳过策略 | 长文本策略配置 | TTFT 不明显劣化 |
| D6 长文本评测 | 评估长上下文召回、幻觉、引用准确性 | 长文本评测报告 | 质量无退化 |

重点判断：

```text
如果长文本 DFlash precompute 成本超过收益，应对长 prompt 启用 threshold skip 或 window。
```

## 6.5 工作流 E：推理性能优化

目标：

让 DFlash 在 NPU 生产环境中稳定释放收益。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| E1 max_num_batched_tokens 调优 | 按 effective budget 配置 8192/16384/32768 | 推荐部署参数 | 高并发 TTFT 正常 |
| E2 图模式调优 | 对比 FULL_AND_PIECEWISE、FULL_DECODE_ONLY、PIECEWISE | 图模式报告 | graph 命中率和稳定性达标 |
| E3 capture sizes 调优 | 按 `并发 * (K+1)` 设置档位 | capture size 配置 | 热点并发命中 FULL graph |
| E4 input expand 优化 | 评估 single-grid/2D-grid/NPU 多核风险 | 算子优化报告 | 高并发 input expand 不再是大头 |
| E5 precompute 优化 | 优化 hidden copy、RoPE、KV cache update | precompute 优化 patch | DFlash first pass 耗时下降 |
| E6 动态策略 | 高并发/长文本/低接受率时动态降级 | 动态策略实现 | 无明显坏 case |

关键配置公式：

```text
effective_scheduler_budget
  = max_num_batched_tokens
    - max_num_seqs * (num_speculative_tokens - 1)
```

```text
cudagraph_capture_size
  = 并发请求数 * (num_speculative_tokens + 1)
```

## 6.6 工作流 F：离线评测体系

目标：

建立 DFlash 专属的离线评测体系，能同时评价效果和性能。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| F1 性能评测脚本 | 固定并发、长度、输出 token、采样参数 | benchmark 脚本 | 可一键复现 |
| F2 效果评测集 | 固定业务评测集和人工样例 | eval 数据集 | 版本化管理 |
| F3 接受率评测 | 统计平均接受长度和 accept@k | 接受率报告 | 可按场景分桶 |
| F4 长文本评测 | RAG、Agent、长上下文专项评估 | 长文本报告 | 质量和性能都可比较 |
| F5 三方对比 | 无投机、MTP、DFlash 对比 | 对比报告 | 明确收益结论 |

推荐输出表格：

```text
model / method / concurrency / prompt_len / output_len /
TTFT / TPOT / throughput / accept_len / accept@k / graph_hit_rate
```

## 6.7 工作流 G：上线灰度与回滚

目标：

保证 DFlash 可以安全进入生产流量，并能快速回滚。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| G1 影子流量 | DFlash 旁路生成，不返回用户 | 影子评测报告 | 无异常输出和性能风险 |
| G2 小流量灰度 | 1% 流量启用 DFlash | 灰度报告 | 错误率、延迟无异常 |
| G3 A/B 实验 | DFlash vs MTP vs 无投机 | A/B 报告 | DFlash 收益明确 |
| G4 动态开关 | 按场景、长度、并发关闭 DFlash | 开关策略 | 可实时生效 |
| G5 回滚机制 | 退回 MTP 或无投机 | 回滚预案 | 分钟级回滚 |

上线门槛：

```text
质量无回退
TTFT 不劣化或可接受
TPOT / throughput 优于基线
错误率无上升
显存和稳定性可控
```

## 6.8 工作流 H：监控与数据回流

目标：

上线后持续监控 DFlash 收益，并用线上数据继续提升投机头。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| H1 接受率监控 | 平均接受长度、accept@k、拒绝率 | 监控 dashboard | 可按业务分桶 |
| H2 性能监控 | TTFT、TPOT、吞吐、graph 命中率 | 性能 dashboard | 异常可报警 |
| H3 质量监控 | 用户反馈、badcase、业务指标 | 质量 dashboard | 可追踪问题 |
| H4 数据回流 | 采样线上请求和目标模型输出 | 增量训练数据 | 定期生成 |
| H5 周期增训 | 每周/月更新 DFlash head | 新版本 head | 接受率持续提升 |

监控维度：

```text
业务 route
prompt 长度
output 长度
并发
模型版本
DFlash head 版本
num_speculative_tokens
```

## 6.9 工作流 I：多模型规模化复制

目标：

将单模型经验沉淀成标准流程，复制到多个行业增训模型。

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| I1 标准训练模板 | 固化数据、训练、评测配置 | 训练模板 | 新模型可复用 |
| I2 标准部署模板 | 固化推理参数和图模式配置 | 部署模板 | 新服务可快速上线 |
| I3 多模型 head 管理 | 管理不同模型/业务的 DFlash head | head registry | 版本可追踪 |
| I4 自动评测流水线 | 训练后自动跑接受率和性能评测 | CI/eval pipeline | 自动产出报告 |
| I5 最佳实践沉淀 | 汇总参数、风险、案例 | 最佳实践文档 | 可指导后续项目 |

## 6.10 工作流 J：多模态 VLM 训练适配

目标：

将当前仅支持 LLM 的 DFlash 训练流程扩展到多模态 VLM，为后续图文问答、OCR、文档理解、视觉 Agent 等场景落地做准备。

当前状态：

```text
当前 DFlash 训练暂时只支持 LLM。
VLM 训练不作为第一批上线阻塞项，但需要作为专项适配工作规划。
```

具体任务：

| 任务 | 内容 | 产出物 | 验收标准 |
| --- | --- | --- | --- |
| J1 VLM 模型链路梳理 | 梳理 vision encoder、projector、LLM、processor/tokenizer 调用链 | VLM 训练链路分析 | 明确需要采集的 hidden states 位置 |
| J2 多模态数据格式定义 | 定义图片/视频/OCR/文本输入与目标输出格式 | VLM DFlash 数据 schema | 训练侧和推理侧格式一致 |
| J3 hidden states 采集适配 | 适配图文混排输入下的 hidden states 采集 | VLM hidden states 采集脚本 | shape、层号、位置和 dtype 正确 |
| J4 position/mask 适配 | 处理视觉 token、文本 token 的 position ids、attention mask、slot mapping | 多模态 mask/position 方案 | 与目标 VLM forward 对齐 |
| J5 VLM DFlash 训练适配 | 修改训练脚本支持 multimodal embeddings / fused hidden states | VLM DFlash 训练脚本 | 能完成小规模训练 |
| J6 VLM 推理加载验证 | 验证 VLM DFlash head 在 vLLM/vLLM-Ascend 中加载和推理 | VLM 推理 smoke test | 图文请求可跑通 |
| J7 VLM 效果评测 | 构建图文问答、OCR、文档理解、视觉 Agent 评测 | VLM 评测报告 | 文本质量和视觉 grounding 无退化 |
| J8 VLM 接受率评测 | 统计不同多模态任务下的 accept@k 和平均接受长度 | VLM 接受率报告 | 明确是否具备加速价值 |

负责人：

```text
A 主责，B 协助。
```

分工说明：

- A 负责 VLM 数据 schema、训练目标、hidden states 采集位置、样本筛选、效果评测和 checkpoint 选择。
- B 负责采集脚本工程化、训练脚本改造、推理加载验证、性能压测和部署参数适配。

建议节奏：

```text
LLM DFlash POC 和灰度稳定后，再启动 VLM 专项。
VLM 适配不阻塞第一批 LLM 行业模型落地。
```

## 7. 两人分工建议

实际落地可以按两个人推进，但不建议让算法训练负责人承担所有数据、训练、评测和工程化工作。更合理的分工是：

```text
A：算法策略/训练效果负责人
B：推理性能/算法工程负责人
```

其中 A 更关注“训什么、怎么训、效果是否变好”，B 更关注“如何把训练和评测流程工程化、如何稳定部署并跑出性能”。

### 7.1 总体分工

| 负责人 | 核心职责 | 主要交付物 |
| --- | --- | --- |
| A：算法策略/训练效果负责人 | 数据策略、样本分桶、训练方案、K 值选择、接受率分析、业务效果判断、badcase 分析 | 数据配比方案、训练配置、DFlash head 选择结论、接受率报告、效果评测结论 |
| B：推理性能/算法工程负责人 | 数据处理脚本、hidden states 采集流水线、训练/评测脚本工程化、vLLM-Ascend 部署、NPU 性能调优、图模式配置、灰度和监控 | 数据处理工具、训练/评测流水线、部署脚本、推理参数、性能报告、上线和回滚方案 |

### 7.2 按工作流分工

| 工作流 | 主负责人 | 协同人 | 说明 |
| --- | --- | --- | --- |
| A：目标场景与基线确定 | B | A | B 负责部署和性能基线，A 负责业务场景、评测集和效果基线 |
| B：训练数据集建设 | B | A | B 负责数据抽取、清洗脚本、hidden states 采集；A 负责数据筛选标准、分桶策略和数据配比 |
| C：DFlash 训练实验 | A | B | A 负责训练方案、K 值和 checkpoint 选择；B 负责训练脚本、任务提交、日志整理和加载验证 |
| D：长文本专项能力 | A | B | A 负责 RAG/Agent/长文本样本设计和效果判断；B 负责长文本数据生成工具、window/skip 策略和长文本压测 |
| E：推理性能优化 | B | A | B 负责 `max_num_batched_tokens`、图模式、capture sizes、NPU 算子；A 负责接受率、K 值和场景开关策略 |
| F：离线评测体系 | B | A | B 负责自动评测脚本、benchmark 和报表生成；A 负责指标定义、效果判定和 badcase 分类 |
| G：上线灰度与回滚 | B | A | B 负责部署、灰度、监控、回滚；A 负责线上质量观察和数据回流筛选 |
| H：监控与数据回流 | B | A | B 负责监控指标采集和数据回流链路；A 负责回流数据筛选、再训练策略和版本选择 |
| I：多模型规模化复制 | B | A | B 负责模板化工具和流水线；A 负责不同模型/业务的数据策略和效果验收 |
| J：多模态 VLM 训练适配 | A | B | A 负责 VLM 数据 schema、hidden states 采集位置、训练目标和效果评测；B 负责脚本工程化和推理验证 |

### 7.3 P0 阶段两人任务清单

第一阶段建议只做最小闭环，不要铺太大。

| 优先级 | 任务 | 负责人 | 产出 |
| --- | --- | --- | --- |
| P0 | 固定目标模型和业务场景 | A+B | POC 场景说明 |
| P0 | 建立无投机 / MTP / DFlash 性能基线 | B | 性能基线报告 |
| P0 | 定义 POC 数据筛选标准和分桶 | A | 数据策略说明 |
| P0 | 准备 1万~5万条 POC 数据处理脚本 | B | POC 数据集和处理脚本 |
| P0 | 采集 hidden states 并校验格式 | B | hidden states 数据 |
| P0 | 训练第一版 DFlash head 并选择 checkpoint | A | DFlash head v0 |
| P0 | 接入 vLLM-Ascend 推理 | B | 可运行部署脚本 |
| P0 | 验证 `max_num_batched_tokens=16384`、图模式和 capture sizes | B | 推荐推理参数 |
| P0 | 统计平均接受长度、accept@k 和分桶接受率 | B | 自动接受率报表 |
| P0 | 分析接受率和选择 K 值 | A | 接受率分析结论 |
| P0 | 对比 DFlash vs MTP | B | 性能对比报告 |
| P0 | 验证业务效果无退化 | A | 效果评测报告 |

### 7.4 P1 阶段两人任务清单

| 优先级 | 任务 | 负责人 | 产出 |
| --- | --- | --- | --- |
| P1 | 设计 10万~50万条业务数据配比 | A | 业务数据配比方案 |
| P1 | 扩展数据处理和 hidden states 采集流水线 | B | 业务训练数据集 |
| P1 | 做 K=4/8/15 对比实验 | A | K 值选择报告 |
| P1 | 自动化训练任务和日志汇总 | B | 训练流水线 |
| P1 | 增加 RAG/Agent/长文本专项数据策略 | A | 专项数据设计 |
| P1 | 实现专项数据构造和评测脚本 | B | 专项数据集和评测脚本 |
| P1 | 优化长文本 DFlash 策略 | B | window/skip 策略 |
| P1 | 建立一键压测脚本 | B | benchmark 脚本 |
| P1 | 建立分桶接受率报表 | B | accept@k 分桶报告 |
| P1 | 分析分桶接受率和 badcase | A | 训练改进建议 |
| P1 | 小流量或影子流量验证 | B | 灰度报告 |
| P1 | 梳理 VLM DFlash 训练适配方案 | A | VLM 适配设计文档 |

### 7.5 P2 阶段两人任务清单

| 优先级 | 任务 | 负责人 | 产出 |
| --- | --- | --- | --- |
| P2 | 设计线上数据回流筛选规则 | A | 回流数据策略 |
| P2 | 建设周期性线上数据回流链路 | B | 增量训练数据 |
| P2 | DFlash head 周期增训和版本选择 | A | 新版本 head |
| P2 | 自动化 head 评测和版本登记 | B | head registry / eval report |
| P2 | NPU 算子和 precompute 优化 | B | 性能 patch / 优化报告 |
| P2 | 多业务 route 动态开关 | B | 动态策略配置 |
| P2 | 多模型复制的数据策略 | A | 多模型训练建议 |
| P2 | 多模型复制工具模板 | B | 标准训练和部署模板 |
| P2 | VLM 多模态数据 schema 和训练目标定义 | A | VLM 数据和训练方案 |
| P2 | VLM hidden states 采集与训练脚本适配 | A | VLM DFlash 训练 POC |
| P2 | VLM 推理加载与性能验证 | B | VLM 推理 smoke test / 性能报告 |

### 7.6 两人协作节奏

建议按周推进：

| 周期 | A：算法策略/训练效果负责人 | B：推理性能/算法工程负责人 | 周产出 |
| --- | --- | --- | --- |
| 第 1 周 | 定义 POC 场景、数据标准、评测集 | 建立无投机/MTP/DFlash 基线，准备数据处理脚本 | POC 基线和数据样本 |
| 第 2 周 | 确定训练配置并选择 v0 checkpoint | 跑通 hidden states 采集、训练脚本、DFlash 部署 | v0 可用版本 |
| 第 3 周 | 做 K 值、数据配比、接受率和 badcase 分析 | 做高并发、长文本、图模式压测，生成自动报表 | 训练+推理对比报告 |
| 第 4 周 | 补 RAG/Agent/长文本数据策略 | 做专项数据工具、灰度脚本、监控和回滚 | 小流量灰度准备 |
| LLM 稳定后 | 启动 VLM 数据 schema、hidden states 采集位置和训练目标设计 | 配合 VLM 脚本工程化、推理加载和性能验证 | VLM 适配 POC |

判断是否进入灰度的最小标准：

```text
1. DFlash 效果不低于目标模型原始输出。
2. DFlash 性能不低于 MTP。
3. 平均接受长度稳定达到 3~4 或更高。
4. 高并发 TTFT 不异常。
5. 训练、评测、部署流程可以由脚本复现。
6. 可以一键关闭 DFlash 回退到 MTP 或无投机。
```

## 8. 优先级建议

### P0 必须完成

- 固定目标场景和模型。
- 建立无投机 / MTP / DFlash 性能基线。
- 构建第一版业务训练数据。
- 训练第一版 DFlash head。
- 验证 `max_num_batched_tokens=16384`、图模式、capture sizes。
- 完成离线效果一致性验证。

### P1 强烈建议完成

- RAG / Agent / 长文本专项数据。
- 分桶接受率评测。
- 动态跳过策略。
- 线上影子流量。
- 监控 dashboard。
- VLM 训练适配方案设计：明确多模态数据 schema、hidden states 采集位置、训练目标和评测口径，由 A 主责。

### P2 持续优化

- DFlash NPU 算子优化。
- 多业务 head。
- 自动数据回流。
- 周期增训。
- 多模型复制。
- VLM DFlash 训练适配：在 LLM 流程稳定后，补齐多模态 VLM 的训练、推理加载和效果评测。

## 9. 里程碑

### M1：单模型 POC

目标：

- 选定 1 个行业增训模型。
- 训练 1 个 DFlash head。
- 跑通 NPU 推理。
- 接受长度达到 `3~4`。
- 性能不低于 MTP。

当前状态已经基本达到。

### M2：行业数据增强训练

目标：

- 引入真实业务数据。
- 提升平均接受长度。
- 分桶统计 RAG、Agent、长文本接受率。
- DFlash 性能超过 MTP。

### M3：长文本专项优化

目标：

- 覆盖 Agent、RAG、长上下文问答。
- 增加长文本训练样本。
- 完成 threshold skip / window 策略。
- 长文本 TTFT 不劣化。

### M4：生产灰度

目标：

- 线上影子流量。
- 小流量灰度。
- 支持动态降级。
- 完成监控和报警。

### M5：规模化复制

目标：

- 形成标准训练和部署模板。
- 支持多个行业模型。
- 支持多业务 route 的 DFlash head。
- 建立持续数据回流和增量训练机制。

### M6：多模态 VLM 训练适配

目标：

- 明确 VLM 多模态输入、processor、tokenizer、position ids、attention mask 的训练链路。
- 明确 DFlash head 应对齐哪些 hidden states，尤其是图文混合序列中视觉 token、文本 token、回答 token 的边界。
- 完成小规模 VLM DFlash 训练 POC。
- 完成 VLM DFlash head 推理加载 smoke test。
- 建立 VLM 场景下的效果、接受率、TTFT、TPOT 评测口径。

分工：

- A 主责：VLM 数据 schema、训练目标、hidden states 采集位置、样本过滤、效果评测和 checkpoint 选择。
- B 协助：采集脚本工程化、训练脚本适配、推理加载验证、性能压测和部署参数建议。

说明：

- 当前训练流程暂时只支持 LLM，不支持 VLM。
- VLM 适配不阻塞第一阶段 LLM 行业模型落地，但需要作为后续专项能力建设。

## 10. 最终判断

当前实验已经说明：在 `Qwen3.6-35B-A3B` 上，DFlash 即使平均接受长度只有 `3~4`，推理性能也已经基本持平 MTP。

这说明 DFlash 不是只有在极高接受率下才有价值。对于行业增训模型，只要进一步利用领域数据提升接受率，就有很大概率获得超过 MTP 的推理收益。

因此，DFlash 投机解码方案具备在行业增训大模型中落地的明确价值。后续重点应从“功能验证”转向：

```text
领域数据训练 -> 接受率提升 -> 长文本能力补齐 -> NPU 推理机制优化 -> 灰度上线 -> 持续增量训练 -> VLM 训练适配
```
