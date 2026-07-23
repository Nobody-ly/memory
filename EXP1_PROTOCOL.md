# Exp1 实验协议与运行说明

## 任务

Exp1 在已经看到当前用户消息的前提下，比较三种方法对用户当前状态的理解：

1. `self_model`：只使用真实对话历史和当前消息。
2. `flat_profile`：额外使用从训练对话生成的扁平用户画像。
3. `explicit_model`：额外使用从同一训练对话生成的五层用户画像。

三种方法使用相同的目标消息、历史、模型和严格输出 Schema。Exp1 不使用
Bayesian 更新、current state、memory、RAG、typed context block 或新增 baseline。

## REALTALK 对齐方式

- 按论文 Table 8 为每位用户固定一段训练对话 `Ca` 和一段测试对话 `Cb`。
- `Ca` 和 `Cb` 都按时间取最早 3 个 session。
- 连续的同说话人消息先合并为一个语义 turn。
- Flat 与五层画像都只由 `Ca` 的 3 个 session 生成，输入语料完全相同。
- 画像在整段 `Cb` 测试中固定，不使用 `Cb` 更新；在线更新留给 Exp4。
- `Cb` 使用消息级滚动测试：每个该用户的合并消息都是一个测试点。
- 每个目标消息的历史是该 `Cb` 片段中它之前的全部真实合并 turn。
- 上一个目标消息会进入下一个测试点的历史，当前目标和未来消息不会泄露。
- Exp1 与论文生成任务唯一必要的区别：论文预测不可见的下一消息，Exp1
  已看到当前消息并判断它表达的当前状态。

默认上下文上限是 60,000 字符。超限时只从最旧的完整 turn 开始删除，并记录
`context_truncated=true`；设为 `0` 可关闭上限。当前 10 位用户的默认 3-session
数据共 519 个测试点，最大历史约 20,328 字符，均不会触发截断。

## 指标

保留与论文对应的数据维度，但按照当前实验约定统一使用同配置 Kimi 对真实目标消息
生成参考状态：

- Emotion Accuracy：Kimi 严格枚举标签。
- Sentiment Accuracy：Kimi 严格枚举标签。
- Reflectiveness Accuracy：按 REALTALK 定义进行严格布尔判断。
- Grounding Accuracy：按 REALTALK 定义进行严格布尔判断。
- Intimacy Absolute Difference：Kimi 输出 0 到 1 的参考值。
- Empathy Absolute Difference：EPITOME 三项得分之和的绝对差。

参考状态与三种方法使用同一个 Kimi 配置，但参考状态只读取真实历史和真实目标消息，
不读取任何一种方法的画像或预测结果。这样不需要下载额外权重，也不会对某个方法提供
额外信息。

Exp1 还保留 Emotion/Sentiment Macro-F1，便于检查类别不均衡。Topic
Consistency 是原 Exp1 的扩展指标，只作探索分析，不参与主排名和配对显著性结论。
ROUGE 与 BERTScore 不适用，因为 Exp1 不生成下一条消息。

结果同时保存：

- `speaker_macro`：先对每位用户计算，再对用户平均，作为论文主表默认聚合。
- `micro`：把所有目标消息放在一起计算，作为补充。
- 完整标签、预测、混淆矩阵、EI 原始值与逐样本绝对差，方便离线重算。

## 运行

```powershell
uv sync
python -m unittest discover -s tests -v

python -m src.experiments.exp1_user_understanding `
  --speakers Emi `
  --max-eval-points 2 `
  --output-dir data/exp1_realtalk_smoke_emi_2
```

全量运行：

```powershell
python -m src.experiments.exp1_user_understanding `
  --profile-sessions 3 `
  --test-sessions 3 `
  --max-eval-points 0 `
  --output-dir data/exp1_realtalk_full
```

`--max-eval-points` 是每位用户的上限，`0` 表示测试全部合并消息。
`--profile-max-tokens` 只用于避免画像 JSON 被模型截断，不会对已经完整生成的画像
做定长裁剪。

## 输出与恢复

- `results.jsonl`：每个完整三方法 triplet 一行。
- `metric_records.jsonl`：逐样本逐方法长表，包含后续统计所需原始值。
- `summary.json`：macro/micro 指标、分类明细、提升量和配对结果。
- `run_manifest.json`：commit、源码哈希、Kimi 模型、Schema 和运行配置。
- `checkpoint.json`：原子调用缓存与恢复状态。

网络失败由 LLM 客户端最多重试 6 次，逻辑或 Schema 失败最多重试 3 次。失败测试点
记录为 `excluded_incomplete_triplet`，不会记成 0 分。恢复运行只补失败操作，不重复
调用和统计成功结果。

已有结果可离线重算，不产生 API 调用：

```powershell
python -m src.experiments.exp1_recompute_metrics `
  data/exp1_realtalk_full/results.jsonl
```
