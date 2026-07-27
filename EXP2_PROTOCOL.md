# Exp2 实验协议

## 研究问题

Exp2 检验：更完整的用户理解，是否既能提高下一条用户消息的状态预测，也能帮助智能体生成更贴近真实对话的共情回复。

这不是对 REALTALK 原任务的逐字复现。REALTALK 直接生成下一条真实消息；本实验保留师姐原定的四组对比，并将流程拆成两个阶段：

1. 在目标用户消息不可见时，预测下一条用户消息的状态。
2. 目标用户消息出现后，生成智能体回复，并与真实对话中的下一条回复比较。

## 数据划分

- 沿用 REALTALK Table 8 的说话人级跨对话划分。
- 从训练对话 `Ca` 的前 3 个 session 提取固定五层用户画像。
- 从智能体自己的训练对话提取固定 persona。
- 在测试对话 `Cb` 的前 3 个 session 内按合并消息逐点滚动。
- 连续同一说话人的消息先合并。
- 每个预测点只看目标消息之前的真实历史，不读取目标或未来消息。
- 用户画像在 Exp2 中固定，不做 Bayesian 更新；画像演化留给 Exp4。

## 四组条件

### 预测阶段

| 条件 | 可见信息 |
|---|---|
| LLM Only | 最近一组完整的用户消息和对方回复 |
| Dialogue History | 目标前全部滚动对话历史 |
| User Profile | 对话历史 + 固定五层用户画像 |
| Full Framework | 对话历史 + 同一份画像 + 原仓库状态块 |

原仓库状态块由 `BACKGROUND_REASONING` 生成，包含：

- `current_state`
- `projected_state`
- `activated_persona`

状态只根据上一组已经完成的用户消息和智能体回复更新，并因果继承。第一条测试消息没有已完成交换时，状态为空。

### 回复生成阶段

四组都能看到当前真实用户消息，也都使用同一份智能体 persona。

| 条件 | 额外信息 |
|---|---|
| LLM Only | 无历史、无用户画像 |
| Dialogue History | 目标前历史 |
| User Profile | 历史 + 五层用户画像 |
| Full Framework | 历史 + 五层画像 + 状态块 + 原仓库共情对齐推理 |

这样比较的是用户建模信息带来的增益，而不是某一组额外知道“当前用户说了什么”。

## 指标

### 未来状态预测

主要指标：

- Emotion Accuracy / Macro-F1
- Sentiment Accuracy / Macro-F1
- Intimacy Absolute Difference，越低越好

扩展指标：

- Topic Consistency，仅作探索性记录，不进入主排名

Emotion、Sentiment、Intimacy 的真实标签使用 REALTALK 对应的固定 CardiffNLP 模型。Topic 由评审模型生成参考文本，因此只保留为扩展项。

### 回复生成

与 REALTALK 评价方向对齐：

- ROUGE-L
- Style Similarity（沿用原仓库 `0.6 * ROUGE-L + 0.4 * lexical overlap`）
- BERTScore F1，可在所有回复生成后批量计算
- Reflectiveness Accuracy
- Grounding Accuracy
- Sentiment Accuracy
- Emotion Accuracy
- Intimacy Absolute Difference，越低越好
- Empathy / EPITOME Absolute Difference，越低越好
- EPITOME Component MAE，分别比较 ER / IN / EX 后取平均，越低越好
- EPITOME Vector Accuracy，三个分项完全一致才算正确

同时保存真实回复和生成回复各自的 EPITOME 总分，便于展示与后续重新聚合；总分本身不直接作为“越高越好”的主排名，因为普通寒暄中过强共情并不一定更合适。

所有真实回复、生成回复和逐项标签都保存在 `results.jsonl`，即使暂不运行 BERTScore，后续仍可离线补算。

## 可追溯与恢复

- 每个 LLM 操作按输入、模型、Schema 和提示词哈希缓存。
- 每个完整样本立即写 checkpoint。
- 重试失败不会记为 0 分；不完整样本保留错误记录并排除汇总。
- `results.jsonl` 保存逐样本原始结果。
- `metric_records.jsonl` 保存预测和生成两阶段的长表记录。
- `summary.json` 保存 speaker-macro、micro 和随测试位置变化的趋势。
- `run_manifest.json` 保存数据、模型、提示词、Schema、分类器版本和调用统计。

## 与旧实现的区别

- 旧实现只做了未来状态标签预测，没有执行设计文档要求的回复生成。
- 旧 `current_state` 是上一条用户消息的标签，不是仓库设计的状态块。
- 旧 Future Schema 混入了 EPITOME；现在 EPITOME 只评价生成回复。
- 旧 LLM Only 看不到最近对方回复；现在加入最近完整交换。
- 旧 History 条件会重复注入最近用户消息；现在每条历史只出现一次。
- 旧标答的 Emotion、Sentiment、Intimacy 由同一个 Kimi 评审生成；现在改为 REALTALK 固定分类器。
