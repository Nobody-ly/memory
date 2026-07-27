# Exp2 实验协议（未来状态预测）

## 这项实验测什么

Exp2 判断：在还没看到用户下一条消息时，长期画像和显式状态加工能不能更准确地预测用户下一轮的状态。

它和 Exp1 的区别只有目标是否可见：

- Exp1：已经看到当前用户消息，理解这条消息的状态。
- Exp2：看不到目标用户消息，只能根据之前的真实对话预测它的状态。

## 数据怎么用

- 按 REALTALK Table 8 为每位说话人确定 Ca 和 Cb。
- Ca 时间最早的 3 个 session 用于生成一份固定五层画像。
- Cb 时间最早的 3 个 session 用于测试。
- 连续的同说话人消息先合并成一个语义 turn。
- Cb 中每条合并后的目标用户消息都是一个测试点。
- 每个测试点只能读取目标消息之前的真实消息。
- 测试期间不更新画像，避免把 Exp4 的在线更新机制混入 Exp2。

## 四组对比

1. `llm_only`：只读取最近一次已经观察到的用户消息。
2. `dialogue_history`：读取目标之前的完整滚动历史。
3. `user_profile`：完整历史 + 固定完整五层画像。
4. `full_framework`：与 User Profile 相同的原始信息，再增加由历史中最后一条已观察用户消息生成的结构化当前状态。

Full 使用的 current state 只由过去生成，不读取目标消息或目标标签。因此 User Profile 与 Full 的原始信息相同，差别是 Full 多了一次显式状态加工。

## 提示词和输出

- 保留原预测 System Prompt 的核心任务描述。
- 删除不同方法专属的倾向性提示。
- 标签、字段和 JSON 格式只通过 API 级严格 Schema 约束。
- 画像生成提示词原有的结构要求保留。
- 完整画像和上下文不做固定字符切片；只有总上下文超过配置上限时，才删除最旧完整 turn。

## 评价指标

论文对齐的主指标：

- Future Emotion Accuracy / Macro-F1
- Future Sentiment Accuracy / Macro-F1
- Future Intimacy Absolute Difference

论文对齐的辅助指标：

- Future Reflectiveness Accuracy
- Future Grounding Accuracy
- Future Empathy Absolute Difference

Exp2 扩展指标：

- Future Topic Consistency，仅作探索性分析，不参与主要排名。

所有指标分别展示，不再计算没有论文依据的加权总分。论文主表使用 speaker-level macro average，同时保留 micro average 和逐样本原始结果。

ROUGE 与 BERTScore 属于文本生成指标，不用于这一结构化状态预测阶段。后续回复生成阶段必须保存真实回复和生成回复，再单独计算。

## 可恢复运行

- Profile、参考标注、历史状态和四种预测均按输入哈希缓存。
- 一个测试点必须四组全部成功才进入汇总。
- 每完成一个测试点立即写 checkpoint。
- 网络重试由 `LLMClient` 处理，Schema 或解析失败由 operation checkpoint 重试。
- 输出包括 `results.jsonl`、`metric_records.jsonl`、`summary.json`、`run_manifest.json` 和 `checkpoint.json`。
