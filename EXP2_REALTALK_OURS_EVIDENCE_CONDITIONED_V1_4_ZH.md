# REALTALK Ours Evidence-Conditioned V1.4

## 目的

V1.4 是在 V1.3 Ca 开发集扩大到 24 条后进行的一次定向修正。V1.3 的 24 条候选全部被 Appendix C Judge 判定为 Grounding，且 13 条为错误增加；Reflectiveness 有 11 条错误增加、0 条漏掉。主要偏差是 Decision 将下一句预测误写成了维持和深化关系的策略。

## 唯一方法改动

- Decision 明确执行下一话语预测，而不是关系优化或对话延续最大化。
- 追问、情绪解释、赞扬、自我更新和延长对话不再是默认成分；只有人物证据和当前义务支持时才选择。
- Actor 只自然落实该意图，不自行把消息变得更暖、更长或更适合继续交流。
- 不在生成 Prompt 中出现八项指标或 Judge 标签。

## 严格控制变量

- 模型仍为 `deepseek-v4-flash`，thinking 关闭，解码参数不变。
- 数据、Ca Session 1-2 建模与 Session 3 因果预测点、24 个样本 ID、完整历史和 Schema 不变。
- 使用 `--profile-source-dir` 读取并重新校验 V1.3 的 `self_domains.json` 与 `user_domains.json`；不重新调用 Self/User Domain。
- V1.4 只重新调用 24 次 Decision 和 24 次 Actor。
- 新 manifest 保存来源协议、来源 run signature、Self/User Domain SHA256 和人物清单。

## 比较规则

- V1.3 与 V1.4 使用相同 24 个 result ID 和 Ground Truth。
- 先计算八项指标，再拆分原 6 条、新增 18 条和累计 24 条。
- GPT 三项使用 REALTALK Appendix C Prompt；相同 Ground Truth 标签从 V1.3 checkpoint 复用。
- 24 条仍是 Ca 内部开发诊断，不作为论文 Table 2 正式结果。

## 停止规则

如果 V1.4 没有同时降低错误 Grounding、错误 Reflectiveness 和过度 Empathy，不继续扩大数据；保存结果并重新分析。若方向一致，再讨论是否扩到 Ca 68 条或进入冻结后的 Cb 小门槛。
