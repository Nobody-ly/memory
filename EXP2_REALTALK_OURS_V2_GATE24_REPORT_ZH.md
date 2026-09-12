# REALTALK Ours V2 Gate 24 报告

## 结论

V2 的 24 条配对评估已经完成，但没有通过预设 Gate，因此不扩展到 60 条，也不把这版作为最终结果。

这次结果是同一批 24 个样本、同一份 Ground Truth、同一套本地指标和 GPT Judge 的配对比较。V2 生成阶段没有 unresolved，也没有发现 JSON、内部策略或身份泄漏。

## 实验身份

- 协议：`realtalk_task1_ours_behavior_conditioned_v2`
- 代码分支：`paper-boost/exp2-v2-behavior-conditioned`
- 生成模型：`deepseek-v4-flash`
- Generation thinking：关闭
- 评估样本：Gate 24，覆盖 6 位人物、3 个 Session
- 评估方式：V9 与 V2 使用相同样本和真实目标消息
- V2 运行目录：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v2-gate12-25bb53d`
- 配对评估目录：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v2-gate24-eval-25bb53d-r9`
- 说明：评估脚本沿用了旧的 `v15` 文件命名，表中的 V15 实际指本次 V2 候选，不代表 V15 方法。

## 配对结果

| 指标 | V9 | V2 | 变化 | 方向判断 |
|---|---:|---:|---:|---|
| ROUGE | 0.263 | 0.309 | +0.046 | 改善 |
| BERTScore | 0.875 | 0.885 | +0.010 | 改善 |
| Sentiment | 0.583 | 0.792 | +0.208 | 改善 |
| Emotion | 0.667 | 0.667 | +0.000 | 持平 |
| Intimacy AD | 0.058 | 0.071 | +0.012 绝对误差 | 变差 |
| Reflectiveness | 0.958 | 0.875 | -0.083 | 变差 |
| Grounding | 0.708 | 0.667 | -0.042 | 变差 |
| Empathy AD | 0.333 | 0.500 | +0.167 绝对误差 | 变差 |

这里 AD 指绝对误差，越低越好。报告脚本的 signed improvement 已将 AD 反向处理，但上表直接展示原始绝对误差变化，避免误读。

## 与论文 Table 2 的参照

| 方法 | ROUGE | BERTScore | Reflectiveness | Grounding | Sentiment | Emotion | Intimacy AD | Empathy AD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 w/o fine-tune | 0.14 | 0.76 | 0.62 | 0.40 | 0.53 | 0.43 | 0.06 | 1.80 |
| 论文 w/ fine-tune | 0.14 | 0.78 | 0.77 | 0.62 | 0.59 | 0.46 | 0.07 | 1.24 |
| V9 matched Gate 24 | 0.263 | 0.875 | 0.958 | 0.708 | 0.583 | 0.667 | 0.058 | 0.333 |
| V2 matched Gate 24 | 0.309 | 0.885 | 0.875 | 0.667 | 0.792 | 0.667 | 0.071 | 0.500 |

这只是小规模、6 人的 protocol-aligned exploratory comparison，不是完整 519 条的最终论文级结论。

## 结果解释

V2 的行为条件化设计确实让回复更贴近表面内容，因此 ROUGE、BERTScore 和 Sentiment 上升；但它同时把部分自然的目标人物表达压缩成了“回答当前内容”，导致：

1. 对真实答案中的自我反思和个人表达跟随不足，Reflectiveness 下降。
2. 行为策略中 `question_policy=none` 仍不能可靠阻止模型追加问题，说明 Decision 到 Actor 的控制仍有软约束冲突。
3. 统一的行为条件描述仍没有充分恢复目标人物在不同语境下的亲密度变化，Intimacy AD 和 Empathy AD 变差。
4. 不能据此断言 User Domain 或 Self Domain 本身错误；当前 Gate 太小，而且主要改变了 Decision/Actor 接口，尚未隔离画像质量与表达策略的贡献。

## 按计划执行的检查

- Gate 12：生成成功，覆盖 Session 1/2/3，0 unresolved。
- Gate 24：生成成功，24/24，0 unresolved。
- V2 与 V9 的样本 ID 已通过位置映射逐条对齐。
- Ground Truth 已逐条校验一致。
- 本地指标和 GPT Judge 均使用同一批样本。
- BERTScore 使用现有论文评价实现的 `roberta-large` 配置；论文没有公开 checkpoint revision，因此 manifest 记录本地版本。
- 没有运行 Gate 60，避免在未通过版本上继续扩大。

## 下一步边界

本版本冻结保存，不删除也不覆盖。下一版应优先回到 Decision 层，保留 V2 对内容贴近性的收益，同时放开由 Self Domain 和当前历史决定的自我表达/反思机会；不能只继续调整 Actor 表面措辞。下一版仍应从新的小 Gate 开始，不能把 Gate 24 的失败结果直接拼入新的正式结果。

