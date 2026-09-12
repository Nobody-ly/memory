# REALTALK Ours V2 Gate 60 报告

## 结论

V2 已从 Gate 24 扩展到 60 条，并完成同 ID 配对 Judge。生成阶段 60/60 成功，0 unresolved；但 Gate 60 未通过，因此不继续扩展到 Gate 120 或完整 519 条。

本轮明确说明：V2 改动改善了表面文本贴合、情感分类和亲密度误差，但损害了 Grounding 和 Empathy AD，不能作为 V9 的稳定替代版本。

## 实验身份

- 协议：`realtalk_task1_ours_behavior_conditioned_v2`
- 模型：`deepseek-v4-flash`
- thinking：关闭
- 样本：60 条，10 位人物、3 个 Session、每个 Session 2 条
- V2 运行目录：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v2-gate12-25bb53d`
- 配对评估目录：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v2-gate60-eval-r10`
- 参考：完整 V9 519 条结果与完整 V9 Judge checkpoint
- 评估：V9/V2 使用相同 60 个样本、相同 Ground Truth 和相同 Appendix C Judge

## Gate 60 配对结果

| 指标 | V9 | V2 | 变化 | 解释 |
|---|---:|---:|---:|---|
| ROUGE | 0.228 | 0.245 | +0.017 | 改善 |
| BERTScore | 0.873 | 0.874 | +0.001 | 基本持平 |
| Sentiment | 0.733 | 0.817 | +0.083 | 改善 |
| Emotion | 0.683 | 0.767 | +0.083 | 改善 |
| Intimacy AD | 0.072 | 0.070 | -0.002 原始误差 | 略改善 |
| Reflectiveness | 0.850 | 0.833 | -0.017 | 下降 |
| Grounding | 0.667 | 0.600 | -0.067 | 明显下降 |
| Empathy AD | 0.650 | 1.000 | +0.350 原始误差 | 明显恶化 |

AD 越低越好；表中 AD 直接展示原始绝对误差，避免把方向反转后的 signed improvement 误读成原始分数。

## 与论文 Table 2

| 指标 | 论文逐列最优 | V2 Gate 60 | 是否超过论文最优 |
|---|---:|---:|---|
| ROUGE ↑ | 0.140 | 0.245 | 是 |
| BERTScore ↑ | 0.780 | 0.874 | 是 |
| Reflectiveness ↑ | 0.770 | 0.833 | 是 |
| Grounding ↑ | 0.620 | 0.600 | 否 |
| Sentiment ↑ | 0.590 | 0.817 | 是 |
| Emotion ↑ | 0.460 | 0.767 | 是 |
| Intimacy AD ↓ | 0.060 | 0.070 | 否 |
| Empathy AD ↓ | 1.240 | 1.000 | 是 |

所以 V2 Gate 60 仍有 6/8 指标超过论文逐列最优，但 Grounding 和 Intimacy AD 没有超过。更重要的是，V2 相比同样本 V9 的 Grounding 和 Empathy AD 变差，不能只看论文横向表而忽略配对退化。

## Gate 24 到 Gate 60 的意义

Gate 24 中 V2 的 Reflectiveness 和 Grounding 看起来仍高于论文，但 Gate 60 扩展到全部 10 位人物后，Grounding 降到 0.600，低于论文微调行 0.620。这说明 Gate 24 的结果受人物覆盖不足和样本波动影响，不能作为最终性能判断。

Gate 60 的变化方向比 Gate 24 更有代表性：

- V2 对直接内容和情感标签的贴合变好。
- V2 对“当前伙伴到底说了什么、是否需要回应以及回应是否合适”的定位变差。
- Empathy AD 的恶化说明 V2 不是简单地“更自然”或“更像目标人物”，而是在部分样本中过度执行自己的行为条件，偏离了当前交互需要。

## 生成审计

- 60/60 生成成功。
- unresolved：0。
- 生成消息气泡：单气泡 53 条，双气泡 7 条。
- 生成问题率：0.533333。
- 没有把模型生成结果回灌到后续历史。
- 没有读取未来消息、Ground Truth、Judge 分数。
- 没有使用 Verification、候选搜索或重写。

## 下一步边界

V2 Gate 60 已保存，不继续 Gate 120/519。下一版应优先修正 Decision 层的交互义务和关系适配，而不是继续扩大 V2：

1. 重新检查 `user_state.interaction_need` 与 `behavior_policy.primary_goal` 是否一致。
2. 降低 Self Domain 行为先验对当前伙伴动作的覆盖。
3. 让 Grounding 的要求体现在“回应当前伙伴内容”上，而不是通过增加问题或表面相似度实现。
4. Empathy AD 需要单独检查情绪强度和回应距离，不能只依赖五层 User Domain。

V2 是一版完整、可复核的探索结果，不是最终 Table 2 Ours 行。

