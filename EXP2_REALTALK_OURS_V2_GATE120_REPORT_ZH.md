# REALTALK Ours V2 Gate 120 报告

## 结论

V2 已从 Gate 60 扩展到 Gate 120，并完成 120 条同 ID 配对 Judge。生成阶段 120/120 成功、0 failures，但 Gate 120 未通过，因此不继续到完整 519 条。

Gate 120 的结果比 Gate 60 更稳定，但没有改变核心判断：V2 能改善表面内容与情感分类，不能稳定改善 Grounding、Reflectiveness 和 Empathy AD。

## 实验身份

- 协议：`realtalk_task1_ours_behavior_conditioned_v2`
- 模型：`deepseek-v4-flash`
- thinking：关闭
- 样本：120 条，10 位人物、3 个 Session、每个 Session 4 条
- V2 运行目录：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v2-gate12-25bb53d`
- 配对评估目录：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v2-gate120-eval-r11`
- 参考：完整 V9 519 条结果与完整 V9 Judge checkpoint

## Gate 120 配对结果

| 指标 | V9 | V2 | 变化 | 判断 |
|---|---:|---:|---:|---|
| ROUGE | 0.195 | 0.201 | +0.006 | 略改善 |
| BERTScore | 0.867 | 0.866 | -0.001 | 基本持平 |
| Sentiment | 0.675 | 0.742 | +0.067 | 改善 |
| Emotion | 0.617 | 0.642 | +0.025 | 略改善 |
| Intimacy AD | 0.069 | 0.068 | -0.001 原始误差 | 略改善 |
| Reflectiveness | 0.800 | 0.783 | -0.017 | 下降 |
| Grounding | 0.625 | 0.592 | -0.033 | 下降 |
| Empathy AD | 1.008 | 1.083 | +0.075 原始误差 | 恶化 |

AD 越低越好；上表展示原始绝对误差。

## 与论文 Table 2

| 指标 | 论文逐列最优 | V2 Gate 120 | 是否超过论文最优 |
|---|---:|---:|---|
| ROUGE ↑ | 0.140 | 0.201 | 是 |
| BERTScore ↑ | 0.780 | 0.866 | 是 |
| Reflectiveness ↑ | 0.770 | 0.783 | 是，但优势很小 |
| Grounding ↑ | 0.620 | 0.592 | 否 |
| Sentiment ↑ | 0.590 | 0.742 | 是 |
| Emotion ↑ | 0.460 | 0.642 | 是 |
| Intimacy AD ↓ | 0.060 | 0.068 | 否；但优于论文微调行的 0.070 |
| Empathy AD ↓ | 1.240 | 1.083 | 是 |

V2 在 Gate 120 为 6/8 指标超过论文逐列最优，但 Grounding 和 Intimacy AD 仍未超过。更关键的是，V2 相对同样本 V9 的 Grounding、Reflectiveness 和 Empathy AD 都下降。

## Gate 24、60、120 趋势

| 指标 | Gate 24 变化 | Gate 60 变化 | Gate 120 变化 |
|---|---:|---:|---:|
| ROUGE | +0.046 | +0.017 | +0.006 |
| BERTScore | +0.010 | +0.001 | -0.001 |
| Sentiment | +0.208 | +0.083 | +0.067 |
| Emotion | 0.000 | +0.083 | +0.025 |
| Intimacy AD | +0.012 变差 | -0.002 改善 | -0.001 改善 |
| Reflectiveness | -0.083 | -0.017 | -0.017 |
| Grounding | -0.042 | -0.067 | -0.033 |
| Empathy AD | +0.167 变差 | +0.350 变差 | +0.075 变差 |

小规模 Gate 24 的高分明显有样本偏差；扩展后 ROUGE/BERTScore 优势逐渐消失，而 Grounding 和 Empathy AD 的退化仍然存在。

## 生成审计

- 120/120 生成成功。
- unresolved/failures：0。
- 气泡分布：单气泡 104 条、双气泡 14 条、三气泡 1 条、四气泡 1 条。
- 生成问题率：0.45。
- lambda_trace 均值约 0.601，范围 0.0 到 0.9。
- 没有读取未来消息、Ground Truth 或 Judge 分数。
- 没有将模型输出回灌到历史。

## 结论边界

V2 不应继续直接跑完整 519 条，因为 Gate 120 已经显示：它不是全面优于 V9 的版本。下一版应修改 Decision 层的交互义务与情绪/关系适配，尤其是 Grounding 和 Empathy AD；V2 的 Prompt、Schema、生成和评估结果全部保留作为探索性对照。

