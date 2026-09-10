# REALTALK Ours Evidence-Conditioned Ca Dev24 报告

## 结论

V1.3 从 6 条扩展到 24 条后，确认原 6 条不能代表整体：模型把下一句预测系统性地做成了关系经营。24/24 个候选都被 Judge 判为 Grounding，且 11 条错误增加 Reflectiveness。

V1.4 只校正 Decision/Actor 的任务目标，不更换模型、不重做 Self Domain 或 User Domain。相同 24 条上，八项指标中六项改善、一项持平、一项下降。错误 Reflectiveness 从 11 条降到 4 条，Empathy AD 从 1.583 降到 0.625，说明“预测实际下一句而非优化关系”的改动有效。

V1.4 仍有两个未解决点：Grounding 为 0.500，低于论文微调行的 0.620；Intimacy AD 为 0.063，略差于论文逐列最佳 0.060。当前 24 条只是 6 位人物、3 个 Ca 对话的内部开发诊断，不能作为 Table 2 正式结果。

## 控制变量

- 数据：同一组 Ca Session 3 的 24 个 result ID。
- 人物：Emi、Paola、Nicolas、Nebraas、Fahim Khan、Muhhamed。
- 模型：`deepseek-v4-flash`，thinking 关闭。
- V1.4 复用 V1.3 的 Self/User Domain；只重新调用 24 次 Decision 和 24 次 Actor。
- Self Domain SHA256：`6119df7c8bd0a15b33bcd676b72756d0f7c34f19dac58be302bb3cb0280c80cb`。
- User Domain SHA256：`9024f9a5d375774d4f800dbfa42715450b47d48b99e4fde3ab4d2accf2d9047a`。
- V1.3 predictions SHA256：`2b2f8a88a8d31db68eea7862f43707cf05eeda6bb850d48a80d6c56ce5ad37bb`。
- V1.4 predictions SHA256：`fe58ada3b2f3fa0f4ce86303eb1493826bc8452fc7bf6525821a104d7e627529`。
- GPT 指标使用 REALTALK Appendix C Prompt 和 `gpt-4o-mini`；V1.4 复用同 ID 的 72 个 Ground Truth 判断。

## 分阶段结果

表内为 speaker macro mean。`Intimacy AD` 和 `Empathy AD` 越低越好，其余越高越好。

| 子集 | 版本 | ROUGE | BERTScore | Reflect. | Grounding | Sentiment | Emotion | Intimacy AD | Empathy AD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 6 条 | V1.3 | 0.123 | 0.848 | 0.667 | 0.333 | 1.000 | 0.667 | 0.088 | 1.667 |
| 原 6 条 | V1.4 | 0.142 | 0.855 | 1.000 | 0.667 | 1.000 | 0.833 | 0.072 | 0.167 |
| 新 18 条 | V1.3 | 0.139 | 0.844 | 0.500 | 0.500 | 0.667 | 0.667 | 0.072 | 1.556 |
| 新 18 条 | V1.4 | 0.165 | 0.856 | 0.778 | 0.444 | 0.667 | 0.500 | 0.060 | 0.778 |
| 全 24 条 | V1.3 | 0.135 | 0.845 | 0.542 | 0.458 | 0.750 | 0.667 | 0.076 | 1.583 |
| 全 24 条 | V1.4 | **0.159** | **0.856** | **0.833** | **0.500** | **0.750** | 0.583 | **0.063** | **0.625** |

## 与论文 Table 2 并列参考

论文两行是完整 Cb 协议；V1.3/V1.4 是 Ca Dev24，下面只能用于方向判断，不能声称直接超过论文。

| 方法 | ROUGE | BERTScore | Reflect. | Grounding | Sentiment | Emotion | Intimacy AD | Empathy AD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| REALTALK w/o FT | 0.140 | 0.760 | 0.620 | 0.400 | 0.530 | 0.430 | **0.060** | 1.800 |
| REALTALK w/ FT | 0.140 | 0.780 | 0.770 | **0.620** | 0.590 | 0.460 | 0.070 | 1.240 |
| Ours V1.3 Ca Dev24 | 0.135 | 0.845 | 0.542 | 0.458 | 0.750 | 0.667 | 0.076 | 1.583 |
| Ours V1.4 Ca Dev24 | **0.159** | **0.856** | **0.833** | 0.500 | **0.750** | **0.583** | 0.063 | **0.625** |

## 错误结构

| 版本 | Reflect TN | Reflect TP | Reflect FP | Reflect FN | Ground TN | Ground TP | Ground FP | Ground FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1.3 | 11 | 2 | 11 | 0 | 0 | 11 | 13 | 0 |
| V1.4 | 18 | 2 | 4 | 0 | 3 | 9 | 10 | 2 |

V1.4 的主要收益来自减少不必要的心理解释、赞扬、延展和过度共情。Grounding 的问题从“全部都问/落地”变成了更正常的精度与召回权衡，但仍有 10 条错误增加和 2 条漏掉。Emotion 在新增 18 条上下降，后续扩容必须观察它是否持续。

## BERTScore 说明

REALTALK 论文明确使用 BERTScore，但没有公开 package 版本、checkpoint revision 和完整计算脚本。当前实现不是自创指标，而是标准英文配置：`bert-score 0.3.13`、`roberta-large`、第 17 层、`idf=false`、`rescale_with_baseline=false`。因此指标种类与论文一致，具体运行版本属于本复现中已固定并披露的实现选择。

## 下一步

冻结 V1.4，不再利用这 24 条调 Prompt。建议下一步将相同 V1.4 扩到完整 Ca Dev68，先验证 Grounding、Emotion 和 Intimacy 的稳定性；只有方向保持，再进入 Cb 的小规模冻结测试。V1.4 若在 Dev68 明显回落，应回到错误结构分析，而不是继续添加局部规则。

## 远端产物

- V1.3 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-3-ca-dev6-60f856b`
- V1.3 Dev24 评测：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-3-ca-dev24-eval-v2-19bdf65`
- V1.4 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-4-ca-dev24-8fb8064`
- V1.4 Dev24 评测：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-4-ca-dev24-eval-8fb8064`
