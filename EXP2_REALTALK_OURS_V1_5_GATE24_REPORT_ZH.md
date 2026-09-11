# REALTALK Ours V1.5.1 Gate 24 结果

## 结论

V1.5.1 完成了 Gate 6 和 Gate 24 的生成与配对评价，但 Gate 24 未通过，按渐进计划停止，不运行 Gate 68、连续困难窗口或完整 519 条。

本版本的结构目标已经生效：Gate 24 为 24/24 成功、0 unresolved；`lambda_trace` 出现 `0.3/0.4/0.6` 三个值；输出包含 1 气泡和 2 气泡。问题不在结构合同是否执行，而在结构控制改变了回复内容后，整体任务指标发生了退化。

## 24 条配对结果

基线是同一批 24 个 ID 的既有 V1.4 结果，V1.5.1 使用相同 Ground Truth 和 Appendix C Judge。

| 指标 | V1.4 | V1.5.1 | 变化 |
|---|---:|---:|---:|
| ROUGE | 0.159 | 0.136 | -0.023 |
| BERTScore | 0.856 | 0.850 | -0.006 |
| Reflectiveness | 0.833 | 0.542 | -0.292 |
| Grounding | 0.500 | 0.583 | +0.083 |
| Sentiment | 0.750 | 0.792 | +0.042 |
| Emotion | 0.583 | 0.625 | +0.042 |
| Intimacy AD | 0.063 | 0.085 | +0.021（恶化） |
| Empathy AD | 0.625 | 1.042 | +0.417（恶化） |

V1.5.1 Gate 24 的停止原因：Reflectiveness 下降超过 `0.05`，Intimacy AD 恶化超过 `0.01`，Empathy AD 恶化超过 `0.15`。因此不应把 V1.5.1 作为候选最终结果。

## 结构审计

- Gate 6：6/6，0 unresolved；λ 为 `0.3/0.4/0.6`，1 气泡 3 条、2 气泡 3 条。
- Gate 24：24/24，0 unresolved；1 气泡 7 条、2 气泡 17 条；平均 λ `0.416667`；提问率 `0.541667`。
- 生成阶段未启用 thinking、候选搜索、语义 Verification、重写或模型替换。
- Judge 使用 REALTALK Appendix C Prompt；BERTScore 使用论文指标对应的 `roberta-large` CPU 配置。

## 解释

V1.5.1 证明了“动态 λ + 气泡结构计划 + Actor 合同”可以在工程上运行，但 Gate 24 显示它可能把原本自然的目标人物表达过度结构化：Reflectiveness 和 Empathy 下降，而 Grounding 小幅上升。下一版不能直接继续扩大样本，也不能只放宽 Actor 合同；应回到 Decision/Turn Plan 的内容分配，先分析逐样本失败类型，再设计新的独立 V1.5.x。

## 产物

- 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-5-1-ca-dev6-4aa7f1f`
- 评价：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-5-1-ca-dev24-eval-4aa7f1f`
- 远端代码提交：`4aa7f1f`

本结果是小规模、协议对齐的诊断，不是完整 Table 2 主结果，也不支持宣称 V1.5.1 优于论文。
