# REALTALK Ours V3.1.1 Gate 30 阶段报告

日期：2026-09-15  
实现提交：`0c6ef58`  
配对评价工具提交：`fd2f20c`  
协议：`realtalk_task1_ours_behavior_calibrated_v3_1_1`  
模型：`deepseek-v4-flash`，全阶段 thinking=false  
数据：REALTALK Table 8，前三个连续 Session，Gate 30 为十位人物各三个确定性样本  
范围：`protocol-aligned exploratory diagnostic`，不是完整 Table 2 主结果

## 运行产物

- 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v3-1-1-gate30-0c6ef58`
- 配对评价：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v3-1-1-gate30-judge-0c6ef58`
- Canonical V9 predictions SHA256：`ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81`
- V3/V9 配对：按 `(speaker, target_turn_id)` 对齐，30/30 Ground Truth 逐条一致；原始 V3 结果未改写。

## 完整性

- 生成：`30/30`，零 unresolved。
- Judge：`30/30`，零 unresolved；V9 reference 标签复用既有 checkpoint，V3 候选侧使用 Appendix C `gpt-4o-mini`。
- 历史压缩：关闭；历史截断：关闭；生成结果回灌：关闭；Ground Truth 不进入生成。
- 结构分布：十位人物、三个 Session 均覆盖；lambda_trace 在本 Gate 取值范围 `0.0–0.9`。
- 本地 BERTScore：沿用仓库实现，`roberta-large`、英文、BERTScore `0.3.13`，不涉及训练。

## 配对结果

| 指标 | 论文 w/o fine-tune | 论文 w/ fine-tune | V9 matched 30 | V3.1.1 30 | V3.1.1 - V9 |
|---|---:|---:|---:|---:|---:|
| ROUGE-L | 0.14 | 0.14 | 0.255 | **0.291** | +0.036 |
| BERTScore | 0.76 | 0.78 | 0.878 | **0.885** | +0.006 |
| Reflectiveness | 0.62 | 0.77 | 0.933 | **0.967** | +0.033 |
| Grounding | 0.40 | 0.62 | 0.700 | **0.833** | +0.133 |
| Sentiment | 0.53 | 0.59 | 0.733 | **0.833** | +0.100 |
| Emotion | 0.43 | 0.46 | 0.767 | 0.733 | -0.033 |
| Intimacy AD | 0.06 | 0.07 | 0.060 | 0.072 | +0.012（变差） |
| Empathy AD | 1.80 | 1.24 | 0.500 | 0.533 | +0.033（变差） |

AD 指标越低越好，因此 V3 的正向方向统一另行解释。V3 相对论文最优行：Emotion 仍高于论文 `0.46`，Empathy AD 仍低于论文 `1.24`；Intimacy AD 仅略高于论文 `0.07`。

## Gate 判断

按 V3 计划中 Gate 30 的“Reflectiveness、Grounding、Intimacy AD 至少两项相对 V9 改善”规则，V3 改善了前两项，但 Intimacy AD 恶化 `0.0118`。标准配对报告的更严格检查因此标记为未通过。由于 Gate 30 只有每人每 Session 一条，不能据此修改 Prompt 或宣称完整性能；当前应把 Intimacy 风险记录下来，在进入 Gate 60 前进行稳定性确认或明确停止。

## 解释

这轮结果支持 V3 的结构改动确实改善了反思识别和 grounding 行为，但没有证明亲密度已解决。Gate 30 中开场样本占比较高，真实消息长度和多气泡结构差异也会放大 Intimacy AD 波动。后续不得根据这 30 条反复调 Prompt；若继续，应按固定 V3.1.1 配置扩大到 Gate 60，并单独报告亲密度，不读取 Judge 标签进入生成。
