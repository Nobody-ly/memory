# REALTALK Ours V3.1.1 Gate 60 阶段报告

日期：2026-09-15  
生成实现提交：`26ce66f`  
协议：`realtalk_task1_ours_behavior_calibrated_v3_1_1`  
模型：`deepseek-v4-flash`，thinking=false  
范围：十位人物、三个 Session、每个 speaker-session 两条，共 60 条

## 产物与完整性

- 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v3-1-1-gate60-26ce66f`
- 配对评价：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v3-1-1-gate60-judge-26ce66f`
- 生成 `60/60`，Judge `60/60`，零 unresolved。
- 通过 `--parent-output` 复用已完成 Gate 30，只新增后 30 条样本的 Decision/Actor；父样本未重新生成。
- V3/V9 按 `(speaker, target_turn_id)` 映射，60/60 Ground Truth 逐条相同；原始运行 ID 和预测未改写。
- 本地五项使用仓库固定实现；GPT 三项使用 REALTALK Appendix C Prompt 和既有 `gpt-4o-mini` 端点。

## 结果

| 指标 | 论文逐列最优 | V9 matched 60 | V3.1.1 60 | V3.1.1 - V9 |
|---|---:|---:|---:|---:|
| ROUGE-L ↑ | 0.14 | 0.228 | **0.241** | +0.013 |
| BERTScore ↑ | 0.78 | 0.873 | **0.874** | +0.002 |
| Reflectiveness ↑ | 0.77 | 0.850 | **0.917** | +0.067 |
| Grounding ↑ | 0.62 | 0.667 | **0.700** | +0.033 |
| Sentiment ↑ | 0.59 | 0.733 | **0.783** | +0.050 |
| Emotion ↑ | 0.46 | 0.683 | **0.717** | +0.033 |
| Intimacy AD ↓ | 0.06 | 0.072 | 0.082 | -0.010 |
| Empathy AD ↓ | 1.24 | 0.650 | 0.850 | -0.200 |

表中 AD 的差值已按“越低越好”解释：负值表示 V3 相对 V9 变差。

## 判定

V3.1.1 相对论文逐列最优为 `7/8` 项更好，唯一未超过的是 Intimacy AD。相对同样本 V9，六项提升，Intimacy AD 与 Empathy AD 变差。

Gate 60 的目标三项中 Reflectiveness 和 Grounding 改善，Intimacy AD 恶化约 `0.010`；其余本地指标全部非劣，但 Empathy AD 恶化 `0.200`，超过计划允许的 `0.150`。自动 Gate 判定为失败，因此按锚点停止，不运行 Gate 120 或完整 519。

## 下一版依据

本轮证明行为条件校准能稳定提升 Reflectiveness 和 Grounding，但当前 Actor/Policy 会把自然回应写得更具情感和关系色彩，造成 Empathy 与 Intimacy 强度偏离真实消息。下一版只能针对“强度校准”建立新版本，并重新从 Gate 6 开始；不得利用当前 60 条 Judge 标签进入生成 Prompt，也不得修改或覆盖 V3.1.1 产物。
