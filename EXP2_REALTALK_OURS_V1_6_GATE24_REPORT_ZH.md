# REALTALK Ours V1.6 Gate24 报告

## 结论

V1.6 已完成 24 条生成和配对 Judge，但没有通过 Gate24，因此不进入 68 条扩展。
这次结果说明：取消 V1.5.1 的固定气泡数和硬提问权限，单独采用软 `response_guidance`，没有解决当前任务的主要问题。

本轮保留为独立诊断结果；V1.4/V9、V1.5.1 及其他旧运行均未修改。

## 固定配置

- 协议：`realtalk_task1_ours_evidence_conditioned_v1_6`
- 代码提交：`d9738d9`
- 模型：`deepseek-v4-flash`
- Decision：`temperature=0.2`、`top_p=0.9`、`thinking=false`
- Actor：`temperature=0.6`、`top_p=0.9`、`thinking=false`
- 数据：与 V1.4/V9 相同的 24 个配对样本
- Self Domain/User Domain：复用 V1.4 固定画像，避免重复画像调用
- Judge：同一 `gpt-4o-mini` 端点和官方 Appendix C 评价流程
- BERTScore：沿用 REALTALK 评价实现的 `roberta-large` CPU 配置

## 生成检查

- 生成：`24/24`
- unresolved：`0`
- 生成气泡分布：单气泡 10 条，多气泡 14 条
- 旧运行目录未覆盖

## 配对结果

| 指标 | V1.4/V9 | V1.6 | 变化 | 判断 |
|---|---:|---:|---:|---|
| ROUGE | 0.159 | 0.139 | -0.020 | 下降 |
| BERTScore | 0.856 | 0.851 | -0.005 | 下降 |
| Reflectiveness | 0.833 | 0.667 | -0.167 | 下降 |
| Grounding | 0.500 | 0.417 | -0.083 | 下降 |
| Sentiment | 0.750 | 0.750 | +0.000 | 持平 |
| Emotion | 0.583 | 0.583 | +0.000 | 持平 |
| Intimacy AD | 0.063 | 0.080 | +0.017 AD | 变差 |
| Empathy AD | 0.625 | 1.042 | +0.417 AD | 变差 |

AD 指标越低越好。Judge 报告采用配对同一 Ground Truth，并以 6 位人物的 speaker macro mean 汇总。

## 与论文 Table 2 的位置

| 方法 | ROUGE | BERTScore | Reflectiveness | Grounding | Sentiment | Emotion | Intimacy AD | Empathy AD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper w/o fine-tune | 0.14 | 0.76 | 0.62 | 0.40 | 0.53 | 0.43 | 0.06 | 1.80 |
| Paper w/ fine-tune | 0.14 | 0.78 | 0.77 | 0.62 | 0.59 | 0.46 | 0.07 | 1.24 |
| V1.6 matched gate | 0.139 | 0.851 | 0.667 | 0.417 | 0.750 | 0.583 | 0.080 | 1.042 |

V1.6 仅在 ROUGE、BERTScore、Sentiment、Emotion 和 Empathy AD 上达到或超过论文中的部分基准，未达到论文微调行的 Reflectiveness、Grounding、Intimacy AD 要求；同时相对 V1.4/V9 明显下降。因此不能把 V1.6 作为改进版本。

## 产物

远端生成目录：

`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-6-ca-dev24-d9738d9`

远端评估目录：

`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-6-ca-dev24-eval-d9738d9`

配对报告位于评估目录的 `paired_report/paired_report.md`。本轮 Gate 未通过，按计划不运行 Gate68。

## 解释边界

这是 24 条的小规模、协议对齐诊断，不足以证明全量性能；但它足以否定“仅取消硬气泡/问题规则即可改善 V1.5.1”的局部假设。下一步若继续，应改变 Decision/Actor 的信息组织或画像迁移方式，而不是继续在同一 24 条上微调软规则。
