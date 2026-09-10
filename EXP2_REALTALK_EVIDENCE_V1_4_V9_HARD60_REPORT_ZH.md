# REALTALK Evidence-Conditioned V1.4：V9 连续困难 60 条压力测试

## 结论

冻结的 V1.4 在 Ca Dev24 上表现良好，但不能完整迁移到旧 V9 最差的 Vanessa 连续窗口。相对 V9，V1.4 改善了 ROUGE、Reflectiveness、Sentiment、Emotion 和 Intimacy AD，BERTScore 基本持平；Grounding 明显下降，Empathy AD 轻微恶化。

因此 V1.4 不能直接扩大到正式 519 条。它验证了“下一话语预测而非关系优化”有助于减少错误反思和亲密度偏差，但没有解决提问/承接时机、动态 lambda 和多气泡结构。

## 固定数据

- 来源：冻结 V9 `predictions.jsonl` 的第 373–432 条，共 60 条。
- 人物：Vanessa。
- 范围：Session 2 后半段 32 条、Session 3 前半段 28 条。
- 首条：V9 `vanessa:message_45:session_2:turn_15`，V1.4 `cb:Vanessa:session_2:turn_15`。
- 末条：V9 `vanessa:message_104:session_3:turn_54`，V1.4 `cb:Vanessa:session_3:turn_54`。
- `speaker/session/turn_id/ground_truth` 逐项校验：60/60 一致。
- Ground Truth 集合 SHA256：`f73d515d3994c2362d977435467e606f2667107b323af2b9479457799bcb7137`。
- 每条 V1.4 输入仍为预测点前的完整真实历史；窗口起点不截断上下文。

## 协议与执行

- 模型：`deepseek-v4-flash`，thinking 关闭。
- V1.4 Prompt、Schema 和解码保持冻结，不读取 V9 输出、未来消息或 Ground Truth。
- 只新增可审计连续窗口入口；实现提交 `b29ff6528a2752c2da6c62aa67eaeb3fb8e4ec9b`。
- 生成：60/60，123 个阶段操作，零 unresolved。
- Judge：REALTALK Appendix C Prompt，`gpt-4o-mini`。
- V1.4 复用 V9 同一 Ground Truth 的 180 个 Judge 标签，只重新判断候选；360/360 完成、零错误。

## 连续困难 60 条

`Intimacy AD` 和 `Empathy AD` 越低越好，其余越高越好。

| 指标 | 论文逐列最优 | V9 | V14.12 | V1.4 | V1.4 - V9 |
|---|---:|---:|---:|---:|---:|
| ROUGE | 0.140 | 0.107 | **0.121** | 0.117 | +0.011 |
| BERTScore | 0.780 | **0.843** | 0.843 | 0.842 | -0.001 |
| Reflectiveness | 0.770 | 0.650 | 0.667 | **0.700** | +0.050 |
| Grounding | 0.620 | 0.517 | **0.567** | 0.417 | -0.100 |
| Sentiment | 0.590 | 0.333 | **0.450** | **0.450** | +0.117 |
| Emotion | 0.460 | 0.300 | 0.283 | **0.317** | +0.017 |
| Intimacy AD | 0.060 | 0.0817 | 0.0850 | **0.0789** | -0.0027 |
| Empathy AD | 1.240 | **1.317** | 1.333 | 1.333 | +0.017 |

这段只有一个人物，论文数值来自完整协议，不能直接宣称横向优劣。它的用途是验证方法在已知困难连续区域上的稳定性。

## 最差连续 30 条

固定为上述窗口中的第 6–35 条，即原 V9 第 378–407 条。

| 指标 | V9 | V14.12 | V1.4 | V1.4 - V9 |
|---|---:|---:|---:|---:|
| ROUGE | 0.101 | **0.126** | 0.107 | +0.007 |
| Reflectiveness | 0.500 | **0.700** | **0.700** | +0.200 |
| Grounding | 0.567 | **0.667** | 0.467 | -0.100 |
| Sentiment | 0.433 | 未单列 | 0.400 | -0.033 |
| Emotion | **0.367** | 0.233 | 0.300 | -0.067 |
| Intimacy AD | 0.1005 | 0.1044 | **0.0944** | -0.0061 |
| Empathy AD | **1.233** | 1.267 | 1.367 | +0.133 |

V1.4 在最差 30 条上稳定改善 Reflectiveness 和 Intimacy，但 Grounding、Emotion、Empathy 不足，不能称为总体改善。

## 错误结构

### 连续 60 条

| 版本 | Reflect TN | TP | FP | FN | Ground TN | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V9 | 37 | 2 | 13 | 8 | 28 | 3 | 14 | 15 |
| V1.4 | 34 | 8 | 16 | 2 | 16 | 9 | 26 | 9 |

V1.4 对 Reflectiveness 的主要收益来自将漏判从 8 条降到 2 条。Grounding 同时修复了 6 条漏判，却额外制造 12 条错误增加，净准确率下降 0.10。

## 行为统计

| 统计 | Ground Truth | V9 | V1.4 |
|---|---:|---:|---:|
| 含问号 | 28.3% | 15.0% | 31.7% |
| 多气泡 | 45.0% | 0.0% | 0.0% |
| 平均字符 | 84.6 | 72.4 | 114.9 |

V1.4 的问题不是统一少问，而是总体问号率接近真值、逐轮位置却不一致。V1.4 仍把 45% 的真实多气泡 turn 压成单段，并且单段平均长度偏长。

`lambda_trace` 在 60 条上全部为 `0.3`，说明当前 Prompt 中的软权衡记录没有形成真实动态变化。这个现象在该压力测试中应被视为方法实现未充分发挥，而不是实验二本身的正常结果。

## 下一步判断

不建议直接扩大 V1.4 到 519 条，也不建议继续用这 60 条逐句调 Prompt。下一版应在 Ca 开发数据上解决三个可迁移问题：

1. Decision 根据当前轮次而不是“保持交流”判断是否提出问题，降低 Grounding FP。
2. 恢复人物级多气泡/动作组合预测，而不是把内容塞进更长单段。
3. 让 lambda 的变化有明确、可审计的触发依据；若它仍恒定，应简化为隐式权衡而非保留虚假的动态数值。

完成后再以完全相同的 60 条做一次冻结验证，不能继续根据这些 Ground Truth 定制规则。

## 服务器产物

- 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-4-v9-worst-contiguous60-b29ff65`
- 评测：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-4-v9-worst-contiguous60-eval-b29ff65`
- Crosswalk：评测目录下 `crosswalk.json`
- 复用参考标签：评测目录下 `reference_checkpoint.json`
