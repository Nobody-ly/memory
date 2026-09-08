# REALTALK V14 连续困难窗口压力测试

## 1. 目的与选择规则

本测试不把分散错误样本重新拼接。它按照冻结 V9 `predictions.jsonl` 的原始 519 条顺序，
扫描连续窗口，并只使用 V9 的 Reflectiveness、Grounding 和 Intimacy AD 结果确定困难区间。
V14 输出不参与选样。

固定窗口为 V9 第 373–432 条，共 60 条：

- 首条：`vanessa:message_45:session_2:turn_15`
- 末条：`vanessa:message_104:session_3:turn_54`
- 人物：Vanessa
- Session 2：32 条
- Session 3：28 条

每条预测仍读取该目标点之前的完整真实历史；窗口起点不截断模型上下文。

## 2. 协议不变量

- V14.12 Prompt 和 Schema 哈希与冻结版本相同；
- 模型：`deepseek-v4-flash`，thinking 关闭；
- Self Domain、User Domain 和 V9 语义动作合同冻结；
- 不读取 Ground Truth、Cb future 或 Judge 标签；
- 完整历史，无压缩、无截断；
- 仅新增连续窗口选择入口，实现提交：
  `868349799abba65cc2db5214a09aeab01850dca7`。

生成 60/60、Judge 360/360，均为零 unresolved。预测文件 SHA256：
`68e48db1b2cfa525ff983546eac0c9a98f7dc57f7290e4cfc13785bd7fd535bc`。

## 3. 连续困难 60 条结果

| 指标 | 论文逐列最优 | V9 同 60 条 | V14.12 | V14.12 - V9 |
|---|---:|---:|---:|---:|
| ROUGE | 0.140 | 0.107 | 0.121 | +0.014 |
| BERTScore | 0.780 | 0.843 | 0.843 | -0.001 |
| Reflectiveness | 0.770 | 0.650 | 0.667 | +0.017 |
| Grounding | 0.620 | 0.517 | 0.567 | +0.050 |
| Sentiment | 0.590 | 0.333 | 0.450 | +0.117 |
| Emotion | 0.460 | 0.300 | 0.283 | -0.017 |
| Intimacy AD（低优） | 0.060 | 0.0817 | 0.0850 | +0.0033 |
| Empathy AD（低优） | 1.240 | 1.317 | 1.333 | +0.017 |

V14.12 在该连续困难段改善了 ROUGE、Reflectiveness、Grounding 和 Sentiment，但没有把
Reflectiveness、Grounding 或 Sentiment 提升到论文最优线；Emotion、Intimacy AD 和
Empathy AD 略微恶化。

配对变化：

- Reflectiveness：修正 6 条 V9 错误，同时改坏 5 条 V9 正确样本，净增 1 条；
- Grounding：修正 8 条 V9 错误，同时改坏 5 条，净增 3 条；
- Intimacy AD：31 条改善、29 条恶化，但恶化幅度更大，导致平均值变差。

## 4. 最差连续 30 条复核

上述窗口内 V9 第 378–407 条是最差的连续 30 条。无需额外生成，直接从同一结果切片：

| 指标 | V9 | V14.12 | 变化 |
|---|---:|---:|---:|
| ROUGE | 0.101 | 0.126 | +0.026 |
| Reflectiveness | 0.500 | 0.700 | +0.200 |
| Grounding | 0.567 | 0.667 | +0.100 |
| Emotion | 0.367 | 0.233 | -0.133 |
| Intimacy AD（低优） | 0.1005 | 0.1044 | +0.0039 |
| Empathy AD（低优） | 1.233 | 1.267 | +0.033 |

V14 的行为结构对最差局部的 Reflectiveness、Grounding 和文字匹配有明显作用，但没有解决
情绪与亲密度校准。

## 5. 原问题逐项复查

### 输出单元与多气泡

数据单元保持为论文规则下合并连续同角色气泡后的目标消息，换行保留原始气泡。该困难段：

- Ground Truth 多气泡：27/60；
- V9 多气泡：0/60；
- V14 多气泡：15/60；
- V14 Actor 对 Decision 气泡计划执行：60/60。

因此 Actor 的多气泡能力已经恢复，但 Decision 仍低估困难段的多气泡需求，问题只解决了一半。

### 动态 lambda

V14 该段 lambda 均值 0.288，范围 0.2–0.6；12 条 partner-adaptive、48 条 self-led。
相对 V9 约 0.035 且几乎全 self-led 已恢复动态变化，但当前分布是否最优仍无直接监督标签。

### Reflectiveness 与 Grounding 时机

两个指标都有净修复，但修正和回退同时存在。问题已从“完全缺少结构能力”缩小为 Decision
对具体轮次的动作判断仍不够稳定。问题权限执行为 56/60，仍有 4 条 Actor 偏离计划。

### Intimacy

统一升温或降温仍不成立。V14 在约一半样本改善、另一半恶化，最终平均 AD 更高；下一版应
根据 Vanessa 在当前 Cb 已发生历史中的关系语气和消息位置做条件校准，而不是增加统一温暖度。

## 6. 当前结论

V14 的 Ca 行为案例库、Message Bundle 和动态 lambda 方向有效，确实修复了连续困难段中的
部分 Reflectiveness、Grounding、Sentiment 和多气泡结构。但现版本仍不能取代 V9：

- 多气泡数量由 Decision 低估；
- Reflectiveness/Grounding 修正率和回退率接近；
- Intimacy 与 Emotion 没有改善；
- 该窗口只有 Vanessa，属于人物级压力测试，不是 Table 2 主结果。

下一轮优化应围绕“条件化气泡数量、反思/提问时机、当前 Cb 关系语气”修改 Decision，保留
Actor 的多气泡执行能力与现有 Ca 检索，不应继续统一增强反思、追问或温暖表达。

## 7. 服务器产物

- 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-v9-worst-contiguous60-v1-8683497`
- 本地指标：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-v9-worst-contiguous60-local-v1`
- GPT Judge：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-v9-worst-contiguous60-judge-v1`
