# REALTALK Ours V15：Cb 后验行为控制器

## 1. 实验身份

- 协议：`realtalk_task1_ours_v15_cb_posterior_controller`
- 模型：`deepseek-v4-flash`，Controller 与 Actor 均关闭 thinking
- 基础：冻结 V9 Self Domain、逐样本五层 User Domain 和完整因果历史
- 重建：重新生成 Behavior Controller 与 Response Actor
- 状态：protocol-aligned exploratory comparison

V9、V10–V14 的代码与产物保持不变。V15 不读取 V9 Decision、V9 生成文本、Cb
未来消息、Ground Truth、Judge 标签或论文指标。

## 2. 方法

```text
Self Domain
+ 五层 User Domain（最多激活两条）
+ Ca 聚合行为弱先验
+ 当前点前 Cb 目标人物行为
+ 完整真实历史
        ↓
Behavior Controller
        ↓
动态 lambda + 1–6 个 turn units
        ↓
Response Actor
        ↓
确定性结构检查；仅违约时重试 Actor
```

Ca 只提供消息长度、气泡、问题、自我披露与结构统计，不提供原始相似文本。Cb 统计只读取
当前目标点前已经发生的真实目标人物消息；Cb 证据充分且与 Ca 冲突时优先当前 Cb。

`turn_units` 是 Actor 唯一的动作与问题控制源。lambda 是同次 Controller 决策的可审计轨迹，
不使用固定阈值或公式生成文本。

为避免把普通的“继续说一句”误标为追问，机器 Schema 使用无歧义的
`clarification-question`、`follow-up-question` 和 `reciprocal-question` 标签；这三类单元必须
明确给出提问对象，其他动作不得携带问题。
`topic-shift-statement` 明确表示陈述式换题；通过问题换题仍须使用上述提问动作。User Domain
激活只输出稳定 `fact_id`，程序再解析回原始层级和值，避免模型改写画像事实。

Ca 内部开发的临时 User Domain 为避免服务端长输出截断，每层保留最多 3 条最强事实；三个
Session 的源文本仍逐字完整输入。该限制只用于 Ca Prompt 开发，正式 Cb 继续复用 V9 已冻结的
逐样本完整五层 User Domain。
模型偶尔输出的 `s1:t6` 仅确定性规范为等价源 ID `session_1:turn_6`；原始响应保留审计，
规范化后仍必须通过“只引用已观察伙伴消息”的证据校验。

## 3. 开发与冻结

先进行 Ca 内部开发：

1. Ca Session 1–2 建立临时 Self/User Domain；
2. Ca Session 3 构造因果预测点；
3. 运行 6 条结构预检，再扩展到 30 条、10 位人物；
4. 只允许在 Ca 开发阶段调整 Prompt；
5. Prompt、Schema 和模型参数冻结后才进入 Cb。

Ca 开发结果不进入 Table 2。正式 Cb 使用 V9 基于 Ca 前三个 Session 建立的 Self Domain。

## 4. Cb 渐进 Gate

固定嵌套清单：`6 → 18 → 30 → 60 → 120 → 519`。选样只基于人物、Session 和位置，
不使用文本、V9 错误或 Judge 分数。同一版本只补新增样本。

- Gate 6/18：结构、因果边界、身份、调用可靠性和人工输出检查。
- Gate 30：10 人 × 3 Session × 1 条，首次全覆盖配对评价。
- Gate 60：每个 speaker-session 两条，检查三项目标至少两项改善和其余指标非劣。
- Gate 120：要求三项目标点估计全部改善、至少 7/10 人物方向一致。
- Gate 519：完整 protocol-aligned exploratory Table 2 对比。

进入 Cb 后不得修改 Prompt。任何修改必须升级为 V15.x、新目录并从 Gate 6 重新开始。

## 5. 评价

沿用 REALTALK 八项指标：ROUGE、BERTScore、Reflectiveness、Grounding、Sentiment、
Emotion、Intimacy AD、Empathy AD。GPT 三项复用 Appendix C Prompt，V9/V15 共享 Ground
Truth 判断。

主汇总使用 speaker macro mean 和 population std。配对分析固定 seed `20260909`，按 10 位
speaker 聚类 bootstrap 10,000 次。AD 指标反向后统一以正值表示 V15 改善。

额外诊断：Reflectiveness/Grounding Precision、Recall、F1；Intimacy 有符号误差；逐人物、
逐 Session、Actor 重试率和调用 token。

## 6. 完整目标

| 指标 | V15 完整 519 条目标 |
|---|---:|
| ROUGE | `> 0.14` |
| BERTScore | `> 0.78` |
| Reflectiveness | `> 0.77` |
| Grounding | `> 0.62` |
| Sentiment | `> 0.59` |
| Emotion | `> 0.46` |
| Intimacy AD | `< 0.06` |
| Empathy AD | `< 1.24` |

未达到时保存并如实报告，不继续利用同一 519 条调 Prompt。

## 7. V15.1 实施审计记录

首个冻结 Prompt 版本完成 Ca 30：30/30、零 unresolved、零事实归属警告。随后 Cb Gate 6
完成 6/6，Cb Gate 18 完成 18/18，但人工审计确认 3 条启发式事实归属警告中有 1 条是真实的
伙伴当前事实向目标人物迁移。原实现只记录 warning 而仍生成完成标志，与本计划“仍有归属问题
则 unresolved 并停止 Gate”不一致，因此该 Cb 目录只保留为失败诊断，不能继续 Gate 30。

V15.1 协议升级为 `realtalk_task1_ours_v15_1_cb_posterior_controller`，只做两项修正：

- 事实归属启发式仅检查第一人称陈述句与伙伴最新事实的独特重合，排除问题和无第一人称的
  泛化评论，降低误报；
- 命中后不自动改写语义，直接记录样本 unresolved 并停止当前 Gate。Controller/Actor 同时
  明确禁止把伙伴的地点、天气、活动、计划、健康、工作和物品镜像为目标人物当前事实。

V15.1 必须使用新目录并重新从 Ca Gate 6 开始，不能复用原 V15 的 Cb 生成缓存。
