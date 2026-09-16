# REALTALK Ours V3.3 条件化行为决策计划

## 目标

V3.2.1 已完成结构稳定性验证，但在十人物200条上相对V9：

- Reflectiveness：0.710 -> 0.680
- Grounding：0.650 -> 0.585
- Empathy AD：1.215 -> 1.245

因此 V3.3 不再继续扩大 V3.2.1，也不直接运行完整519条。目标是解决：

> 同一个策略在不同人物和不同对话位置下，错误地产生或遗漏回问、自我披露和情绪回应。

## 不变项

- REALTALK Table 8、Ca/Cb 划分、前三个连续 Session、完整真实历史不变。
- 五层 User Domain 不变：Core、Regulation、Cognition、Identity、Behavior。
- Self Domain、User State、动态 lambda、Behavior Policy、Response Actor 继续存在。
- 模型固定 `deepseek-v4-flash`，thinking关闭；不微调、不换模型、不使用GPU。
- Judge继续使用现有论文 Appendix C Prompt、`gpt-4o-mini`端点和八项指标。
- V9、V3.2.1以及全部历史结果只读，不覆盖、不合并成新的主结果。
- 不在 Cb 测试答案上逐条调 Prompt；V3.3 的行为规则先由 Ca 内部开发确定。

## V3.3 核心改变

### 1. 从固定动作改为条件化 Turn Controller

当前 V3.2.1 已经有行为统计，但决策仍容易把“当前伙伴动作”直接映射成统一回答或追问。
V3.3 将人物行为先验拆为条件统计：

- partner_direct_question：对方直接提问时，人物回答、回问、自我披露的比例；
- partner_disclosure：对方分享信息时，人物回应、附和、自我披露的比例；
- partner_affect：对方表达情绪时，人物回应和反思的比例；
- topic_continuation：话题延续时，人物继续话题、切换话题和提问的比例；
- session_opening / conversation_closure：开场和结束时的特殊行为。

每个统计项至少保存：样本数、比例、消息长度、问句率、自我披露率、反思率和多气泡率。
样本数不足时使用人物总体统计；不强行制造人物偏好。

### 2. Decision 输出改成“动作概率 + 本轮选择原因”

不把概率直接当作生成答案，也不让模型自由输出多个候选答案。
Decision 只输出一个最终计划：

```json
{
  "situation": {
    "partner_act": "direct_question|disclosure|affect|continuation|opening|closure|other",
    "topic": "...",
    "affect": "none|positive|negative|mixed|uncertain",
    "uncertainty": "low|medium|high"
  },
  "behavior_evidence": {
    "matched_condition": "partner_direct_question",
    "sample_count": 6,
    "usual_actions": ["answer", "self_disclose"],
    "confidence": 0.72
  },
  "alignment": {
    "orientation": "self_led|balanced|partner_adaptive",
    "lambda_trace": 0.48,
    "adaptation_source": "current_partner_turn",
    "affected_dimensions": ["turn_composition"]
  },
  "turn_plan": {
    "primary_action": "answer|acknowledge|self_disclose|clarify|follow_up|topic_shift|close",
    "secondary_action": "none|answer|acknowledge|self_disclose|follow_up",
    "question_mode": "none|clarifying|reciprocal|follow_up",
    "reflection_mode": "none|state_only|brief",
    "disclosure_mode": "none|surface|specific",
    "message_shape": "single|multi",
    "length_band": "short|typical|long",
    "content_slots": ["..."]
  }
}
```

`behavior_evidence` 只允许引用 Ca 统计 ID 和当前可见历史 ID，不能引用 Cb 答案或未来内容。
`lambda_trace` 是同一次 Decision 中的审计变量，必须影响 `turn_plan`，不单独调用模型。

### 3. 不强制“必须问”或“禁止问”

V3.3 采用三层决策：

1. 当前交流义务：直接问题优先回答，对方情绪优先回应，对方分享优先接住；
2. 人物条件统计：判断这个人物在此类位置通常是否回问或自我披露；
3. 当前 Cb 已出现行为：若已有足够当前伙伴证据且与 Ca 冲突，优先当前 Cb 行为。

最终只生成一个 `turn_plan`。如果统计置信度低，选择最小必要动作，不额外编造追问。
如果当前问题需要对方补充信息，才允许 `clarifying`；普通闲聊不自动追问。

### 4. Empathy 仅作状态输入，不作统一风格约束

User State 保留，但只描述当前对方的显式状态、互动需求和不确定性。
不把所有普通消息都解释成需要安慰，也不把 empathy 直接写成“必须温暖”。

- 明确负面情绪：允许回应情绪，再完成当前交流义务；
- 中性信息交换：优先自然回答、分享或继续话题；
- 情绪不确定：不做心理诊断，不强加反思；
- 目标人物没有证据表现反思时，不因 User State 自动增加反思。

### 5. Actor 只执行最终计划

Actor 继续看到完整历史、Self Domain 的身份/表达视图、User State 和唯一 `turn_plan`。
不看到完整 User Domain、Judge 指标、Ground Truth 或 λ 理由。

Actor 合同：

- 完成 `primary_action`；
- 只有 `secondary_action` 非 `none` 时才添加第二个同话题成分；
- 只有 `question_mode` 不是 `none` 时才能提问；
- 只有 `reflection_mode` 不是 `none` 时才能简短反思；
- 不自动把“回答”扩展成追问，也不自动加入共情话术；
- 允许真人式近况、观点和主动话题，但必须与 Self Domain 一致。

## Ca 内部开发

不使用 Cb Ground Truth 调 Prompt。

1. 从每位人物 Ca Session 1-2 建立临时 Self Domain 和条件行为统计；
2. 用 Ca Session 3 构造预测点；
3. 运行30条 Ca 开发样本，10人每人3条；
4. 人工检查五类场景：直接提问、伙伴分享、情绪消息、普通延续、开场/结束；
5. 只检查动作时机、人物风格、事实归属和是否机械追问；
6. Prompt、Schema、阈值冻结后，才进入 Cb。

Ca 开发输出不进入 Table 2，也不与 V9 分数混合。

## Cb 渐进 Gate

所有 Cb 样本从真实源顺序确定性取样，不按分数挑选：

| Gate | 数据 | 目的 |
|---|---:|---|
| 6 | 2人×3条 | 检查 Schema、动作权限和因果边界 |
| 30 | 10人×3条 | 首次全人物比较 |
| 60 | 10人×6条 | 检查跨人物稳定性 |
| 120 | 10人×12条 | 判断是否值得扩展 |

Gate 6 和30只允许修正格式和明显逻辑错误；Gate 30 后冻结 Prompt。
Gate 60/120 不再修改 Prompt。每个 Gate 使用相同样本时只补新增记录，不重复生成。

## 通过标准

相对同 ID V9，Gate 30 不是最终结论，只做结构检查。

Gate 60：

- Grounding 不得下降超过0.03；
- Reflectiveness 不得下降超过0.03；
- Empathy AD 不得恶化超过0.15；
- 其余五项不得有两项下降超过0.02；
- 至少6/10人物没有同时损失两个目标指标。

Gate 120：

- Grounding、Reflectiveness、Empathy AD 三项至少两项改善；
- 三项不能全部比V9差；
- 退步不能集中在同一人物群体；
- 所有结构错误和未解决错误为零。

任一 Gate 不通过即停止当前 V3.3，不继续519条，不在同一 Cb 样本上反复试错。

## 当前决定

先实现统计提取、V3.3 Schema、Ca 开发输入和 Actor 合同测试；通过静态审查后再运行 Ca 30条。
V3.2.1 的200条结果保持为冻结诊断基线，不回滚、不删除、不改名。
V3.3 先验证“条件化行为选择”是否解决跨人物 Grounding 和 Reflectiveness 退步，
验证前不做完整519条。
