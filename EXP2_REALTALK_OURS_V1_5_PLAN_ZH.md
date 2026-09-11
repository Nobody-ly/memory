# REALTALK Ours V1.5 实施计划

## 1. 目标

V1.5 针对 V1.4 在连续困难窗口暴露的三个问题进行一次受控改进：

1. Grounding 判断位置不准：问号率接近真实值，但很多问题出现在错误轮次。
2. 真实目标消息有多气泡，V1.4 仍全部生成成单段消息。
3. `lambda_trace` 在困难窗口中恒定为 `0.3`，没有形成有效的动态权衡。

V1.5 的目标是提高 Grounding，同时保留 V1.4 已经获得的 Reflectiveness、Intimacy 和 Empathy 改善。

V1.5 不以 V9 的生成文本为输入，不使用 Judge 分数选择样本，不读取 Cb 未来消息，不训练或微调模型。

## 2. 冻结内容

以下内容保持与 V1.4 一致：

- REALTALK Table 8 的 Ca/Cb 划分；
- Ca 和 Cb 使用前三个连续 Session；
- 预测点之前的完整真实历史；
- 不压缩、不截断、不检索替代完整历史；
- `deepseek-v4-flash`；
- thinking 关闭；
- Self Domain 的输入范围和五层 User Domain；
- Generation 的纯文本输出；
- REALTALK Appendix C 的三个 GPT Judge 指标；
- ROUGE、BERTScore、Sentiment、Emotion、Intimacy AD 的现有实现；
- V1.4、V1.3、V9 的代码和产物保持不变。

V1.5 必须使用新协议名和新输出目录，不覆盖任何旧结果。

## 3. V1.5 的核心修改

### 3.1 Decision 输出消息结构

保留当前 Situation、Alignment 和唯一 Behavior Policy，但增加结构规划：

```json
{
  "situation": {
    "partner_act": "question|statement|disclosure|reaction|closure|unclear",
    "current_topic": "short description",
    "conversational_obligation": "answer|acknowledge|react|share|ask|close|none",
    "uncertainty": "low|medium|high"
  },
  "alignment": {
    "orientation": "self-led|balanced|partner-adaptive",
    "lambda_trace": 0.0,
    "basis": "short evidence-based explanation",
    "affected_dimensions": ["questioning|length|tone|self_disclosure|topic|bubble_count"]
  },
  "turn_plan": {
    "bubble_count": 1,
    "units": [
      {
        "act": "answer|acknowledge|react|self_disclose|explain|follow_up|topic_shift|close",
        "content_slot": "what this unit should cover",
        "question_allowed": false,
        "self_disclosure_allowed": false
      }
    ],
    "relationship_tone": "casual|warm|close|neutral|playful|serious",
    "length_band": "short|typical|long"
  }
}
```

约束：

- `bubble_count` 为 1、2 或 3；
- `units` 数量必须等于 `bubble_count`；
- `follow_up` 是唯一默认允许生成问题的动作；
- `question_allowed=false` 时不得生成问句；
- `self_disclosure_allowed=false` 时不得主动增加人物近况；
- 每个单元必须有明确内容槽位；
- 不加入名为 Reflectiveness、Grounding、Intimacy 或 Empathy 的控制字段；
- 不把评价指标直接写入 Prompt。

### 3.2 Grounding 决策规则

Decision 需要按以下顺序判断：

1. 当前伙伴最后一条消息是否提出了明确问题；
2. 目标人物是否需要回答该问题才能自然完成当前轮次；
3. 目标人物在 Ca 中是否通常会在此类位置提问；
4. 当前对话是否已经存在一个未完成的信息槽位；
5. 只有满足前述证据时才生成 `follow_up`。

不能因为“继续交流更自然”就自动追加问题。目标人物也可以只回答、只反应、只分享，或者用短消息结束当前轮次。

### 3.3 多气泡结构

Self Domain 中增加可迁移的行为统计，或在已有 Self Domain 的基础上由程序确定性计算：

- 目标人物在 Ca 中的多气泡比例；
- 每个多气泡消息的平均气泡数；
- 目标人物常见的动作顺序；
- 不同 Session、不同伙伴消息类型下的气泡数；
- 目标人物的消息长度分布。

Decision 根据当前场景和人物 Ca 统计选择 1、2 或 3 个气泡。不能对所有样本统一使用 2 或 3 个气泡。

Actor 必须生成与 `bubble_count` 相同数量的非空消息，以换行分隔。每个气泡对应一个 `unit`，不得把多个 Unit 压缩成一段，也不得自行增加额外动作。

### 3.4 λ 的实现

`lambda_trace` 是可审计的动态权衡记录，不直接拼接文本。

其判断依据至少包括：

- 当前伙伴消息是否需要直接回应；
- 当前伙伴是否主动披露情绪或信息；
- 目标人物原有行为与当前场景是否冲突；
- 当前关系语气；
- User Domain 是否存在与当前话题相关的证据；
- Cb 中目标人物已经表现出的行为是否偏离 Ca。

要求：

- λ 不得在整个测试窗口中恒定为同一个值；
- λ 必须影响 `turn_plan` 的至少一个字段；
- `affected_dimensions` 必须与实际计划变化一致；
- λ 不表示共情分数、正确率或 User Domain 置信度；
- 不能用人工固定区间强行制造 λ 的变化。

如果模型仍然输出恒定 λ，记录为失败信号，不通过该 Gate。

## 4. Prompt 设计

Decision 的核心任务固定为：

> 根据目标人物在 Ca 中观察到的表达习惯、当前 Cb 历史和当前伙伴消息，预测这个人在此刻最可能自然产生的下一条消息结构。

Prompt 必须强调：

- 预测具体下一轮行为，而不是优化关系；
- 不默认追问；
- 不默认增加安慰、解释、自我披露或话题延展；
- 直接问题优先回答；
- 真实人物可以只发一句短消息；
- 消息可能包含多个连续气泡；
- 旧的 Ca 活动和计划不能自动变成当前事实；
- Cb 已发生的真实行为可以调整表达习惯，但不能读取未来。

Actor 的 System Prompt 保持人物身份模拟：

```text
You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain.
Follow the private turn plan naturally.
Produce exactly the planned number of non-empty message bubbles.
Output only the message bubbles, separated by newlines.
Do not add a speaker name or explanation.
```

Actor 不看到完整 User Domain、λ 数值、Judge 指标或 Ground Truth，只看到与当前回复有关的 Self Domain、真实历史和 `turn_plan`。

## 5. 开发数据与渐进 Gate

### Gate 1：6 条

用途：检查代码、Schema、历史边界和 Actor 合同。

必须覆盖：

- 至少 4 位人物；
- 至少 2 个 Session 位置；
- 至少 1 条明确问题；
- 至少 1 条非问题消息；
- 至少 1 条 Ca 中存在多气泡行为的人物。

要求：

- 6/6 成功；
- zero unresolved；
- `bubble_count` 与 `units` 一致；
- Actor 输出气泡数正确；
- 无 JSON、身份标签或策略泄漏；
- λ 至少出现两个不同值；
- 不根据均值判断最终效果。

### Gate 2：24 条

复用 V1.5 Gate 1 的 6 条，再增加 18 条。

覆盖 6 位人物、不同 Session 位置、问题和非问题轮次、多气泡与单气泡行为。

评价八项指标，并与同一 ID 的 V1.4 配对：

- Grounding 不低于 V1.4；
- Reflectiveness 不低于 V1.4 超过 0.05；
- Intimacy AD 不恶化超过 0.01；
- Empathy AD 不恶化超过 0.15；
- 其余指标最多允许一项下降超过 0.03。

如果 Grounding 上升但 Reflectiveness 和 Intimacy 明显下降，停止该版本。

### Gate 3：68 条

使用 Ca 开发集完整 68 条，不进入正式 Cb Table 2。

要求：

- 68/68 成功，zero unresolved；
- λ 不恒定；
- 多气泡率不再为 0%；
- Grounding 至少不低于 V1.4 Dev24；
- Reflectiveness 不低于 V1.4 Dev24；
- Intimacy AD 不恶化超过 0.01；
- 结果不能只由单个人物贡献。

Gate 3 通过后冻结 V1.5 的 Prompt、Schema 和解码参数。

## 6. 冻结验证

### 验证一：V9 困难连续 60 条

使用已固定的 V9 第 373–432 条：

- 不修改 Prompt；
- 不根据单条 Ground Truth 调整；
- 复用原有 crosswalk 和 60 条清单；
- 与 V9、V1.4 使用相同 Judge 参考标签。

通过条件：

- Grounding 高于 V1.4 的 `0.417`；
- Reflectiveness 不低于 V1.4 的 `0.700`；
- Intimacy AD 不高于 V1.4 的 `0.0789`；
- Empathy AD 不高于 V1.4 的 `1.333`；
- 多气泡率高于 0%；
- λ 不恒定。

### 验证二：第二个连续窗口

固定另一个人物的连续窗口，选择规则必须提前写入 manifest：

- 不按 V1.5 得分选择；
- 不按单条 Grounding 错误拼接；
- 保持连续原始顺序；
- 使用同样的 Judge 和本地指标。

两个连续窗口都通过后，才讨论 Gate 120 或完整 519 条。

## 7. 评价与报告

每个 Gate 都保存：

- `predictions.jsonl`；
- `self_domains.json`；
- `user_domains.json`；
- `decision_records.jsonl` 或等价审计字段；
- `manifest.json`；
- `unresolved_errors.jsonl`；
- 本地五项指标；
- Appendix C 三项 Judge 指标；
- 逐人物、逐 Session 和逐样本结果；
- V9/V1.4/V1.5 配对变化；
- λ 分布、问题率、多气泡率、平均字符数和动作顺序分布。

Judge 必须明确记录：

- 参考标签是否复用；
- 候选判断数量；
- 错误数量；
- Judge 模型名和端点；
- Prompt hash；
- 结果是否为诊断子集。

## 8. 停止规则

遇到以下任一情况立即保存并停止当前版本：

- 任一 Gate 出现 unresolved；
- λ 恒定；
- 多气泡仍为 0% 且真实数据存在明显多气泡；
- Grounding 提升超过 0.05，但 Reflectiveness 或 Intimacy 明显下降；
- 结果只在一个人物上改善；
- Actor 频繁不执行 `turn_plan`；
- 发现历史、未来消息、Ground Truth 或 Judge 标签泄漏。

停止后建立 V1.5.x 新目录，从 Gate 1 重新开始。不得在失败 Gate 上继续累加样本调 Prompt。

## 9. 预计产物目录

```text
/amax/xidian_ty/Ly/personaemp-exp2/runs/
  realtalk-evidence-v1-5-ca-dev6-<commit>/
  realtalk-evidence-v1-5-ca-dev24-<commit>/
  realtalk-evidence-v1-5-ca-dev68-<commit>/
  realtalk-evidence-v1-5-v9-hard60-<commit>/
  realtalk-evidence-v1-5-second-contiguous-window-<commit>/
```

每个目录独立保存 checkpoint，旧 V1.3、V1.4、V9 目录不得覆盖。

## 10. 执行顺序

1. 建立 V1.5 分支和新协议名。
2. 修改 Schema、Decision Prompt、Actor Prompt 和确定性合同检查。
3. 运行本地静态测试和泄漏测试。
4. 运行 Ca Gate 6。
5. 运行 Ca Gate 24，完成八项指标。
6. 运行 Ca Gate 68，冻结版本。
7. 运行 V9 连续困难 60 条。
8. 运行第二个预先确定的连续窗口。
9. 两个窗口通过后，再决定是否进入 120 或 519。

本计划阶段不更换模型、不训练、不微调、不运行完整 519 条正式结果。
