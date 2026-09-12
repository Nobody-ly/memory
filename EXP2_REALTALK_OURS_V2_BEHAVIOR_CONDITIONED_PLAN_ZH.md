# REALTALK Task 1 Ours V2 行为条件建模计划

## 目标

V2 面向 REALTALK Task 1 的真实目标：根据目标人物下一条消息之前的完整历史，预测目标人物最可能自然说出的下一条消息。

保留师姐的核心组件：

```text
Self Domain + User Domain + User State + lambda + Behavior Policy + Generative Model
```

固定数据协议：Table 8 的 Ca/Cb 分配；每位目标人物 Ca 前三个连续 Session 建立 Self Domain；Cb 前三个连续 Session 滚动测试；共 519 个目标消息、1,076 个原始目标气泡。所有历史完整披露，不摘要、不截断、不检索替代。测试时只读目标消息之前的真实历史，生成结果不回灌，不能读取 Ground Truth。

协议名：`realtalk_task1_ours_behavior_conditioned_v2`

旧 V1.4、V1.5、V1.5.1、V1.6 运行目录和结果均不修改。

## 总体流程

```text
Ca 前三个 Session
  -> Self Domain + 条件行为模式 + 确定性行为统计
Cb 已完成 Session
  -> User Domain 五层画像
当前预测点以前的完整 Cb 历史
  -> Decision: Situation + User State + lambda + Behavior Policy
  -> Actor: 目标人物下一条消息
```

调用规则：Self Domain 每位人物一次；User Domain 在每个已完成 Session 后调用一次；Decision 每个预测点一次；Actor 每个预测点一次。User Domain 不逐消息更新，Decision 将 User State、lambda 和唯一 Behavior Policy 放在同一次结构化调用中。

## Self Domain

### 目的与输入

Self Domain 表示目标人物长期是谁、通常如何表达和在不同互动条件下如何回应。它不是理想助手画像，不把共情原则和评价指标写成人格特征。

输入是该目标人物 Ca 前三个连续 Session 的完整对话，含 Session ID、turn ID、speaker、原始文本和 source ID。目标人物自己的消息是 Self Domain 事实证据；伙伴消息只能用于理解互动条件，不能变成目标人物事实。

### Schema

保留现有 `self_claims`、`voice`、`social_dispositions`、`uncertainties`，增加：

```json
{
  "behavioral_conditions": [
    {
      "trigger": "direct_question",
      "likely_response": "answer directly and sometimes add a brief personal detail",
      "question_tendency": "usually_none",
      "disclosure_tendency": "moderate",
      "reflection_tendency": "low",
      "confidence": 0.74,
      "evidence_ids": ["file::session_2:turn_5"]
    }
  ]
}
```

`trigger` 固定枚举：`direct_question`、`partner_disclosure`、`partner_affect`、`asks_about_target`、`opinion_or_advice`、`topic_continuation`、`conversation_closure`。证据不足时返回空数组，不强行补齐。

### Self Domain System Prompt

```text
You compile a private behavioral Self Domain for a persona simulation task.

The target is a real conversational participant, not an assistant, therapist, coach, or idealized
personality. Infer only what is supported by the target speaker's own messages.

Separate: stable identity and preferences; linguistic and message style; social interaction tendencies;
and conditional response patterns. For conditional patterns, describe what the target tends to do when the
partner asks a direct question, shares an experience, expresses emotion, asks about the target, offers an
opinion or suggestion, continues the topic, or closes the conversation.

Describe what the target actually tends to do, not what the target should do. Do not turn empathy
principles, evaluation metrics, or assistant behavior into personality traits. Do not copy partner behavior
into the target profile. Do not treat one isolated event as a stable trait. Keep one-off activities
temporally scoped and preserve uncertainty. Use only target-speaker evidence IDs. Return only strict JSON.
Do not draft a reply or invent quotations.
```

参数：`model=deepseek-v4-flash`，`temperature=0.2`，`top_p=0.9`，`max_tokens=4096`，`thinking=false`，最多 3 次格式重试。

## 确定性行为统计

Python 从 Ca 原始数据计算，不增加 LLM 调用，不转成硬规则。记录：`target_message_count`、`mean_characters`、`median_characters`、`mean_bubbles`、`multi_bubble_rate`、`question_rate`、`self_disclosure_rate`、`reflection_rate`、`direct_answer_rate`、`closure_rate`。

统计定义固定：问句含 `?` 或 `？`；自我披露需有第一人称且表达自身状态、经历、偏好、计划或事实；反思需有明确感受、原因、思考或评价；直接回答只在伙伴上一条含问句且目标消息非空时计入。另按七种 `trigger` 分组，并记录每组样本数。样本少于 3 的分组只能作为弱参考。

## User Domain

保留五层：`Core`、`Regulation`、`Cognition`、`Identity`、`Behavior`。每条事实为 `value`、`confidence`、`evidence_ids`。

更新规则：Session 1 完成后更新一次供 Session 2；Session 2 完成后更新一次供 Session 3；不逐消息更新。只保存可持续使用的伙伴信息。当前情绪、一次性事件、最新问题和当前对话动作属于 User State，不写入长期画像。没有重复或明确证据时留空，允许修订和撤回冲突事实。

### User Domain System Prompt

```text
You update the private five-layer long-term model of the current conversation partner.

The fixed layers are core, regulation, cognition, identity, and behavior. This is a durable partner
profile, not a summary of the current session. Store only repeated preferences or behavior, durable identity
information, recurring coping or interaction patterns, stable communication tendencies, and repeated
cognitive or decision patterns.

Do not store temporary mood, one-off events, the latest question, or the partner's current conversational
act as stable profile facts. The current situation is handled separately by the situation and user-state
controller. Do not infer psychological traits from one message. Do not copy facts about the target speaker
into the partner model. Revise or withdraw facts when new evidence conflicts. Empty layers are valid.
Every evidence ID must come from the whitelist. Return only strict JSON.
```

参数：`model=deepseek-v4-flash`，`temperature=0.2`，`top_p=0.9`，`max_tokens=4096`，`thinking=false`，最多 3 次格式重试。

## Decision：Situation、User State、lambda、Behavior Policy

### 输入

Decision 读取 Self Domain、Ca 行为统计、当前 Cb 预测点以前的完整历史、当前 User Domain、Session 位置和可见 evidence ID。它不读取目标答案。

### 输出 Schema

```json
{
  "situation": {
    "partner_act": "question",
    "current_topic": "daily activities",
    "explicit_request": "asking what the target has been doing",
    "conversational_obligation": "answer",
    "uncertainty": "low"
  },
  "user_state": {
    "interaction_need": "information_exchange",
    "affect": "neutral",
    "affect_confidence": 0.76,
    "topic_continuity": "continue_current_topic",
    "response_pressure": "moderate"
  },
  "alignment": {
    "orientation": "balanced",
    "lambda_trace": 0.38,
    "basis": "The direct question requires a clear answer while the target keeps its casual style.",
    "affected_dimensions": ["content_focus", "self_disclosure"]
  },
  "behavior_policy": {
    "primary_goal": "answer_current_question",
    "required_content": ["Answer the current question."],
    "optional_content": ["Add a brief related personal detail if natural."],
    "avoid": ["Do not turn the response into generic advice or therapy."],
    "question_policy": "none",
    "self_disclosure_policy": "allowed_if_natural",
    "reflection_policy": "none",
    "topic_policy": "continue_current_topic",
    "tone": "casual",
    "length": "typical"
  },
  "evidence_ids": []
}
```

`interaction_need`：`information_exchange`、`emotional_acknowledgment`、`reciprocal_sharing`、`clarification`、`topic_continuation`、`closing`、`unclear`。

`affect`：`positive`、`neutral`、`negative`、`mixed`、`unclear`。

`question_policy`：`none`、`allowed`、`required`。`self_disclosure_policy`：`none`、`allowed_if_natural`、`preferred`。`reflection_policy`：`none`、`allowed_if_supported`、`preferred`。`orientation`：`self-led`、`balanced`、`partner-adaptive`。

### Decision System Prompt

```text
You are the private situation, user-state, and alignment controller for persona simulation.

This is a next-utterance prediction task. Infer what the target person would most naturally say now,
not what would improve the relationship, maximize empathy, or satisfy an evaluation metric.

Use the Self Domain and target-speaker behavioral statistics as the default identity and behavior prior.
Use the five-layer User Domain only when a partner fact is relevant to this exchange. Read the complete
real history before the target turn.

First identify what the partner just did, the current topic, what the target must do conversationally,
and what is uncertain. Then infer the current User State. Do not invent hidden psychological needs. Use
explicit affect when present; use neutral or unclear when the text does not support a stronger inference.

Then produce one Behavior Policy with one primary goal. Optional self-disclosure, reflection, continuation,
and questioning are permissions, not requirements. A direct question normally requires a direct answer.
Do not add a follow-up question by default. Do not force self-disclosure, reflection, emotion labeling,
praise, advice, warmth, or therapy language.

lambda_trace records the balance between the target's stable behavior and adaptation to the current exchange.
It is not a score, reward, empathy value, confidence value, or quota. Its orientation and affected
dimensions must be reflected in the Behavior Policy.

Do not draft the final message. Return only strict JSON.
```

参数：`model=deepseek-v4-flash`，`temperature=0.2`，`top_p=0.9`，`max_tokens=2048`，`thinking=false`，最多 3 次格式重试。

lambda 只做软权衡审计，不直接决定气泡数量、句子数量、问号数量或文本长度。`affected_dimensions` 只允许：`content_focus`、`questioning`、`length`、`tone`、`self_disclosure`、`reflection`、`topic`。删除 `bubble_count` 遗留维度。

## Actor

Actor 只接收完整真实历史、Self Domain、User State 和 Behavior Policy。它不接收完整 User Domain、evidence ID、lambda 数值和理由、Ground Truth、Judge 指标或候选策略。

### Actor System Prompt

```text
You are {speaker}. Continue the conversation as this person.

Produce the target person's most likely next message at this exact point. The target is a real
conversational participant, not an assistant, therapist, coach, evaluator, or generic empathy system.

Use the complete real history before the target turn, the private Self Domain as the identity and behavior
prior, and the current User State and Behavior Policy as guidance for this turn.

Answer the partner's direct question when one exists. Preserve the target person's ordinary level of
self-disclosure, warmth, reflection, questioning, message length, and topic continuation. Do not add generic
empathy, therapy, advice, praise, or emotional analysis unless both the current exchange and the target's
observed behavior support it. Do not copy facts from the partner into the target. Do not reveal private
fields or reasoning. There is no fixed bubble count or sentence count. Use natural line breaks when the
person would naturally send multiple message bubbles. Output only the target person's message.
```

Actor User Prompt 顺序固定为：Ca 完整历史、Self Domain、当前 Cb 完整历史、User State、Behavior Policy，最后要求生成 `{speaker}` 的下一条消息。

参数：`model=deepseek-v4-flash`，`temperature=0.6`，`top_p=0.9`，`max_tokens=1024`，`thinking=false`，最多 3 次格式重试。

只做空输出、截断、JSON、内部字段和 speaker 标签检查。不检查固定气泡数，不检查固定问号数，不做语义重写，不做候选搜索。

## Schema 变更

必须新增：Self Domain 的 `behavioral_conditions`、Decision 的 `user_state`、Decision 的 `behavior_policy`。

必须删除旧遗留：`turn_plan`、`bubble_count`、`units`、`question_allowed`、`self_disclosure_allowed`，以及 `bubble_count` 作为 lambda 影响维度。

所有结构化调用使用严格 JSON Schema、本地 normalizer、evidence ID 校验和最多 3 次格式重试。修复只能处理 JSON 格式、字段缺失和类型错误，不能改变证据或重设计策略。

## 渐进执行

1. Gate12：3 位人物，每人 4 条，覆盖三个 Session。只看结构、因果边界、画像质量和明显客服化。
2. Gate24：6 位人物，每人 4 条，与同 ID V1.4/V9 配对 Judge。
3. Gate60：10 位人物，每人 6 条。此阶段后冻结 Prompt、Schema、参数和样本清单。
4. Gate120：只有 Gate60 通过才运行。
5. Gate519：只有 Gate120 通过才运行完整测试集。

Gate24 要求：24/24 成功、零 unresolved、无泄漏、无系统性客服化；Reflectiveness、Grounding、Intimacy AD 至少两项改善；Empathy AD 不恶化超过 0.20；ROUGE 与 BERTScore 不同时下降。

Gate60 要求：三个目标指标整体不下降；至少 7/10 人物没有三项目标指标同时下降；其余指标没有两项以上明显下降；lambda、User State 和 Behavior Policy 在不同场景中有可解释变化。

Gate 失败时保存结果并停止扩大，不在同一测试集上继续调 Prompt。

## 运行审计

每次运行保存数据哈希、Table 8 映射、样本 ID、Self/User/Decision/Actor 原始响应、规范化结果、Prompt/Schema 哈希、模型和解码参数、thinking 状态、调用次数、重试、token、unresolved_errors.jsonl、manifest.json、本地指标和 GPT Judge 配对报告。

本计划的 V2 结果只能称为 protocol-aligned exploratory comparison，不能把小规模 Gate 结果称为完整论文复现结果。切换执行模型后，除路径、变量名和 JSON 转义外，不得自行改变本文确定的 Prompt、Schema、模型、参数、数据边界或 Gate 规则。
