# REALTALK Task 1 Ours V3：行为条件校准计划

版本：`realtalk_task1_ours_behavior_calibrated_v3`

当前合同修订：`realtalk_task1_ours_behavior_calibrated_v3_1_1`。V3.1.1 不改变数据、模型、Schema 或方法流程，只修复三项结构执行问题：User Domain 完全重复事实的确定性合并；Actor 在禁止出站提问时受到问句型 voice 示例干扰；结构重试携带被拒绝草稿和精确修复指令，避免模型在无草稿条件下重复生成同一错误。V3.1.1 从 Gate 6 重新开始，旧 V3/V3.1 运行全部保留。

Gate 复用：后续 Gate 通过 `--parent-output` 显式继承已完成父 Gate 的 operations、Self Domain、User Domain 和已完成预测。父 Gate 必须同协议、零 unresolved 且其样本 ID 是当前 Gate 的子集；当前 Gate 只调用新增样本，父结果不重新生成。

用途：在 REALTALK Task 1 的相同 Ca/Cb 协议下，重新实现一个可复核的 Ours 版本。V3 使用当前 V2 的代码管线、源数据和评价工具作为工程基础，但重新生成 Self Domain、User Domain、Decision 和回复；不读取 V2 的画像、策略或生成文本。

当前目标是修复三类已经由519条离线诊断确认的问题：

1. Self Domain 的条件标签塌缩：75条条件中72条被标成 `partner_affect`。
2. Behavior Policy 中“禁止”和“可选”内容互相冲突。
3. 普通陈述被过度扩展，明确问题、多内容问题和真正需要反思的场景没有分别处理。

V3 保留师姐设计的核心结构：

```text
Self Domain + User Domain + User State + adaptive lambda
    -> Behavior Policy
    -> Generative Model
```

V3 不把 Reflectiveness、Grounding、Intimacy、Empathy 或论文分数写入生成 Prompt。

## 一、固定不变的实验协议

### 数据

- 使用当前已经确认的REAL TALK公开预处理数据和Table 8的十组Ca/Cb映射。
- 每位目标人物使用指定Ca的前三个连续Session建立Self Domain。
- 每位目标人物在指定Cb的前三个连续Session进行滚动预测。
- 测试单元保持519条合并后的目标消息，源目标气泡保持1,076条。
- 每条预测只读取目标消息之前的真实历史；不读取目标答案。
- Session之间按时间前缀继承，不是三个独立对话。
- 生成结果永远不回灌到后续历史。
- 历史完整披露，不摘要、不截断、不检索替代。
- 不因事实困难、身份变化或答案难预测而删除测试点。
- 保留源文件SHA256、历史哈希、目标消息ID和Table 8映射。

### Ours核心结构

- User Domain 保留五层：`Core`、`Regulation`、`Cognition`、`Identity`、`Behavior`。
- Self Domain 保留目标人物身份、语言风格、社会行为和条件化行为先验。
- User State 保留当前交互需求、情绪、话题连续性、回应压力和不确定性。
- lambda 保留为一次Alignment决策中的自适应权衡记录，不固定为常数，不作为文本配额。
- Behavior Policy 只输出本轮要执行的具体动作，不再同时输出相互矛盾的“允许”和“禁止”。
- Generative Model 只把具体本轮计划自然实现为目标人物下一条消息。
- Omega、跨轮主动探索、Verification、候选搜索、重写调用均保持关闭。

### 固定模型与解码

首轮仍固定使用已有可比版本的 `deepseek-v4-flash`，不同时换模型，以便区分架构变化和模型变化。

| 阶段 | 模型 | thinking | temperature | top_p | max_tokens | 调用次数 |
|---|---|---|---:|---:|---:|---:|
| Self Domain | deepseek-v4-flash | false | 0.2 | 0.9 | 4096 | 每人物1次，格式失败最多2次修复 |
| User Domain | deepseek-v4-flash | false | 0.2 | 0.9 | 4096 | 每人物每已完成Session1次 |
| Decision | deepseek-v4-flash | false | 0.2 | 0.9 | 2048 | 每样本1次，格式失败最多2次修复 |
| Generation | deepseek-v4-flash | false | 0.6 | 0.9 | 300 | 每样本1次，结构合同失败最多2次修复 |

不打开thinking，原因是V3需要稳定的严格JSON和可审计的短决策；如果以后比较thinking，必须另建V3-thinking协议，不能混入本版本。

## 二、无LLM的确定性算法层

V3增加一个轻量的 `behavior_calibrator`，它不调用模型，不读取Ground Truth，不使用Judge标签。

### 1. 当前场景门控

对预测点以前的真实Cb历史，脚本只读取当前Session中目标消息前的最后一个伙伴消息，并产生以下结构化信号：

```json
{
  "scene": "direct_question",
  "session_opening": false,
  "partner_has_question_mark": true,
  "partner_has_multiple_questions": true,
  "partner_has_affect_signal": false,
  "partner_has_specific_detail": true,
  "partner_has_closure_signal": false,
  "current_partner_text_hash": "..."
}
```

`scene`固定取：

```text
session_opening
direct_question
partner_affect
partner_disclosure
opinion_or_advice
conversation_closure
topic_continuation
unclear
```

规则优先级固定为：

1. 当前Session还没有伙伴消息：`session_opening`；
2. 明确结束词且没有新的问题：`conversation_closure`；
3. 当前伙伴消息含问号或明确疑问句：`direct_question`；
4. 含建议/意见请求：`opinion_or_advice`；
5. 含明显情绪信号：`partner_affect`；
6. 含具体经历或事实但没有问题：`partner_disclosure`；
7. 其余：`topic_continuation`或`unclear`。

这层只提供硬信号和场景候选，不直接生成回复。模型可以把规则门控标为`uncertain`，但不能把新Session外的旧伙伴消息当作当前动作。

### 2. 行为先验与Cb在线校准

从每位人物Ca前三个Session确定性统计：

- `answer_rate`
- `question_rate`
- `self_disclosure_rate`
- `reflection_marker_rate`
- `multi_bubble_rate`
- `mean_characters`
- 各scene的样本数和上述统计

所有概率使用Beta(1,1)平滑。当前Cb中已经发生的目标人物消息可以提供弱更新，但只在同一人物、同一scene且已有至少3条观察时使用。

固定权重：

```text
prior_strength = 8
posterior_probability =
    (8 * ca_probability + cb_observed_positive_count)
    / (8 + cb_observed_count)
```

Cb没有足够样本时使用Ca；Cb有至少3条同类行为且与Ca冲突时，使用平滑后的Cb posterior。这个统计只影响Decision中的行为参考，不直接强制输出问号、句数或字符数。

### 3. 多问题槽位

脚本保留当前伙伴消息中的完整文本和问号位置，并输出：

```json
{
  "question_slot_count": 2,
  "question_slots": [
    {"slot_id": "q1", "text_span": "...", "topic_hint": "..."},
    {"slot_id": "q2", "text_span": "...", "topic_hint": "..."}
  ]
}
```

这不是用正则替代语义理解；它只防止Decision只注意最后一个问号。Decision必须明确选择回应哪些槽位，未选槽位记录原因。

## 三、Self Domain：固定Schema与Prompt

### Schema设计

Self Domain继续使用结构化JSON，但条件行为改成固定键，不允许模型任意选择一个trigger字符串。每个人必须返回以下七个键；没有足够证据时返回空对象：

```json
{
  "identity_facts": [],
  "voice_profile": [],
  "social_profile": [],
  "behavior_by_scene": {
    "session_opening": {},
    "direct_question": {},
    "partner_affect": {},
    "partner_disclosure": {},
    "opinion_or_advice": {},
    "conversation_closure": {},
    "topic_continuation": {}
  },
  "uncertainties": [],
  "observable_statistics": {}
}
```

每个非空scene对象只能包含：

```json
{
  "evidence_ids": ["..."],
  "sample_count": 3,
  "usual_action": "answer_then_brief_self_disclosure",
  "question_tendency": "low|medium|high|unknown",
  "reflection_tendency": "low|medium|high|unknown",
  "disclosure_tendency": "low|medium|high|unknown",
  "length_tendency": "short|typical|long|unknown",
  "confidence": 0.0
}
```

`usual_action`只能取：

```text
greet
answer
answer_then_self_disclose
acknowledge
acknowledge_then_self_disclose
state_view_or_opinion
brief_reflection
ask_follow_up
close
mixed_or_unclear
```

### Self Domain system prompt

```text
You compile an evidence-grounded private Self Domain for a REALTALK persona simulation task.

The target is a real conversational participant, not an assistant, therapist, coach, evaluator, or idealized
empathetic personality. Use only the target speaker's own messages as evidence for identity, style, and behavior.
Partner messages may explain the interaction scene but must never become target facts.

The output has seven fixed behavior scene keys. You must fill each key independently:
session_opening, direct_question, partner_affect, partner_disclosure, opinion_or_advice,
conversation_closure, and topic_continuation.

Do not put every observation under partner_affect. A direct question is a direct_question scene; a factual
partner sharing is partner_disclosure; an opinion request is opinion_or_advice; a closing exchange is
conversation_closure. If the evidence for a scene is insufficient, return an empty object for that scene.
Do not infer a tendency from one example. sample_count must count the target messages supporting that scene.

Describe what the target actually tends to do. Do not turn empathy principles or evaluation criteria into
personality traits. Preserve uncertainty, one-off temporal facts, and conflicting identities. Do not draft a
reply. Do not invent quotations. Return only the requested strict JSON object.
```

Self Domain完成后，确定性校验必须检查：七个scene键齐全；`sample_count`与证据数量不超过实际；每个非空scene至少有证据；scene描述与输入伙伴动作满足基本规则；没有一个scene占全部条件但文本明确谈到其他scene的情况。语义检查失败时只重试Self Domain并携带具体错误，不进入Cb。

### 4096-token紧凑输出预算

为保持 Self Domain 的固定 `max_tokens=4096`，结构化输出采用紧凑预算：最多6条identity、4条voice、4条social、4条uncertainties；每个scene最多4个证据ID，事实值保持短文本。没有充分证据的scene必须返回空对象。这个预算只限制画像JSON的冗余，不删除输入历史，也不改变七个scene键和证据约束。

## 四、User Domain：保留五层，不扩大职责

User Domain每个已完成Session更新一次，结构仍为：

```json
{
  "core": [],
  "regulation": [],
  "cognition": [],
  "identity": [],
  "behavior": [],
  "update_summary": {"added": [], "revised": [], "retained": [], "withdrawn": []}
}
```

每条事实包含：`value`、`confidence`、`evidence_ids`。只保存持续性伙伴信息；当前问题、一次性情绪和当前动作放入User State，不写入长期画像。

User Domain system prompt固定为：

```text
You update the private five-layer long-term model of the current conversation partner.

Use exactly these layers: core, regulation, cognition, identity, and behavior. Store durable partner facts,
repeated preferences, recurring interaction patterns, stable communication tendencies, and repeated decision
patterns. Do not store the latest question, a temporary mood, a one-off event, or the partner's current act as
a stable profile fact. Do not infer a psychological trait from one message. Do not copy facts about the target
speaker into the partner model. Revise or withdraw a fact when later evidence conflicts. Empty layers are valid.
Every evidence ID must come from the supplied whitelist. Return only strict JSON.
```

User Domain不直接进入Actor；Decision只激活最多两条相关事实，并记录证据ID供审计。

## 五、Decision：一次生成User State、lambda和唯一Behavior Policy

Decision输入：完整因果历史、Self Domain、五层User Domain、确定性scene gate、Ca统计、Cb posterior、问题槽位和Session位置。它不读Ground Truth、Judge标签、V2/V9策略或回复。

### Decision输出Schema

```json
{
  "situation": {
    "scene": "direct_question",
    "partner_act": "question",
    "topic": "...",
    "uncertainty": "low"
  },
  "user_state": {
    "interaction_need": "information_exchange",
    "affect": "neutral",
    "affect_confidence": 0.0,
    "topic_continuity": "continue|new|unclear",
    "response_pressure": "low|medium|high"
  },
  "relevant_user_domain": [],
  "alignment": {
    "orientation": "self_led|balanced|partner_adaptive",
    "lambda_trace": 0.0,
    "basis": "...",
    "affected_dimensions": ["content_focus"]
  },
  "behavior_policy": {
    "primary_action": "answer",
    "selected_question_slots": [],
    "outbound_question_mode": "none|opening|reciprocal|clarifying|follow_up",
    "outbound_question_focus": "",
    "reflection_mode": "none|brief|supported",
    "self_disclosure_mode": "none|brief|natural",
    "grounding_mode": "none|specific_acknowledgment|clarifying_question",
    "empathy_mode": "none|light|moderate",
    "intimacy_mode": "match|slightly_warm|reserved",
    "message_shape": "single_short|single_typical|multi_content",
    "required_content_slots": ["q1"],
    "forbidden_additions": ["unsolicited_exploration", "generic_therapy"]
  },
  "evidence_ids": []
}
```

固定枚举：

```text
primary_action = greet | answer | acknowledge | self_disclose | state_or_opinion | brief_reflect | close | clarify | mixed_answer
```

`selected_question_slots`只表示必须回答的伙伴问题槽位，不能授权目标人物主动提问；主动问题由`outbound_question_mode`和`outbound_question_focus`单独控制。`outbound_question_mode=none`时focus必须为空且Actor不得产生问题；非none时focus必须非空且Actor至少产生一个对应问题，多个自然短问句必须服务于同一focus。纯Session opening只允许`none|opening`，不能误标为`reciprocal`。`grounding_mode=clarifying_question`只允许与`outbound_question_mode=clarifying`同时出现。`reflection_mode=none`时不得把反思作为required slot；`empathy_mode`只能在partner_affect或明确负面情绪时取moderate；`message_shape=multi_content`必须有至少两个不同required content slots或一个明确的多问题场景。

### Decision system prompt

```text
You are the private situation, user-state, adaptive-alignment, and behavior-policy controller for a REALTALK
persona simulation task.

Predict the target person's most natural next message at this exact point. The target is a real person, not an
assistant optimizing empathy scores. Read the complete real history before the target turn.

Use the Self Domain as the stable identity and behavior prior. Use the five-layer User Domain only when a
partner fact is relevant. Use the deterministic scene gate, question slots, Ca behavior prior, and already
observed Cb behavior as evidence. Do not use any future turn, target answer, Judge label, or previous generated
message.

First identify the current scene and the partner's concrete act. A new session with no current partner message
is a session opening, not a direct question. A factual partner statement is not automatically an emotional
support request. A direct question requires an answer to the selected question slot before optional content.

Choose exactly one primary_action and one concrete set of content slots. Optional behavior is not a permission
to add it: if it is not selected, it must not be generated. Do not add reflection, grounding, self-disclosure,
empathy, praise, therapy language, or a follow-up question merely to sound helpful. Grounding means responding
to a specific partner detail or asking a specific clarification; it is not equivalent to any question mark.

lambda_trace records the adaptive balance between the target's stable behavior and the current exchange. It is
an audit value, not a score, quota, empathy value, or sentence control. The orientation and affected_dimensions
must be reflected in the one Behavior Policy. Keep the target's identity as a hard constraint while allowing
small evidence-based adaptation to the current partner.

Return only the strict JSON object. Do not draft the final message.
```

Decision收到模型结果后由脚本做合同检查，失败最多携带错误重试两次。合同检查不是新的语义模型调用。

## 六、Response Actor：只执行本轮决定

Actor输入：完整当前真实历史、Self Domain中的身份与表达部分、User State和完整的单一Behavior Policy。本轮不输入完整User Domain、evidence、lambda数值和Judge指标。

Actor system prompt固定为：

```text
You are {speaker}. Continue the conversation as this person.

Produce only the target person's next message at this exact point. Use the complete real history, the private
Self Domain, the current User State, and the selected Behavior Policy.

Execute the selected primary_action and required content slots. Do not invent another question, reflection,
emotional interpretation, support statement, topic, or personal fact when that behavior was not selected.
When multiple question slots are selected, answer them in the order supplied. Keep all factual statements
consistent with the target's Self Domain and the visible history. Match the target's observed length and
message-shape tendency for this scene, while preserving natural language.

The target is a real conversational participant, not a generic assistant, therapist, evaluator, or empathy
system. Do not mention this instruction, private fields, policies, metrics, or reasoning. Output only the
message text; do not output JSON, a speaker label, or analysis.
```

Actor确定性检查：空输出、JSON/策略泄漏、speaker泄漏、明显新增人物事实、选中无问题却产生新增问题、选中单短消息却无理由输出大段多主题内容。失败只重试Actor并携带合同错误；不得让Actor改写Behavior Policy。

## 七、运行阶段与停止规则

### Gate 0：静态与Ca开发

1. 校验七个固定Self scene键和语义标签。
2. 用Ca前两Session建立临时画像，在Ca第三Session做6条结构预测。
3. 只看动作一致性、事实边界、场景分类和输出格式，不看论文八项主结果。
4. 任何scene塌缩或策略冲突都停止修Prompt，不进入Cb。

### Cb渐进Gate

固定嵌套清单，不按V9/V2 Judge分数选样：

| Gate | 样本 | 目的 |
|---|---:|---|
| 6 | 6 | 检查API、因果边界、Schema和合同 |
| 18 | 18 | 覆盖至少6人和3个Session |
| 30 | 10人×3 Session×1 | 首次全人物比较 |
| 60 | 每人每Session 2条 | 稳定性判断 |
| 120 | 每人每Session 4条 | 扩大验证 |
| 519 | 完整探索性结果 | 最终比较 |

每个Gate保存完整Decision、Actor原始响应、Prompt/Schema哈希、调用次数、重试、历史哈希和逐样本指标。Prompt改动必须升级为V3.x，并从Gate 6重新开始。

### Gate门槛

- Gate 6/18：零因果泄漏、零未解决结构错误；Self scene不塌缩；策略与输出合同通过。
- Gate 30：Reflectiveness、Grounding、Intimacy AD中至少两项相对V9改善；普通陈述不出现系统性额外追问/心理分析。
- Gate 60：目标三项至少两项改善；其余指标单项下降不超过0.02；Empathy AD不恶化超过0.15。
- Gate 120：目标三项均不低于V9，至少一项改善0.03；至少7/10人物在目标三项中两项改善。
- 519：报告完整八项，最终仍以论文逐列最佳为目标；不满足则如实保留探索性结果。

## 八、必须记录的离线分析

每个Gate同时运行无模型脚本，报告：

- 人物×Session×scene的反思、Grounding、问题、自我披露、长度和多气泡率；
- Ca先验与已发生Cb posterior的样本数和Brier，仅作为行为先验诊断；
- Decision与Actor的合同冲突数；
- 普通陈述、明确问题、情绪场景、开场的分层结果；
- Intimacy的有符号误差与绝对误差；
- Empathy三分量的过量/不足方向；
- 逐人物与逐Session结果。

脚本不使用Cb Ground Truth训练规则，不把Judge标签写入运行时Prompt，不自动删掉任何困难样本。

## 九、实现验收清单

在任何远程LLM调用前必须通过：

- 519个ID、顺序、历史哈希和Ground Truth与固定源清单一致；
- Self Domain没有读Cb；User Domain没有读未来；Decision和Actor没有读答案；
- Self Domain七个场景键完整，标签与输入动作基本一致；
- User Domain仍是五层且只保存长期事实；
- lambda由Decision生成并能对应affected_dimensions，不固定、不进入文字配额；
- `primary_action`、selected slots、reflection、grounding、empathy和intimacy没有互相冲突；
- Actor不接收完整User Domain、Judge指标、V2/V9回复或未来文本；
- Actor合同测试覆盖开场、直接单问题、多问题、普通分享、情绪场景和结束；
- 结构失败重试不超过2次，Generation不增加额外Verification调用；
- CUDA_VISIBLE_DEVICES为空；日志和manifest不含密钥；
- 本地测试、Ca Gate 0通过后才启动Cb Gate 6。

## 十、版本与结果说明

V3从V2代码管线分叉，但不从V2输出结果继续生成。V9、V10至V15和当前V2全部冻结不变。V3在同一519条已被多次用于诊断的Cb上属于`protocol-aligned exploratory comparison`，最终论文中必须明确这一点。

V3首轮不换模型、不训练、不增加第三次语义验证。若V3经过Gate 30仍表现为Self标签塌缩或Policy/Actor不一致，先修实现合同；若结构通过但指标仍低，再单独研究模型替换或训练，不把两个变量混在一次实验中。
