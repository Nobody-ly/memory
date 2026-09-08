# REALTALK Ours V14：Ca 行为证据与 Turn Bundle 渐进实验锚点

## 1. 目标

V14 从冻结的 V9 完整结果出发，只重新生成 Decision 和 Response Actor。

- 数据：REALTALK Table 8 的逐人物 Ca/Cb 映射。
- 范围：Ca、Cb 均为按时间排序的前三个连续 Session。
- 单位：同一人物连续发送的原始气泡合并为一个目标 turn，气泡正文以换行无损保留。
- 总量：10 位人物、519 个目标 turn。
- 历史：每个目标点只读取它之前的全部真实 Cb 历史，不裁剪、不摘要，不回灌模型输出。
- 模型：`deepseek-v4-flash`，Decision 和 Actor 均关闭 thinking。
- 评价：本地五项指标和 REALTALK Appendix C 的 `gpt-4o-mini` 三项 Judge。

本版本不是重新优化冻结的 V9 测试答案。Ca 证据检索和渐进样本选择均不得使用 Cb
Ground Truth、未来消息、V9 错误标签或 Judge 标签。

## 2. 冻结的 V9 来源

- 实现提交：`5927bbff03fda74eebaeb99e0c57203a644cfd74`
- 生成目录：`realtalk-ours-v9-full519-evidencefix-flash-5927bbf`
- 预测记录：519/519
- 预测 SHA256：`ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81`
- Judge 单元：3114/3114，零 unresolved

V14 直接复用 V9 的 10 份 Self Domain 和每个目标点当时可见的 User Domain。V9
目录保持只读，V14 使用新协议、新分支和新输出目录。

## 3. V9 已确认的问题

### 3.1 真实 turn 被压成单动作

- Ground Truth 含多气泡：289/519（55.7%）。
- V9 生成含多气泡：1/519（0.2%）。
- V9 Actor 强制“一个互斥主动作 + 最多一个回问”，与数据中的自然 turn 结构不符。

### 3.2 自适应权衡塌缩

- `self-led`：518/519。
- `balanced`：1/519。
- 平均 lambda 约 0.035。

V14 将 lambda 重新定义为“当前伙伴情境对本轮计划的影响程度”，由同一次 Decision 调用
生成并通过整个 message plan 起效。它仍是审计轨迹，不参与确定性公式拼接。

### 3.3 三个弱项不是单一频率问题

- Reflectiveness：假阳性 94、假阴性 61，问题是出现时机，而非只需整体增减。
- Grounding：预测正例率与参考接近，但假阳性 109、假阴性 96，同样是时机错配。
- Intimacy：总体既有过高也有过低，不能用统一“变暖/变冷”规则修复。

因此 V14 不把评价指标写入 Prompt，也不进行全局频率硬调节，而是让 Decision 根据当前历史
和相似 Ca 行为证据选择反思、问题、亲密语气与多动作组合。

## 4. Ca 行为证据库

每位人物只用其 Table 8 指定 Ca 的前三个 Session 建立行为证据库。每条证据包含：

- 当前人物过去的完整目标 turn；
- 同一 Session 中紧邻它之前的伙伴 turn；
- 原始 turn/session ID；
- 连续气泡数量和字符数；
- 是否含问题、第一人称和反思标记；
- 确定性互动触发类型。

检索使用互动触发一致性、问句结构、第一人称结构和简单词汇 Jaccard，相同分数按
`example_id` 排序。每个 Cb 目标点最多取 3 条。检索不增加 LLM 调用。

Ca 例子的原文只用于学习人物的互动形状和表达方式，不能把例子中的旧地点、活动、计划或
事件冒充为当前事实。

## 5. V14 Decision

输入：

1. 完整可见 Cb 历史；
2. 冻结的 V9 Self Domain；
3. 当前点冻结的五层 User Domain；
4. 最多 3 条 Ca 行为类比；
5. Ca 总体行为统计和当前点之前的 Cb 人物行为统计。

严格 Schema 输出：

- `situation`：当前话题、伙伴动作、当前交流义务、问题/情绪/支持信号；
- `relevant_user_domain`：最多 2 条现有画像事实；
- `alignment`：动态 orientation、`lambda_trace` 和依据；
- `message_plan`：一个主动作、最多两个兼容辅助动作、1–6 个气泡、问题计划、反思深度、
  关系语气、长度档位、内容方向和语气。

直接问题或支持请求必须真实影响计划，不能再次自动落入接近零的 self-led lambda。除此之外
不强制问题、反思、确认、自我披露或暖化表达。

## 6. V14 Actor

Actor 输入完整可见历史、行为化 Self Domain、稳定身份背景、最多 2 条相关伙伴事实、3 条
Ca 类比、当前 Situation 和唯一 Turn Bundle 计划。

Actor 只输出目标人物消息。计划为多气泡时，用换行分隔，不编号、不加人物名。Actor 不读取：

- 完整 User Domain；
- lambda 数值和理由；
- Ground Truth；
- 评价指标或 Judge 标签。

不启用 Verification、候选搜索、语义重写或生成结果回灌。

## 7. 渐进样本与停止规则

样本清单只依据 speaker、session 和位置构造，固定且嵌套：

- 6：覆盖 6 位人物和全部 3 个 Session，只检查链路、lambda 和结构。
- 18：覆盖全部人物和 Session，观察是否出现系统性错误。
- 30：每个 `speaker × session` 恰好 1 条。
- 60：每个单元 2 条。
- 120：每个单元 4 条。
- 519：完整协议。

每级只在前一级通过后运行。同一 Prompt 版本扩容时复用已经完成的 ID，不重复生成；Prompt
发生修改必须升级协议并从 6 条重新开始。

通过标准：

- 所有记录成功，零 unresolved、零 JSON/身份泄漏；
- 6 条时确认 lambda 不塌缩、Decision 与 Actor 基本一致、多气泡能力真实生效；
- 18/30 条重点看配对 V9 的 Reflectiveness、Grounding、Intimacy AD 和人工样本；
- 30 条之后才允许冻结并进入 60；
- 任何阶段出现明显系统性退化即停止，不用更大样本掩盖问题。

小样本只与完全相同 ID 的 V9 做配对诊断，不能直接宣称超过论文总体结果。论文 Table 2 只在
完整 519 条完成后作为正式横向参照。

## 8. 审计产物

每次运行保存：

- V9 来源路径与 SHA256；
- 数据文件和 Table 8 manifest；
- 不可变 gate ID 清单与 SHA256；
- Ca 行为库、汇总统计和逐点检索结果；
- Decision、lambda、Turn Bundle 计划和 Actor 结构符合度；
- Prompt/Schema/代码哈希；
- 原始响应、重试、token、checkpoint 和 unresolved；
- V9/V14 同 ID 的本地指标、Appendix C Judge 和人工抽检报告。

配对 Judge 复用冻结 V9 checkpoint 中同一 `result_id` 的 Ground Truth 三项标签，仅对 V14
候选调用 Judge。复用不改变 Appendix C Prompt、模型或解析规则，可避免重复判断参考答案带来
随机波动，并将每条新增 Judge 调用从 6 次降为 3 次。

## 9. 明确不改内容

- 不改 Table 8 Ca/Cb 映射和前三个连续 Session。
- 不改 519 条目标集合和真实滚动历史。
- 不重新生成 Self Domain、User Domain 或其 Session 更新节奏。
- 不筛除困难、事实新颖或无法预测的目标。
- 不把八项指标写进生成 Prompt。
- 不重新运行 V9，不覆盖 V9–V13 产物。

## 10. Gate 6 V14 初版审计与 V14.1 修正

V14 初版 Gate 6 完成 6/6、零 unresolved：lambda 为 0.3–0.6，候选多气泡率由 V9
近零提升到 66.7%。但该版本未进入评价和扩容，原因是：

- 1 条 Decision 的 `question_plan=none`，自由文本 `content_direction` 却要求结尾提问；
- 1 条 Actor 因此生成未授权问题；
- 原始 Ca 类比文本直接进入 Actor，存在把旧事实或伙伴事实写成当前自身事实的风险；
- 两条短计划被 Actor 扩展成明显偏长的说明。

V14.1 采用通用协议修正，不根据 Ground Truth 改答案：

- 对 Decision 的问题计划与自由文本方向增加确定性一致性校验；
- Actor 只看到 Ca 类比的气泡数、长度、问句和反思结构，不再看到 Ca 原文；
- Actor 回到 V9 的行为化 Self Domain 视图，稳定身份仍由 Decision 读取；
- 明确保持伙伴事实所有权，禁止把伙伴的工作、活动、感受或计划改写成自身事实；
- 要求精确实现计划气泡数，短计划不因多气泡而扩写成说明文。

Prompt 已变更，因此协议升级为 `realtalk_task1_ours_v14_1_ca_behavior_turn_bundle`，使用
新输出目录并从 Gate 6 重新开始。

V14.1 实际完成 5/6，成功样本中气泡数与问题权限均为 5/5 一致，lambda 覆盖三种方向。
第 6 条在三次 Decision 中均出现“结构化 question_plan 为 none、自由文本方向仍提到问句”的
冲突，被过严的 normalizer 判为 unresolved。V14.2 不替模型改策略，也不重写 Actor：

- 结构化 `question_plan` 成为 Actor 的唯一权威；
- 自由文本冲突记录为 `decision_plan_audit` 警告，不再阻断样本；
- Actor 仍必须按 question_plan 执行，最终是否越权由 `actor_structure_audit` 检查。

因此 V14.2 使用新协议和新目录，再次从 Gate 6 开始。

V14.2 完成 6/6、零 unresolved，lambda 在三种方向各 2 条，但配对 Gate 6 Judge 显示：

| 指标 | V9 同 ID | V14.2 | 判断 |
|---|---:|---:|---|
| Reflectiveness | 0.667 | 0.167 | 明显下降 |
| Grounding | 0.833 | 0.500 | 明显下降 |
| Empathy AD | 1.167 | 1.500 | 下降 |

逐条审计发现 6/6 Decision 都加入了辅助动作。问题不是“多气泡”，而是把多气泡错误解释为必须
包含多个社交动作，导致普通回复出现额外确认、温暖评价、自我解释和问题。V14.3 因此：

- 保留多气泡，但明确多个气泡可以共同实现一个主动作；普通 turn 默认不加辅助动作；
- `reflection_depth=none` 时直接陈述，不使用 `I think/I feel/I guess`、理由或自我分析；
- `surface` 只允许一个简单立场或感受，`brief` 才允许一句反思解释；
- question_plan 允许时只问一个信息问题；`you know?` 等口语 tag 不算信息问题；
- 不为普通回复自动添加赞美、确认、关心或暖化措辞。

V14.3 再次使用新协议和新目录，从 Gate 6 开始。

V14.3 在 Gate 6 出现 5/6：模型减少辅助动作后，`question_plan=reciprocal` 不再重复写入
`supporting_moves`，但旧 Schema 强制两处同时存在，导致 Kevin 连续三次校验失败。其余 5 条中
仍有自由 `content_direction` 与结构化 question_plan 冲突，并使 Actor 产生额外问题。

V14.4 统一控制源：

- `question_plan` 是唯一问句控制字段；辅助动作枚举删除 follow-up/reciprocal-question；
- Actor View 不披露自由 `content_direction`，由完整可见历史和 Situation 决定具体内容；
- Decision 原始 content_direction 继续保存供审计，但不能覆盖结构化问题、反思和动作权限。

Schema 已变更，V14.4 使用新协议和输出目录，再从 Gate 6 开始。

V14.4 完成 5/6；成功样本的问题权限 5/5、气泡数 4/5 一致。Nicolas 的 Decision 三次选择
“acknowledge 主动作 + follow-up 问句”，旧 normalizer 却错误要求 follow-up 问句必须同时是
主动作，导致 unresolved。Turn Bundle 本来就允许“先回应、再追问”，因此 V14.5 删除该反向
绑定，只保留“主动作若为 follow-up，question_plan 必须实际允许问句”的单向约束。Schema
变更后使用新协议和目录，再从 Gate 6 复测。

V14.5 完成 6/6、零 unresolved、气泡结构 6/6，但配对指标仍低于同 ID 的 V9：

| 指标 | V9 同 ID | V14.5 |
|---|---:|---:|
| Reflectiveness | 0.667 | 0.333 |
| Grounding | 0.833 | 0.667 |
| Empathy AD | 1.167 | 2.167 |
| ROUGE | 0.175 | 0.111 |
| BERTScore | 0.859 | 0.844 |

这说明主要退化来自重新决策，而非格式。V14.6 改为 V9-anchored：

- 冻结的 V9 Situation/Alignment/Next Action 作为保守行为先验；
- 只有当前真实历史与相近 Ca 行为证据共同提供强证据时才改变主动作；
- 新模块主要负责 turn 气泡结构和有证据的组合动作，不把单动作自动升级成更温暖、更完整的回复；
- V9 generated message 永不进入 V14 Prompt，避免成为文本重写或自蒸馏。

Prompt 变更后，V14.6 使用新协议和新目录，从 Gate 6 开始。

V14.6 把 V9 Alignment 也放入保守先验，结果 2 条出现“partner-adaptive orientation + 低于
0.55 的旧 lambda”并耗尽重试；其余 4 条全部 self-led，lambda 均值 0.025。V14.7 因此只
锚定 V9 Situation 与 Next Action，不向新 Decision 披露 V9 orientation/lambda。这样保留 V9
较稳的行为选择，同时让新 lambda 根据当前历史、Self/User Domain 与 Ca 行为证据独立计算。
Prompt 变更后使用新协议和目录，从 Gate 6 开始。

V14.7 在扩至 Gate 18 时完成 17/18；缺失样本连续三次选择 self-led/低 lambda 回应直接问题，
被旧 validator 强制拒绝。该规则把“理解并回答伙伴”错误等同于必须偏离人物先验，不符合
Persona Simulation。V14.8 将 orientation/lambda 的经验区间与 partner-trigger 低 lambda 改为
`alignment_audit` 非阻断警告：模型的权衡仍完整记录，是否有效以 message plan 和输出为准，
不再用人为阈值改写人物行为。实现语义变更后使用新协议和目录，从 Gate 6 重新验证。

V14.8 完成 Gate 6 的 6/6、零 unresolved，问题权限 6/6 一致，但计划气泡数只有 2/6 被
Actor 执行。审计进一步发现 Actor View 未收到任何内容方向：Decision 虽然产生
`content_direction`，该字段却在投影时被删去，Actor 只能根据历史重新猜一次内容，造成
内容漂移和过度展开。V14.9 因此：

- 向 Actor 披露冻结 V9 `next_action.content_direction`，仅作为“谈什么”的内容锚点；
- 新 Decision 继续负责“怎么表达”的气泡、关系语气、反思、问题权限与动态 lambda；
- 辅助动作最多一个，避免把普通 turn 扩展成全面、温暖的助手式回答；
- 对可在句界切分的单行输出仅替换句间空白为换行，不增加、删除或重写词语；不能无损切分时
  保留原文并记录审计，不强行破坏句子。

V14.9 使用新协议和新目录，从 Gate 6 重新验证；只有结构与配对评价均无明显退化才扩至 18 条。
