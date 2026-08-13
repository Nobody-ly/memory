# REALTALK Ours Agentic V2 实验锚点

## 1. 固定目标

本实验仅在 REALTALK Persona Simulation 协议下运行 Ours，不训练、不微调、不运行论文基线。数据固定采用论文 Table 8 的逐人 Ca/Cb 分配，Ca 与 Cb 均使用按时间排序的前三个连续 Session；相邻同说话者气泡以换行无损合并，完整规模为 10 位目标人物、519 条目标消息。

协议名固定为 `realtalk_task1_ours_agentic_v2_ntdecision`，所有阶段使用 `qwen3-max-2026-01-23`。Self Domain、User Domain、Decision、Generation 均关闭 thinking。Decision 仍使用严格 Schema、动态 lambda 审计和唯一动作约束；关闭 thinking 用于避免供应商长时间阻塞，并检验过度推理是否造成不必要的伙伴适应。

## 2. 历史与因果边界

- Ca 前三个 Session 的完整双方原文用于生成每位目标人物的一份 Self Domain。
- Cb 前三个 Session 连续滚动；Session 2 继承 Session 1，Session 3 继承前两个 Session。
- 每个目标点看到该点之前的全部真实历史，不能看到当前 Ground Truth。
- 后续点使用真实消息推进历史，不回灌模型预测。
- 不压缩、不摘要、不检索替代、不裁剪历史。
- 相邻气泡合并保留每个气泡的完整文本与原始消息 ID。

## 3. 四阶段实现

### Self Domain Compiler

每人从完整 Ca3 生成一次并在 Cb 测试中固定。结构包括：`identity_context`、`communication_signature`、`interaction_policy_prior`、`affective_social_signature`、`boundaries_and_uncertainty`。消息数、字符均值/中位数、问句率、第一人称率和合并气泡规模由程序确定性计算并校验。

### User Domain Updater

保留 `Core / Regulation / Cognition / Identity / Behavior` 五层，每条事实包含 `value / confidence / evidence_ids`。Session 1 开始为空，只在 Session 1 完成后更新一次供 Session 2 使用，在 Session 2 完成后更新一次供 Session 3 使用。完整已完成 Session 进入更新 Prompt；目标人物消息只提供语境，证据 ID 必须属于伙伴。

### Private Decision Agent

每条样本输入完整真实历史、完整 Self Domain 和当前 User Domain，输出：

- 当前 `situation`；
- 最多两条逐字匹配的 `relevant_user_domain`；
- `self-led / balanced / partner-adaptive` 与审计用 `lambda_trace`；
- 唯一 `next_action`。

`lambda_trace` 是同次决策的可审计轨迹，不进行确定性公式混合。Self Domain 是默认身份，伙伴画像仅在当前相关时激活。Self Domain 主要决定声音、主动性、互动方式和消息规模，不是当前事实来源或需要逐项展示的内容清单；不得把 Ca 的地点、天气、职业、兴趣、惯例或旧事件复演为当前场景，也不得把多个画像兴趣打包进一条回复。Cb 历史未建立当前事实时，仅允许低具体度的自然自我表达；空历史选择符合人物风格的简单开场。`typical` 消息规模以确定性字符中位数为锚。

Decision 必须选择一个而非 `mixed` 的 Primary Move。普通的相关回应、回答问题或理解上下文不自动提高 λ；日常事实和闲聊默认由 Self Domain 主导，仅在明确的情绪、关系或实际需要出现且目标人物会适应时进入 balanced/partner-adaptive。`question_mode` 仅允许明确的 `none` 或 `follow-up`，追问按 Self Domain 的观测问句率和问句习惯校准；同时使用观测第一人称率和主动性恢复自我披露、延续及转移话题的真实节奏，不能形成每条消息都确认对方并追问的固定模式。

对于高问句倾向人物，`answer` 后允许一个同槽位的短对称回问（如双方的近况、计划或偏好），记录为唯一主动作后的 `reciprocal-question`，不视为第二策略。Actor 使用 Ca 的字符均值与中位数形成 short/typical/extended 的具体字符引导，避免把一个动作扩写成额外解释；该引导不是固定句数或硬截断。

### Response Actor

Actor 只接收完整真实历史、完整 Self Domain、`situation` 和唯一 `next_action`，输出目标人物自然下一条消息。它不接收完整 User Domain、证据、λ 数值或依据、评价指标、Future State。

## 4. 禁用项

固定禁用 Omega、Future User State、历史压缩、历史裁剪、Verification、重写调用和多候选 Behavior Policy。最终回复不设置句数限制，不强制共情、建议、追问或个性化。

## 5. Schema、恢复与审计

三个结构化阶段使用严格 JSON Schema，每次最多三次逻辑尝试。百炼 Qwen thinking 不支持强制工具调用，且实测 JSON mode 会阻塞，因此 Decision 将完整 Schema 写入 Prompt，并以同一 Schema 在本地严格校验；其他非 thinking 阶段继续使用端点原生强制 Schema。格式修复只修正 JSON、字段和类型。所有操作使用幂等 key 和检查点；最终完成要求 0 unresolved。原始 content 与 reasoning 分字段保存，报告只披露 thinking 模式、token 与哈希。API 密钥只从环境变量读取。

## 6. 执行门槛

先运行 Akib 前 21 条：要求 21/21、零 unresolved、零结构泄漏。随后计算五项本地指标和三项 `gpt-4o-mini` 指标。进入 519 条完整运行的门槛为：Reflectiveness ≥ 0.60、Grounding ≥ 0.55 且不接近全正例、Empathy AD ≤ 1.80，并且人工抽检无系统性客服或治疗式回复。

八项指标仅用于评价，不进入任何生成 Prompt。未达标时只调整 Self/Decision Prompt，不更换数据或筛选样本。
