# REALTALK Ours V1.6 锚定方案

## 目标

V1.6 针对 V1.5.1 的 Gate 24 结果进行一次受控重构：保留当前情况识别、User Domain、动态 lambda 和完整真实历史；取消固定气泡数、动作序列和强制多动作，让最终回复恢复自然表达。

V1.5.1 的 24 条配对结果为：Grounding 上升，但 Reflectiveness、Intimacy AD、Empathy AD 和 ROUGE 下降。因此 V1.6 不继续强化 Message Bundle，而是把 Decision 改成软性回复指导。

## 固定实验条件

- 新协议：`realtalk_task1_ours_evidence_conditioned_v1_6`
- 模型：`deepseek-v4-flash`
- Decision：temperature 0.2，top_p 0.9，max_tokens 1536，thinking false
- Actor：temperature 0.6，top_p 0.9，max_tokens 1024，thinking false
- Self Domain、User Domain、Ca/Cb、前三个 Session、完整历史和 Ground Truth 保持现有协议
- 不训练、不微调、不使用候选搜索、Verification、重写或 Judge 指标引导

## Decision

Decision 输出 `situation`、`alignment`、`response_guidance` 和 `evidence_ids`。lambda 由同一次 Decision 调用生成，只作为可审计的软权衡记录；它通过 `response_guidance` 影响 Actor，不直接拼接文本，也不直接决定是否提问。

`response_guidance` 只规定当前必须处理的内容、可选内容、应避免的内容、问题权限、自我披露权限、反思权限和长度倾向。它不规定气泡数量或动作序列。

直接问题通常必须回答；只有当前信息缺口或人物习惯明确支持时才允许追问。Self Domain 是默认行为先验，User Domain 只有在当前话题直接相关时才被使用。

## Actor

Actor 输入完整真实历史、Self Domain 和 Decision 产生的软性 `response_guidance`。Actor 不看到 lambda 数值、lambda 理由、完整 User Domain、evidence_ids、Ground Truth 或评价指标。

Actor 可以自然地产生一个或多个气泡，不设置固定气泡数、字符数、问句数、自我披露数或反思数。只检查空输出、JSON/内部结构泄漏、Speaker 标签泄漏和明显 Ground Truth 泄漏。

## 渐进验证

先运行现有 V1.4/V1.5.1 同一批 24 条，使用同一 Judge 和共享 Ground Truth。V1.6 只有满足以下条件才进入 68 条：Reflectiveness 相对 V1.4 不低于 -0.03，Grounding 不低于 -0.03，Intimacy AD 恶化不超过 0.01，Empathy AD 恶化不超过 0.15，ROUGE 不低于 -0.02，BERTScore 不低于 -0.01；同时 24/24 成功且 unresolved 为 0。

24 条通过后再运行 68 条 Ca Dev；68 条通过后再考虑连续 Cb 困难窗口。任何一轮失败都保留产物并停止当前版本。

## 可追溯字段

Manifest 记录协议、模型、解码参数、Prompt/Schema 哈希、选中样本哈希、lambda 分布、生成问题率、历史完整性和错误数。所有 V1.4、V1.5.1 和 V1.6 产物使用独立目录。
