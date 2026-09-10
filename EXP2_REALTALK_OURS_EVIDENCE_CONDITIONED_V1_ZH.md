# REALTALK Ours Evidence-Conditioned V1

状态：实现锚点。协议：`realtalk_task1_ours_evidence_conditioned_v1`。

该实现依据 `D:/codex_workspace/realtalk-from-zero-research-20260910/RESEARCH_AND_DESIGN_ZH.md`
与 `EXECUTOR_PLAN_ZH.md`，从任务和公开数据重新设计，不继承 V9-V15 的 Self/User/Decision 产物。

核心流程：完整 Ca 建立目标人物 Self Domain；已完成的 Cb Session 低频更新五层伙伴画像；
每个目标点先联合推断 scene、动态 lambda 和唯一交流意图，再由 Actor 自然续写一整个合并轮次。

本版本不使用动作/问句/反思/气泡配额，不做候选搜索、语义验证或语义重写。Omega 和 Future
User State 关闭。模型固定 `deepseek-v4-flash` 且所有阶段关闭 thinking。

数据口径：论文 Table 8 十个 speaker-specific Ca/Cb；Ca/Cb 均使用前三个 Session；同 Session
相邻同角色消息以换行无损合并。正式规模为本地协议重建的 519 个轮次、1,076 个原始目标气泡。
论文未公布 519 这个精确数，官方仓库也尚未公开完整 Persona Simulation 代码。

开发仅使用不属于任何正式 Cb 的 Chat_4、Chat_5、Chat_10：Session 1-2 建临时 Self，
Session 3 做因果预测，按 6 -> 24 -> 68 扩大。冻结后正式 Cb 才按 10 -> 30 -> 90 -> 519 扩大。

生成与 Ground Truth 在代码中分离；任何 Prompt 构造器只接收 `GenerationInput`。`events_session_*`、
`qa`、未来消息和当前答案均不得进入 Prompt。所有运行使用新目录，旧结果保持不变。
