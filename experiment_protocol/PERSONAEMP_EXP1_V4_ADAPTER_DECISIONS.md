# PersonaEmp Exp1 V4 Adapter Decisions

本文件记录 76-query V1 正式结果之后、扩大运行之前冻结的 Exp1 适配决策。

## Agent Persona And Self Domain

PersonaEmp 只提供用户侧 memory，不提供 agent 侧长期对话。仓库中的
`agent_persona_generation.py` 面向 REALTALK 的第二位说话人，不能从 PersonaEmp 的用户
memory 合法生成 Agent Persona。

Exp1 因此禁用 Agent Persona 和 Self Domain。PersonaEmp 专用 Alignment adapter 会明确
禁止推断 agent 特征，并在解析后确定性地将 `understanding.self_domain` 标记为
`{"status": "disabled"}`，将 alignment 固定为 `user_domain_only`。五层用户画像、未来状态
预测、自适应探索和 empathy-state 决策继续保留。全局 Deep Empathy Prompt、REALTALK
实验以及生产 Agent 均不修改。

所有方法最终回复仍共享同一 System Prompt。该 Prompt 只规定通用陪伴者角色与行为边界，
不提供个体化 Agent Persona。

师姐最终实验设计没有单列 Self Domain 消融。仓库中的
`empathy_alignment_analysis.py` 是 REALTALK 上的补充定性分析，不属于当前 Exp1 主表。

## Omega

PersonaEmp 的 memory item 数量不是交互轮数。Exp1 不再以
`len(memory_items)` 作为时间变量，固定使用 `interaction_count=0`，关闭无法从数据确定的
时间轮次衰减，同时保留基于五层画像完整度的自适应探索。query 之间不继承状态。

## Response Length

PersonaEmp 论文及师姐 Exp1 设计没有规定回复必须为一个段落或 2--4 句。V4 从四种方法
共同的 Exp1 回复 Prompt 中移除段落数、句数及简洁要求。仍保留统一的 350-token 上限、
直接回应当前需求、禁止虚构及最多一个追问等共同约束。

这一变更影响四种方法的生成输入，因此 V4 的 76-query 结果必须全部重新生成和评价。
旧 V1/V4-server 结果仅作为历史基线，不得与 V4 adapter 输出合并计算均值。
