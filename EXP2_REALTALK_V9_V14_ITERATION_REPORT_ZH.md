# REALTALK Exp2 V9-V14 迭代报告

## 1. 冻结基线

- V9 实现提交：`5927bbff03fda74eebaeb99e0c57203a644cfd74`
- V9 生成目录：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v9-full519-evidencefix-flash-5927bbf`
- V9 预测 SHA256：`ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81`
- 完整度：519/519；Judge 3114/3114；零 unresolved
- V9 完整结果：ROUGE 0.154、BERTScore 0.858、Reflectiveness 0.698、Grounding 0.596、
  Sentiment 0.628、Emotion 0.524、Intimacy AD 0.072、Empathy AD 1.215。

V14 runner 启动时强制校验上述提交、SHA256、记录数、上下文、Ground Truth 和 Self Domain
哈希。V9 目录从未被覆盖或修改。

## 2. 最终有效改造：V14.12

V14.12 提交：`60c496fd140356c28ffe15abcd3fab6e76d25491`。

实现原则：

- 冻结 V9 Self Domain、User Domain 及 Session 更新；
- 从论文 Table 8 指定 Ca 的前三个 Session 构建目标人物行为样例库；
- Ca 检索不读取 Cb future、Ground Truth 或 Judge 标签；
- 新 Decision 使用动态 lambda、关系语气、反思深度、气泡和问题槽位；
- V9 `primary_move/content_direction/question` 映射为不可变语义合同，防止新 Decision
  创造当前自传事实或更换话题；
- Actor 不读取 V9 生成文本；
- 气泡归一化只把已有句间空白替换为换行，词序和词元保持不变；
- DeepSeek-v4-flash，thinking 关闭，完整真实历史，无压缩、无未来泄漏。

结构检查：Gate 6 与 Gate 18 均为零 unresolved；Gate 18 气泡 18/18 一致，问句权限
17/18 一致。唯一不一致是 Actor 在无问句计划下额外生成一个真实信息问句，已记录审计。

## 3. Gate 18 配对结果

18 条覆盖 10 位人物，V9 与 V14.12 使用完全相同 result ID；Ground Truth Judge 标签从 V9
checkpoint 复用，每条只判断一次。GPT 指标使用论文 Appendix C Prompt 和同一
`gpt-4o-mini` 端点。以下子集结果只用于诊断，不能作为 Table 2 主结果。

| 指标 | 论文逐列最优 | V9 同 18 条 | V14.12 | V14.12 - V9 |
|---|---:|---:|---:|---:|
| ROUGE | 0.140 | 0.149 | 0.134 | -0.015 |
| BERTScore | 0.780 | 0.854 | 0.850 | -0.005 |
| Reflectiveness | 0.770 | 0.750 | 0.650 | -0.100 |
| Grounding | 0.620 | 0.450 | 0.700 | +0.250 |
| Sentiment | 0.590 | 0.700 | 0.650 | -0.050 |
| Emotion | 0.460 | 0.550 | 0.650 | +0.100 |
| Intimacy AD（低优） | 0.060 | 0.0585 | 0.0596 | +0.0011 |
| Empathy AD（低优） | 1.240 | 1.500 | 1.150 | -0.350 |

V14.12 在该子集超过论文逐列最优 6/8；未通过项是 ROUGE 和 Reflectiveness。相对同 ID
V9，Grounding、Emotion 和 Empathy 明显改善，但 Reflectiveness 下降。

Reflectiveness 误差分析：参考答案中 5/18 被判为反思，V14.12 候选中 10/18 被判为反思；
混淆为 TN=7、FP=6、TP=4、FN=1。主要问题是普通动作被 Actor 扩展成额外感受、评价或理由，
不是缺少反思。

## 4. Gate 30 均衡扩容验证

为检验 Gate 18 是否受小样本波动影响，V14.12 在 Prompt、Schema、模型和解码配置完全不变
的条件下扩展到 30 条。样本覆盖 10 位人物，每人 3 条；三个 Session 各 10 条，构成 30 个
不同的 speaker-session 单元。V9 与 V14.12 仍严格使用相同 result ID，Ground Truth Judge
标签继续共享。

| 指标 | 论文逐列最优 | V9 同 30 条 | V14.12 | V14.12 - V9 |
|---|---:|---:|---:|---:|
| ROUGE | 0.140 | 0.137 | 0.130 | -0.008 |
| BERTScore | 0.780 | 0.853 | 0.851 | -0.002 |
| Reflectiveness | 0.770 | 0.633 | 0.567 | -0.067 |
| Grounding | 0.620 | 0.567 | 0.667 | +0.100 |
| Sentiment | 0.590 | 0.633 | 0.600 | -0.033 |
| Emotion | 0.460 | 0.467 | 0.533 | +0.067 |
| Intimacy AD（低优） | 0.060 | 0.0656 | 0.0622 | -0.0035 |
| Empathy AD（低优） | 1.240 | 1.267 | 1.200 | -0.067 |

V14.12 在 Gate 30 超过论文逐列最优 5/8；未通过项为 ROUGE、Reflectiveness 和 Intimacy AD。
相对同 ID 的 V9，Grounding、Emotion、Intimacy AD 和 Empathy AD 改善，但 ROUGE、
BERTScore、Reflectiveness 和 Sentiment 下降。尤其 Reflectiveness 从 Gate 18 的 0.650 降至
0.567，表明 Gate 18 的局部结果不能外推为稳定提升。

结构审计同样未完全通过：30 条气泡计划均匹配，但问题权限仅 28/30 匹配；两条 Actor 在
Decision 未授权时自行追加问题。因此 V14.12 不进入 Gate 60，避免在已知未过 Gate 30 的
版本上继续消耗评价成本。

## 5. 后续尝试与停止

V14.13 将 Actor 温度由 0.6 降到 0.3，并收紧普通动作的反思和长度规则。Gate 6 仍出现明显
扩写，说明退化不是采样温度造成。

V14.14 只让 Actor 读取完整历史、行为化 Self Domain 和 Turn Plan；User Domain、Ca 类比、
Situation 和 lambda 只在 Decision 生效，不二次注入 Actor。Gate 18 完成 18/18、零 unresolved，
但结果为：ROUGE 0.130、BERTScore 0.849、Reflectiveness 0.600、Grounding 0.550、Sentiment
0.800、Emotion 0.450、Intimacy AD 0.052、Empathy AD 1.350。它相对 V14.12 多项下降，按渐进
门槛停止，不进入 Gate 30。

## 6. 产物位置

- V14.12 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-progressive-v1-60c496f`
- V14.12 Gate 18 快照：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-gate18-snapshot-v1-60c496f`
- V14.12 本地指标：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-gate18-local-v1`
- V14.12 Judge：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-gate18-judge-v1`
- V14.12 Gate 30 快照：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-gate30-snapshot-v1-60c496f`
- V14.12 Gate 30 本地指标：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-gate30-local-v1`
- V14.12 Gate 30 Judge：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-12-gate30-judge-v1`
- V14.14 生成：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-14-progressive-v1-78fc884`
- V14.14 本地指标：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-14-gate18-local-v1`
- V14.14 Judge：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v14-14-gate18-judge-v1`

## 7. 当前判断

V14.12 是应保留的诊断候选，但 Gate 30 已确认它不足以升级为完整实验版本。V14.13/V14.14
证明单纯降低温度、增加负向规则或减少 Actor 输入不能解决 Reflectiveness；V14.12 的均衡
扩容又证明 Gate 18 的优势不稳定。下一轮应从冻结的 V9 重新建立新版本，使用未见样本验证
新的内容实现机制，或更换更能严格执行 Turn Plan 的 Actor 模型，并重新从 Gate 6 开始，
不能把 V14.12 继续扩到 60 或 519 条。
