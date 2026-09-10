# REALTALK Ours V15：Cb 后验行为控制器

## 1. 实验身份

- 当前协议：`realtalk_task1_ours_v15_17_cb_posterior_controller`
- 模型：`deepseek-v4-flash`，Controller 与 Actor 均关闭 thinking
- 基础：冻结 V9 Self Domain、逐样本五层 User Domain 和完整因果历史
- 重建：重新生成 Behavior Controller 与 Response Actor
- 状态：protocol-aligned exploratory comparison

V9、V10–V14 的代码与产物保持不变。V15 不读取 V9 Decision、V9 生成文本、Cb
未来消息、Ground Truth、Judge 标签或论文指标。

## 2. 方法

```text
Self Domain
+ 五层 User Domain（最多激活两条）
+ Ca 聚合行为弱先验
+ 当前点前 Cb 目标人物行为
+ 完整真实历史
        ↓
Behavior Controller
        ↓
动态 lambda + 1–6 个 turn units
        ↓
Response Actor
        ↓
确定性结构检查；仅违约时重试 Actor
```

Ca 只提供消息长度、气泡、问题、自我披露与结构统计，不提供原始相似文本。Cb 统计只读取
当前目标点前已经发生的真实目标人物消息；Cb 证据充分且与 Ca 冲突时优先当前 Cb。

`turn_units` 是 Actor 唯一的动作与问题控制源。lambda 是同次 Controller 决策的可审计轨迹，
不使用固定阈值或公式生成文本。

为避免把普通的“继续说一句”误标为追问，机器 Schema 使用无歧义的
`clarification-question`、`follow-up-question` 和 `reciprocal-question` 标签；这三类单元必须
明确给出提问对象，其他动作不得携带问题。
`topic-shift-statement` 明确表示陈述式换题；通过问题换题仍须使用上述提问动作。User Domain
激活只输出稳定 `fact_id`，程序再解析回原始层级和值，避免模型改写画像事实。

Ca 内部开发的临时 User Domain 为避免服务端长输出截断，每层保留最多 3 条最强事实；三个
Session 的源文本仍逐字完整输入。该限制只用于 Ca Prompt 开发，正式 Cb 继续复用 V9 已冻结的
逐样本完整五层 User Domain。
模型偶尔输出的 `s1:t6` 仅确定性规范为等价源 ID `session_1:turn_6`；原始响应保留审计，
规范化后仍必须通过“只引用已观察伙伴消息”的证据校验。

## 3. 开发与冻结

先进行 Ca 内部开发：

1. Ca Session 1–2 建立临时 Self/User Domain；
2. Ca Session 3 构造因果预测点；
3. 运行 6 条结构预检，再扩展到 30 条、10 位人物；
4. 只允许在 Ca 开发阶段调整 Prompt；
5. Prompt、Schema 和模型参数冻结后才进入 Cb。

Ca 开发结果不进入 Table 2。正式 Cb 使用 V9 基于 Ca 前三个 Session 建立的 Self Domain。

## 4. Cb 渐进 Gate

固定嵌套清单：`6 → 18 → 30 → 60 → 120 → 519`。选样只基于人物、Session 和位置，
不使用文本、V9 错误或 Judge 分数。同一版本只补新增样本。

- Gate 6/18：结构、因果边界、身份、调用可靠性和人工输出检查。
- Gate 30：10 人 × 3 Session × 1 条，首次全覆盖配对评价。
- Gate 60：每个 speaker-session 两条，检查三项目标至少两项改善和其余指标非劣。
- Gate 120：要求三项目标点估计全部改善、至少 7/10 人物方向一致。
- Gate 519：完整 protocol-aligned exploratory Table 2 对比。

进入 Cb 后不得修改 Prompt。任何修改必须升级为 V15.x、新目录并从 Gate 6 重新开始。

## 5. 评价

沿用 REALTALK 八项指标：ROUGE、BERTScore、Reflectiveness、Grounding、Sentiment、
Emotion、Intimacy AD、Empathy AD。GPT 三项复用 Appendix C Prompt，V9/V15 共享 Ground
Truth 判断。

主汇总使用 speaker macro mean 和 population std。配对分析固定 seed `20260909`，按 10 位
speaker 聚类 bootstrap 10,000 次。AD 指标反向后统一以正值表示 V15 改善。

额外诊断：Reflectiveness/Grounding Precision、Recall、F1；Intimacy 有符号误差；逐人物、
逐 Session、Actor 重试率和调用 token。

## 6. 完整目标

| 指标 | V15 完整 519 条目标 |
|---|---:|
| ROUGE | `> 0.14` |
| BERTScore | `> 0.78` |
| Reflectiveness | `> 0.77` |
| Grounding | `> 0.62` |
| Sentiment | `> 0.59` |
| Emotion | `> 0.46` |
| Intimacy AD | `< 0.06` |
| Empathy AD | `< 1.24` |

未达到时保存并如实报告，不继续利用同一 519 条调 Prompt。

## 7. V15.14 事实归属修正

V15.13 的 Ca 6/30 均通过，但 Cb Gate 6 仍有一条 Controller 把伙伴的 New York
近况写成目标人物经历。V15.14 在 Controller 前增加确定性的事实归属风险提示：比较伙伴最新一轮
与当前点以前的目标人物发言及稳定身份，只列出伙伴独有词项，以及只在伙伴同句出现、未在目标
证据同句出现的词项关系。该机制不写死地点、天气或具体样本，不新增模型调用，也不读取未来、
Ground Truth、Judge 或 V9 生成结果；它只提醒 Controller 不要把伙伴事实改写成目标人物自传。

输出后原有事实归属审计继续作为阻断检查。V15.14 必须重新完成 Ca 6、Ca 30 和 Cb 6，不能
复用 V15.13 的生成结果。

## 8. V15.15 Controller 可靠性修正

V15.14 的 Ca 6/30 均通过；Cb Gate 6 在 4/6 停止。Emi 的 Controller 已正确识别不受支持的
New York 前提，但重复复述行为统计直至 1600-token 上限，三次均形成截断 JSON。Muhhamed 的
Controller 则把确认和未经整体证据支持的“我的办公室也很冷”自我披露塞进同一 acknowledge
单元，随后被事实归属审计正确阻断。

V15.15 要求 `decision_basis` 不超过 120 词、各计划槽不超过 40 词、完整 JSON 不超过 900
token，且本地 Schema 校验同样执行前两项上限。Controller 的格式修复仍携带具体解析错误，
但不再把已截断的超长坏 JSON 回灌给模型。每个自传关系必须由一条目标人物发言或一个稳定
Self Domain 项整体支持；acknowledge 单元不得隐藏 self-disclose 动作。该修改不增加模型调用、
不读取未来或 Judge，也不改数据和 V9 上游状态。V15.15 从 Ca 6 重新开始。

## 9. V15.16 审计语法修正

V15.15 的 Ca Gate 6 通过，Ca Gate 30 在 29/30 停止。唯一失败是审计器把被动结构
`coding can be used to create...` 中的 `used to` 错当成第一人称过去习惯，尽管该回答已由可见
目标人物历史支持。V15.16 只把明确的 `I used to` 识别为过去习惯；`be used to create` 不再
触发回溯声明。Prompt、Schema、数据、模型和解码参数均不变，新增回归测试后从 Ca Gate 6
重新验证。

## 10. V15.17 否定归属与问题优先级

V15.16 的 Ca 6/30 均通过；Cb Gate 6 在 5/6 停止。Emi 的 Controller 已不再采用伙伴的
New York 前提，而是生成否定句 `I'm not living in New York`，被旧审计器误判为正向事实转移。
V15.17 将明确的第一人称否定声明排除出“采用伙伴事实”告警。与此同时，当合并伙伴消息的早期
问题前提不受支持、而最后一个问题可以独立回答时，Controller 必须静默略过早期问题，不为纠正
而主动复述该实体。只有剩余问题无法连贯回答时才纠正前提。该修改不放宽任何正向自传事实。

## 7. V15.1 实施审计记录

首个冻结 Prompt 版本完成 Ca 30：30/30、零 unresolved、零事实归属警告。随后 Cb Gate 6
完成 6/6，Cb Gate 18 完成 18/18，但人工审计确认 3 条启发式事实归属警告中有 1 条是真实的
伙伴当前事实向目标人物迁移。原实现只记录 warning 而仍生成完成标志，与本计划“仍有归属问题
则 unresolved 并停止 Gate”不一致，因此该 Cb 目录只保留为失败诊断，不能继续 Gate 30。

V15.1 协议升级为 `realtalk_task1_ours_v15_1_cb_posterior_controller`，只做两项修正：

- 事实归属启发式仅检查第一人称陈述句与伙伴最新事实的独特重合，排除问题和无第一人称的
  泛化评论，降低误报；
- 命中后不自动改写语义，直接记录样本 unresolved 并停止当前 Gate。Controller/Actor 同时
  明确禁止把伙伴的地点、天气、活动、计划、健康、工作和物品镜像为目标人物当前事实。

V15.1 必须使用新目录并重新从 Ca Gate 6 开始，不能复用原 V15 的 Cb 生成缓存。

### V15.2 lambda 语义修正

V15.1 的 Ca Gate 30 在 29/30 停止：同一 Decision 三次输出
`partner-adaptive + lambda_trace=0 + affected_dimensions=[]`。原始理由显示，当前伙伴的脆弱披露
触发了适应性回应，但该回应也完全符合人物稳定 Self prior，因此相对默认行为的偏离量确实可为
零。此前新增的“非 self-led 必须非零 lambda”并非原计划要求，错误地把离散 orientation 当成
lambda 数值区间。

V15.2 保持 orientation 表示本轮总体取向，lambda 只表示相对 Self prior 的偏离程度：

- `lambda_trace=0` 时 `affected_dimensions=[]`；
- `lambda_trace>0` 时至少列出一个实际受影响维度；
- 不再依据 `self-led|balanced|partner-adaptive` 人为设置 lambda 区间。

事实归属阻断和其余 V15.1 改动保持不变；V15.2 使用新目录重新从 Ca Gate 6 开始。

### V15.3 身份可见性与问题前提修正

V15.2 首次 Cb Gate 6 正确阻断了一条伙伴事实迁移：伙伴询问 New York，Controller 直接让 Emi
把 New York 当成自己的现居地，而冻结 Self Domain 实际记录 Emi 在 Los Angeles。审计同时发现
Actor 使用的是旧 behavioral-only 投影，没有看到计划要求的 `identity_context` 与稳定边界。

V15.3 因此：

- Actor 的私有 Self Domain 增加 `identity_context` 和 `boundaries_and_uncertainty`，仍不披露 User
  Domain、lambda、Judge 或 Ground Truth；
- Controller 回答涉及目标人物的伙伴问题前，必须先用 Self Domain 和目标人物既往 Cb 发言验证
  问题前提；冲突或无依据时不得接受该前提；
- 事实归属审计把合法 Self Domain 事实加入目标人物证据池；
- 单个 question unit 必须对应一个信息槽，禁止把姓名和来源等两个问题用 `and` 捆绑。

V15.2 的失败 Cb 目录保持不变，V15.3 再次从 Ca Gate 6 开始。

### V15.4 Self Domain 证据边界修正

V15.3 Cb Gate 6 的结构检查通过，但人工复核发现 Self Domain 内部的伙伴示例污染了身份判断：
Emi 的 `identity_context` 明确记录 Los Angeles，交互策略描述却以伙伴适应 New York 为例；
Controller/Actor 生成了 Emi 正在适应 New York，审计器又因整份 Self Domain 出现 New York 而误放行。

V15.4 明确分工：

- Controller 仍读取完整 Self Domain，但额外单列权威 `identity_context` 进行问题前提核对；
- Actor 只读取 `identity_context`、`communication_signature`、稳定边界和可观察统计；动作选择、
  伙伴适应和情绪策略已经由 turn plan 决定，不再向 Actor 重复披露含伙伴示例的策略描述；
- 事实归属审计仅将 `identity_context` 视为稳定人物事实，不把行为描述中的伙伴案例当作目标事实。

V15.3 的 Cb Gate 6 保留为失败诊断；V15.4 使用新目录重新运行 Ca Gate 6。

### V15.5 目标事实归属索引与组合审计

V15.4 完成 Ca Gate 30，随后 Cb Gate 6 的结构检查也通过；但人工复核发现 Controller 仍可能从
完整交错历史中混淆事实所有者。三条初始疑例经逐条因果历史复核后结论不同：Akib 此前已经由
自己陈述当地寒冷和结冰，不属于迁移；Emi 的“正在享受 New York”没有目标人物明确当前事实；
Muhhamed 的“我的办公室也很冷”把伙伴同一陈述中的办公室和寒冷关系组合成了自己的事实。

V15.5 因此只修正证据归属，不改变计划空间：

- Controller 在完整因果历史之外额外看到按 turn ID 列出的“目标人物自己此前说过的话”，帮助
  区分交错历史中的事实所有者；完整历史仍全部保留，不压缩、不检索、不读取未来；
- 确定性审计对地点、天气和工作场所等所有权敏感概念做同义归一，并检查伙伴陈述中的概念组合
  是否曾由目标人物在同一陈述中建立；单独出现过两个概念不等于该关系已经成立；
- 已被目标人物先前陈述支持的事实继续放行，命中无支持的镜像关系则阻断当前 Gate，不自动改写。
- Ca 开发 runner 与 Cb runner 调用同一事实归属审计；不再以固定零警告代替真实检查。

V15.4 的 Ca/Cb 目录作为诊断保留。V15.5 使用全新目录从 Ca Gate 6、Ca Gate 30、Cb Gate 6
重新开始，人工事实归属审计通过后才允许进入 Cb Gate 18。

### V15.6 新建议的时间归属修正

修正真实 Ca 归属审计后，V15.5 Ca Gate 6 在 5/6 停止：伙伴刚刚建议目标人物开 YouTube
频道，生成结果却声称目标人物“以前已经想过”，把当前建议错误地倒签成目标人物既有经历。

V15.6 允许目标人物接受、拒绝或暂时考虑伙伴的新建议，也允许据此形成未来计划；但如果目标人物
自己的可见历史没有证据，不得声称自己以前已经考虑、计划、执行或拥有该建议内容。Ca 和 Cb
runner 的协议标识同步升级为 V15.6，并从全新 Ca Gate 6 重新开始。

### V15.7 结尾赞美动作补全

V15.6 Ca Gate 6 达到 6/6、零归属警告，但相同 Muhhamed 样本仍被 Controller 错分为
`support-request`。原因是伙伴消息先提问，随后以建议和“你会做得很好”的赞美收尾，而严格
Schema 没有表示赞美或鼓励的伙伴动作。

V15.7 新增 `praise-or-encouragement`，明确当合并伙伴轮次以赞美收尾且没有更晚的待答问题时，
当前义务应为 `acknowledge`，不能把对目标人物的赞美误判成伙伴寻求支持。Schema、协议和 Prompt
同步升级，并从全新 Ca Gate 6 重新开始。

### V15.8 归属审计语义校准

V15.7 正确把 Muhhamed 的伙伴结尾识别为赞美，并规划了简短 acknowledge；但 Ca Gate 6 被
两个审计误报阻断：其一是 Akib 对 Godzilla 电影的外部观点，其二是 Muhhamed 对 YouTube
建议的未来考虑。这两者都不是把伙伴事实冒充成目标人物既有经历。

V15.8 保持 Controller、Schema 和 Actor Prompt 不变，仅让确定性审计放行两类合法表达：没有
第二个第一人称主张的外部观点，以及不含“以前/已经/一段时间”等倒签标记的暂定未来反应。
对“我早已计划开 YouTube 频道”等倒签陈述仍然阻断。协议和 Ca runner 标识同步升级，并从
Ca Gate 6 重新验证。

### V15.9 通用词重合误报修正

V15.8 Ca Gate 6 又阻断了 Akib 对外国电影和 Hollywood 的不同个人习惯。该输出围绕伙伴话题
表达目标人物自己的不同倾向，并没有把伙伴事实据为己有；旧审计仅因两个主题词重合而误报。

V15.9 对地点、天气和工作场所关系等所有权敏感概念保持原有阻断；通用主题词重合只有同时出现
`too/also/same/as well` 等明确镜像语言时才触发。这样仍会阻断“我这里也很冷”等迁移，但不会
阻断围绕同一电影话题表达不同观点或习惯。协议标识同步升级，并从 Ca Gate 6 重新验证。

### V15.10 否定经历与倒签动词区分

V15.9 Ca Gate 6 把 `I haven't watched ... but I've heard good things` 中的 `I haven't`
错误匹配为 `I have`，进而误判为倒签经历。V15.10 将倒签信号限定为 `already/before/used to/
for a while`，或“以前考虑、计划、开始、做过、尝试、拥有”等明确动词。否定观看经历和“听说
口碑不错”不再被当作伙伴事实迁移，真正的历史倒签仍阻断。协议标识同步升级并重跑 Ca Gate 6。

### V15.11 赞美义务与计划内容一致性

V15.10 Ca Gate 6 暴露两个不同问题：Emi 合法地回答自己也喜欢科幻电影，却被通用 `also`
重合规则误报；Muhhamed 虽被正确分类为结尾赞美和 `acknowledge`，但 acknowledge 的自由
`content_slot` 仍回答了更早的问题并声称“以前想过”。

V15.11 删除通用主题词的 `too/also` 阻断，只保留所有权敏感概念、关系组合和明确倒签检测。
同时在 Decision 归一化后强制：`praise-or-encouragement + acknowledge` 只能规划 acknowledge
单元，且内容不得虚构此前已经考虑或计划过伙伴刚提出的建议。违反时由同一次结构化调用的既有
重试机制修复，不增加新模型阶段。协议标识同步升级并从 Ca Gate 6 重跑。

### V15.12 纯社交 acknowledge 优先级

V15.11 的一致性校验正确拒绝了错误计划，但三次格式修复都返回相同内容：Decision 理由已经判断
结尾赞美覆盖了更早问题，`content_slot` 却仍受一般“分享自己”先验影响，继续回答 YouTube
建议。V15.12 在 Controller 主 Prompt 中明确优先级：当当前动作是结尾赞美且义务为 acknowledge，
只规划简短感谢或欣赏，不提及、回答、评价、接受或拒绝更早建议，也不增加自我披露、理由或未来
计划。既有一致性校验继续兜底，不新增调用阶段。协议同步升级并从 Ca Gate 6 重跑。

### V15.13 多问题优先级与事实关系边界

V15.12 通过 Ca Gate 30，但 Cb Gate 6 在 4/6 停止。两条错误都来自 Controller：Emi 的合并
伙伴轮次包含三个问题，Controller 机械回答全部问题并接受未确认的 New York 当前前提；
Muhhamed 的历史分别出现过 home office、感冒和寒冷，Controller 却把这些分散概念与伙伴的
办公室暖气故事组合成“我的办公室也很冷”。

V15.13 明确：合并轮次有多个问题时，只优先最后仍活跃且前提有证据的槽位，不要求逐个完整回答；
目标人物在不同陈述中分别出现两个概念，不构成二者关系的证据，也不能据此编造与伙伴平行的
个人经历。关系没有目标人物证据时，只回应伙伴经历。协议同步升级，并重新执行 Ca Gate 6/30
后再进入 Cb Gate 6。
