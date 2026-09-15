# V3.2 困难连续窗口稳定性修复

## 不变项

模型 deepseek-v4-flash，thinking 关闭；Table 8、Ca/Cb 前三个 Session、全部真实历史、五层 User Domain、Self Domain、自适应 lambda、Judge Prompt 均不变。不用生成文本回灌历史，不读取目标答案，不使用 GPU。

## 审计纠正

- 原始 after:2 失败响应的 finish_reason=length、completion_tokens=4096，证明是输出截断。
- 旧 JSON 扫描器在外层解析失败后继续扫描内层，可能错误返回内部数组。“模型返回单元素数组”的历史解释没有证据，不应继续沿用。
- 旧 hard6 只使用连续窗口前六条，均在 Session 2，不会触发供 Session 3 使用的 after:2 更新。其成功不能证明 after:2 已修复。
- 一个失败样本可同时产生 operation 和 sample 两条记录，不能把它说成两个不同样本。
- 关闭词汇式反思拒绝只是合同变化，不是性能改善证据，更不是短反思长度约束。

## 本次实现

1. 新协议 realtalk_task1_ours_behavior_calibrated_v3_2_contract_reliability。部署到独立代码快照和新运行目录，不覆盖旧运行。
2. JSON 仅接受完整根值或完整代码块，不扫描内部对象，不自动解包数组。输出截断在解析前拒绝。
3. 结构化 API 请求之后，启用 Draft202012Validator 全量 schema 校验，再检查画像证据、提问关系等语义合同。
4. required_content_slots 为非空的 1–4 项数组；message_length 为 short/typical。message_shape 不由模型重复决定，程序由数组长度推导。数组项不等同于气泡数。
5. Self Schema 的数量上限与现有 compact normalizer 对齐；User Domain 输出预算 8192，不裁剪输入。
6. 结构化最多三次；Actor 最多三次（首次 temperature=0.6，合同修复 temperature=0）。保留原计划和完整输入，不进行多候选择优。网络层最多三次，关闭 SDK 隐式重试。
7. 保留未经授权的问题、身份标签、私有结构泄漏和空输出拒绝。词汇式反思不作硬拒绝，真实效果由 Judge 判断。
8. parent6 扩到60时复用成功的画像、操作和预测；要求 protocol、Prompt、Schema、实现哈希、解码全部一致。

## 冻结测试

源窗口沿用历史 Vanessa 连续60条（Session 2 共32条、Session 3 共28条）。它是按V9困难程度事后选取的诊断集，不能替代主表或独立验证。

- 结构预检选该窗口下标 0/7/23/31/32/59，覆盖旧 Actor 失败点、两个 Session 及第二次画像更新。
- 所有519条重建 ID、历史哈希、Ground Truth 都先核对 canonical V9。V9 predictions SHA256 必须为 ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81。
- 评分只从 canonical V9 full519 judge 获取；不把 V14 judge 目录误当 V9。
- 六条通过结构/样本检查并人工检查后扩到60，只新增54条。生成出现 unresolved 即停，不自动从头重跑。
- Judge 固定现有 gpt-4o-mini、Appendix C Prompt，共享同ID Ground Truth 标签。Intimacy AD 使用已有本地分类器，不由GPT替代。
- 六条分数只作排错。60条若有效，再考虑其他人物的冻结连续窗口；单个人物结果不能授权完整519条直接扩大。

## 验证

已运行针对解析截断、完整代码块、单元素数组拒绝、schema额外字段、可变数组、确定性组合推导、合同修复、缓存复用、重试上限和现有Judge的测试。最终执行状态写入新运行目录 stage6_status.json / stage60_status.json；失败保留原始响应。
