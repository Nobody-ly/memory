# V3.2.1 其他人物连续窗口验证

## 冻结范围

继承 `4302989` 的生成实现、Prompt、Schema、模型和解码参数。
`deepseek-v4-flash`，全部关闭 thinking，CPU-only。
Self Domain、五层 User Domain、状态、动态 lambda、策略和最终输出均按原实现运行。
不调整提示词，不读旧答案指导生成，不回灌生成历史，不覆盖任何旧运行。

启动脚本逐项校验上一轮 manifest 的 protocol、model、prompt_hashes、schema_hashes、implementation_hashes、decoding。
本轮新增的只有选样、执行编排和分人物结果汇总。

## 样本

按 Table 8 顺序固定 Emi、Nicolas、Kevin，各自 Cb 前20条目标消息，共60条。
每人内部为源顺序连续前缀，不按 Judge 标签、错误、难度或候选结果挑选。
与上一轮 Vanessa 60条完全不重叠。

- Emi：Session 1 共18条，Session 2 共2条。
- Nicolas：Session 1 共20条。
- Kevin：Session 1 共14条，Session 2 共6条。

这是其他人物、较早对话位置的泛化诊断，不覆盖全部三个 Session。
519条此前参与过方法诊断，不能称为完全独立确认集。

## 评价

核对所有519条真实目标和历史哈希与 canonical V9 一致；只提取相同ID的 V9 输出。
三项GPT评价使用现有 gpt-4o-mini 和既有论文 Prompt，与V9共享参考答案标签。
V9预测标签来自冻结的历史完整Judge，不重新抽取另一版本结果冒充V9。
五项本地指标使用现有固定实现与模型；八项全部披露。
主汇总为 speaker macro mean，另保存每人20条结果。本轮人数相同，macro等于micro。

任一生成出现未解决错误则停止，不自动无限重启。不因低分删除样本。
完成60条后先报告，不自动运行完整519条或调整当前Prompt。

## 上一轮参考

Vanessa困难连续60条全部生成和八项评价完成，零unresolved。
同样本V9到V3.2.1：Reflectiveness 0.6500到0.6667；Grounding 0.5167到0.7000；
Empathy AD 1.3167到1.1833；Intimacy AD 0.081650到0.081068。
不能将单人人物困难窗口的改善视为整体性能结论。

## 运行与产物

入口：`tools/run_realtalk_v321_other_windows.sh OUTPUT_ROOT`。
独立运行目录内保存 selection.json、ids60.json、generation60、evaluation60、stage_status.json。
`--evaluate-only` 仅允许完整且无未解决错误的生成结果进入评价，不重新生成。
完整生成模块不变，旧V9、Vanessa及所有历史运行保持原状。
