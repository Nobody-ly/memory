# REALTALK Evidence-Conditioned Ours：Ca Dev6 实施报告

日期：2026-09-10。性质：开发链路与定性预检，不是 Table 2 结果。

## 1. 最终可复核实现

- 分支：`paper-boost/exp2-evidence-conditioned-v1`
- 当前提交：`60f856bf69119f6efb10fdb322b62a9a916bb175`
- 协议：`realtalk_task1_ours_evidence_conditioned_v1_3`
- 模型：`deepseek-v4-flash`，所有阶段 `thinking=false`
- 正式数据静态校验：10 人、519 个合并目标轮次、1,076 个原始目标气泡
- Ca 开发池：Chat 4/5/10 的 6 位人物，Session 1-2 建 Self，Session 3 预测，共 68 条
- 本轮只运行开发 Gate 6；未运行 dev24/dev68、正式 Cb、Judge 或本地论文指标

实现保留 Self Domain、五层 User Domain、当前 scene、自适应 lambda、唯一 Behavior Policy 和
自然文本 Actor。完整对话不压缩、不裁剪；不使用动作、问句、反思或气泡配额，不做候选搜索、
语义 Verification、微调、Omega 或 Future State。

## 2. 实施中发现并修复的问题

### 证据 ID 跨文件冲突

源数据中的 `D1:1` 等 ID 会在不同 Chat 文件重复。画像与策略证据已改用
`{Chat文件名}::{session turn_id}`；原始 `dia_id` 只作审计。519 条 Prompt 静态检查确认当前
答案 ID 与后续消息 ID 均不可见。

### 新 Session 边界不可见

V1 的三条人物开场都把上个 Session 末尾话题当成当前话题。V1.1 增加只由因果位置得到的：

```json
{
  "target_session": "session_3",
  "observed_turns_in_target_session": 0,
  "starts_new_session": true
}
```

该信息在目标文本出现前已知，不含答案内容，也不规定必须问候。

### Self Domain 输出截断

V1.1 和 V1.2 均在 `self:nebraas` 连续三次达到 4,096 completion tokens，
`finish_reason=length`，运行正确标记为 incomplete。根因是画像将大量消息 ID 作为穷举证据。

V1.3 将 Self 固定为最多 12/6/6/8 条 claim/voice/social/uncertainty，每项最多 4 个代表性
source ID，并限制字段长度。完整 Ca 仍向 Decision 和 Actor 披露，所以没有裁剪历史。

## 3. V1.3 运行结果

远端输出：

`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-3-ca-dev6-60f856b`

- 6 Self、12 User update、6 Decision、6 Actor 全部完成
- 总调用 31 次（含 1 次模型预检）
- 网络重试 0、格式重试 0、unresolved 0
- Prompt tokens 170,488；completion tokens 36,001
- 6 份 Self 的最长单次 completion 从截断的 4,096 降至 2,650
- 所有 Actor 输出非空，无角色标签、JSON 或私有结构泄漏
- lambda 为 0.3 或 0.4；Gate 6 情境相近，不能据此判断全局动态性

## 4. 定性判断

新 Session 表示修复是部分有效：Emi 与 Nicolas 的人物开场改为新会话问候，但 Emi 仍较长地
续接旧书友会话题；Muhhamed 仍回答上个 Session 的电影问题。伙伴回复点整体能理解最近消息，
但常生成历史中合理、真值中不存在的新近况。后者是单参考开放续写的固有不确定性，不能简单用
禁令消除。

Self 不再截断，但 6 人均填满各 section 的数量上限，说明它仍偏密集。完整 Ca 同时可见时，
这些条目是否提供净收益尚未由本轮证明。

因此本轮结论是：

1. 数据、因果边界、Schema、缓存、失败收尾和完整调用链已经跑通。
2. 不能把 6 条定性样本称为性能提升，也不能与论文 Table 2 比分。
3. 暂不进入 dev24 或正式 Cb。下一步应先决定是否只对“新 Session 的话题连续性判断”做第二个
   预先声明的 Prompt 变体，或接受其为开放预测误差后直接用 dev24 检验整体分布。

## 5. 保留的诊断目录

- V1 成功但未显式表示新 Session：
  `/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-ca-dev6-3b3a814`
- V1.1 因 Self 截断而 incomplete：
  `/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-1-ca-dev6-aba645b`
- V1.2 因过长 evidence 列表而 incomplete：
  `/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-evidence-v1-2-ca-dev6-eae8b72`

这些目录均不得合并或覆盖；后续只从 V1.3 的提交、Prompt/Schema 哈希和新目录继续。
