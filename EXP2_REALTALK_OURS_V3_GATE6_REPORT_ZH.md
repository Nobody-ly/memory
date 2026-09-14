# REALTALK Ours V3 Gate 6 阶段报告

日期：2026-09-14

## 固定版本

- 协议：`realtalk_task1_ours_behavior_calibrated_v3`
- 实现提交：`3b186af`
- 模型：`deepseek-v4-flash`
- thinking：关闭
- 数据：REALTALK Task 1，Table 8，Ca/Cb 前三个连续 Session
- Gate：6 条，覆盖 4 位人物和 3 个 Session
- 服务器输出：`/amax/xidian_ty/Ly/personaemp-exp2/runs/realtalk-ours-v3-gate6-3b186af`

## Gate 6 结果

- 成功预测：`6/6`
- unresolved：`0`
- `GENERATION_COMPLETE`：已生成
- Self/User/Decision/Actor：均有审计记录
- 历史压缩：关闭
- 历史截断：关闭
- 生成结果回灌：关闭
- Ground Truth/Judge 标签进入生成：关闭
- GPU：未使用

## 早期失败与修正

1. Actor 在无问题槽位时仍生成问句。修正为在 Actor Prompt 中显式写入问题合同：零问题槽位时不得出现问号或问题；同时保留确定性拒绝和最多两次重试。
2. User Domain 曾引用目标人物自己的证据 ID。修正为同时显示允许的伙伴 ID和禁止的目标人物 ID，并明确 User Domain 只描述伙伴。
3. Nebraas 的 Self Domain 在 4096 token 上限内被截断。保持 4096 参数不变，增加紧凑输出预算：identity 6、voice 4、social 4、uncertainties 4，每个 scene 最多 4 个证据 ID；不足证据的 scene 返回空对象。

## 结论

修正后的 Gate 6 已通过结构和因果边界检查，可以进入 Gate 18。Gate 6 只验证协议、Schema、证据边界和 Actor 合同，不据此判断八项评价指标；Gate 18 完成后再决定是否进入 Gate 30 和 Judge。
