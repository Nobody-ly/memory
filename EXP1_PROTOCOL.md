# Exp1 实验协议与运行说明

## 实验问题

在观察到当前用户消息后，比较三种方法对用户当前情绪、情感极性和话题的理解：

1. `self_model`：使用 agent 自身 persona 推测用户状态。
2. `flat_profile`：使用因果历史生成的扁平用户画像。
3. `explicit_model`：使用相同因果历史生成的五层用户画像。

三种方法使用相同目标消息、相同短期对话和相同严格输出 Schema。Exp1 不使用
Bayesian 更新、current state、memory、RAG 或额外 baseline。

## 时间线

- 测试点位于 session 边界。
- 已完成 `session_1...session_N` 后，目标是 `session_N+1` 第一段合并后的用户输入。
- Profile 只从已完成 session 生成，不读取目标 session 或未来 session。
- 短期上下文默认取最近 3 个已完成 session，并包含目标输入之前的当前 session 消息。
- 连续的同一说话人 chat bubbles 会先合并。

## Ground Truth 和指标

- Emotion：固定 revision 的 Cardiff REALTALK emotion classifier。
- Sentiment：固定 revision 的 Cardiff REALTALK sentiment classifier。
- Topic：保留 Exp1 原有扩展指标，由同一严格 Schema 生成参考 topic。
- Emotion/Sentiment 使用严格标签相等；Topic 使用词项召回重合度。
- 主结果使用 chat-level macro average，同时保存 micro average。

## 安装与运行

```powershell
uv sync --extra realtalk-eval
python -m unittest discover -s tests -v
python -m src.experiments.exp1_user_understanding `
  --chats Chat_1_Emi_Elise `
  --max-eval-points 2 `
  --output-dir data/exp1_smoke_1x2
```

完整运行可删除 `--chats`，并将 `--max-eval-points` 调整为所需数量或 `0`。上下文可用
`--context-sessions 3`、其他正整数或 `all` 调整；`--max-context-chars 0` 表示关闭字符上限。

## 输出与恢复

- `results.jsonl`：完整成功 triplet；每个测试点只写一行。
- `summary.json`：三方法 macro/micro 指标、提升量和诊断信息。
- `run_manifest.json`：commit、源码哈希、模型、Schema、分类器 revision 和配置。
- `checkpoint.json`：原子调用缓存和恢复状态。

网络失败最多重试 6 次，Schema/解析失败最多重试 3 次。失败测试点不会记 0 分，
而是记录为 `excluded_incomplete_triplet`；恢复运行只补缺失操作，不重复统计成功结果。
