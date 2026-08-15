# PersonaEmp Exp1 作者处理数据 Ours V1 锚点

## 1. 实验目标

仅在 PersonaEmp 论文的 Task 1 官方 Random/OOD 测试数据上运行 Ours，随后把 Ours 一行追加到论文 Table 1。不得重建上游 WildChat/AlpsBench，不训练 Ours，也不重跑论文基线。

师姐要求保持论文数据、划分和四项主指标不变：Resonation、Expression、Reception、Average，并分别使用 Qwen Judge 与 DeepSeek Judge。

## 2. 作者数据

本地来源：

- `C:/Users/28119/Downloads/all.json`
- `C:/Users/28119/Downloads/randomsplit-20260815T133439Z-1-001.zip`
- `C:/Users/28119/Downloads/oodsplit-20260815T133440Z-1-001.zip`

已核验：

| 数据 | 记录 | Query |
|---|---:|---:|
| all | 2,780 | 5,463 |
| Random train | 2,502 | 4,918 |
| Random test | 278 | 545 |
| OOD train | 2,536 | 4,958 |
| OOD test | 244 | 505 |

Random 与 OOD 各自的 train+test 在移除 test criteria 后均与 `all.json` 的规范化多重集完全相等。正式 Ours 只运行 545+505 个测试 Query；train 仅用于划分审计。

## 3. 论文任务边界

论文任务生成输入为：

```text
extracted_memory + current query
```

模型输出为：

```text
one assistant response
```

作者数据中的 `persona`、`situation` 和固定 `criteria` 供官方 Judge 评价使用，不属于 Ours 可见的原始生成证据。`reasoning`、`relevant_mem`、`category`、`conversation` 和 criteria 同样不得进入 Ours Prompt。

预测 JSON 必须保持作者测试文件的 session/query 顺序和原始显示 ID。由于显示 ID 并非全局唯一，运行检查点使用 `session_index + query_index + session_id + query_id` 的内部复合 ID，不修改作者数据。

## 4. Ours 最小适配

1. **Self Domain**：所有样本使用仓库 `dataset/test_agent.json` 的同一固定 Agent Persona。它只进入 Alignment，不由任何测试用户数据生成，也不直接进入最终 Actor。
2. **User Domain**：从全部 `extracted_memory.value` 一次生成 Core、Regulation、Cognition、Identity、Behavior 五层画像。完整缓存保留 evidence；Alignment/Actor 只看到 value/confidence。
3. **Current State / Prediction**：根据 memory、五层画像和 query 做单轮状态理解与回复风险规划。不得声称预测准确率、未来已发生状态或跨 Query 继承。
4. **Exploration**：`omega=0.0`，固定 exploit-only；不为收集画像而追问。保留 Exploration 字段用于结构审计。
5. **Alignment**：在固定 Self Domain 与当前 User Domain/State 之间形成一个回复策略。
6. **Actor**：输入 memory、画像 value/confidence、Alignment 和 query，输出纯回复文本。无 2-4 句限制、无 Verification、无候选搜索、无重写。

## 5. Criteria 缺口

Random test 中仅一条缺少 Expression criteria：record index 10、query index 0、`sess_c2de768bca57`、`english_52:0`。

正式处理选择“补一格，不删样本”：删除会把作者固定 Random test 从 545 改为 544，直接破坏 Table 1 测试集一致性。补全时保留原始 ZIP，只使用作者官方 `prepare_criteria.py` Prompt、`deepseek-v4-flash` 和相同解码参数生成该 Expression criteria，并将原始哈希、补丁内容、模型和 Prompt 哈希写入 sidecar manifest。不得手工编写或重生成其他 3,149 个 criteria 单元。

## 6. 最小烟雾测试

从 Random 和 OOD 各固定选择 ES、HEQ、SS 各一条，共 6 条。先验证：

- 6/6 成功、零 unresolved；
- Qwen3-8B thinking 关闭；
- 每条仅 1 次 Alignment 和 1 次 Actor，画像按记录缓存；
- gold persona/situation/reasoning/criteria/conversation 不进入任何生成调用；
- Agent Persona 对全部样本完全相同；
- omega 恒为 0.0，Exploration 恒为 exploit-only；
- 预测文件与对应抽取后的作者测试 JSON 逐位置对齐，可由官方 evaluator 读取。

烟雾测试只证明协议和链路跑通，不用于与论文 Table 1 比较。完整 1,050 条生成及双 Judge 需另建冻结运行目录。
