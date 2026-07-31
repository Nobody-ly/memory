# PersonaEmp 公开数据受控复现实验

## 实验定位

本实验只比较四种无需训练的方法：

1. `base_model`：完整 memory + query。
2. `memory`：扁平用户特征摘要 + query。
3. `rag`：与 query 最相关的 Top-3 memory + query。
4. `ours`：完整 memory + 五层画像 + 现有共情理解、预测和探索。

四种方法使用相同的 `qwen3-30b-a3b-instruct-2507`、共同回复
system prompt 和解码参数。
生成器看不到数据集已有的 persona、situation、category 或
`relevant_mem`。这些字段只用于官方评价或 RAG 的事后 Recall@3 诊断。

官方最终 `English.json`、Random/OOD ID、criteria 和 baseline 生成代码
没有公开。因此这里复现任务定义和结论，不声称复现 Table 1 的绝对分数。

## 环境

```powershell
uv sync --locked
```

PersonaEmp 官方仓库必须固定在：

```text
b555447f267b8057039aab39a4be44725718ea7f
```

API key 只设置在当前终端环境，不写入 `.env` 或代码：

```powershell
$env:PERSONAEMP_GENERATOR_API_KEY="<key>"
$env:PERSONAEMP_GENERATOR_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:PERSONAEMP_GENERATOR_MODEL="qwen3-30b-a3b-instruct-2507"
```

## 1. 重建公开数据

```powershell
python -m src.experiments.personaemp.reconstruction `
  --output-dir data/personaemp_public_reproduction/reconstruction `
  --official-repo D:\codex_workspace\.references\PersonalizedEmpathy-official
```

该命令会：

- 下载 AlpsBench Task 1 dev 和 validation；
- 使用公开 gold memory；
- 重建缺少的 intent gate，并按数据、模型、prompt、Schema 版本缓存；
- 在隔离目录运行官方 filter、persona、situation、query 和 inspection；
- 生成 `English.public-reconstruction.v1.json` 和重建 manifest；
- 记录各阶段数量、类别分布、失败原因与产物 SHA256。

Qwen 使用百炼 OpenAI-compatible `response_format=json_schema`，
并在本地再次校验字段；不会退回只靠 prompt 猜 JSON。

仅检查适配结果、不运行官方 query 流水线时使用 `--adapt-only`。

需要先验证 10--20 条正式 pilot 时，可增加：

```powershell
--source-limit 60
```

它按公开数据固定顺序只处理前 N 条原始记录，但仍完整经过 intent、
官方 filter、persona、query、inspection 和 final filter。该参数只用于
pilot；全量正式重建必须省略。模型兼容补丁的哈希写入
`model_compatibility.json`，不会修改官方 prompt 或实验协议。

## 2. 生成 Random/OOD 划分

正式 OOD 的 Big Five 标签按论文应使用 DeepSeek-v4-flash。该阶段当前
暂缓，不使用 Qwen 静默替代：

```powershell
$env:PERSONAEMP_BIG5_API_KEY="<key>"
$env:PERSONAEMP_BIG5_BASE_URL="<openai-compatible-url>"
$env:PERSONAEMP_BIG5_MODEL="DeepSeek-v4-flash"

python -m src.experiments.personaemp.splitting `
  --dataset data/personaemp_public_reproduction/reconstruction/English.public-reconstruction.v1.json `
  --output-dir data/personaemp_public_reproduction/splits
```

输出包含 Random 9:1、主 OOD、全部簇轮流留出数据和 `split_manifest.json`。

## 3. 四方法小测

先在完整重建数据上平衡抽取每类 4 条，共 12 条：

```powershell
python -m src.experiments.personaemp.cli `
  --dataset data/personaemp_public_reproduction/reconstruction/English.public-reconstruction.v1.json `
  --output-dir data/personaemp_public_reproduction/pilot_12 `
  --dataset-provenance public_reconstruction `
  --balanced-per-category 4
```

正式 Random/OOD 分别将 `--dataset` 换成：

```text
data/personaemp_public_reproduction/splits/random_test.json
data/personaemp_public_reproduction/splits/ood_test.json
```

每个运行目录必须生成四份 predictions、`results.jsonl`、`summary.json`
和 `run_manifest.json`。中断后使用完全相同的命令即可恢复。

## 4. 固定 criteria

```powershell
$env:PERSONAEMP_CRITERIA_API_KEY="<deepseek-key>"
$env:PERSONAEMP_CRITERIA_BASE_URL="<deepseek-url>"
$env:PERSONAEMP_CRITERIA_MODEL="DeepSeek-v4-flash"

python -m src.experiments.personaemp.official_eval prepare-criteria `
  --official-repo D:\codex_workspace\.references\PersonalizedEmpathy-official `
  --dataset <split-run>\evaluation_dataset.json `
  --output <split-run>\criteria.json
```

同一 split 的四种方法必须共享这一份 criteria。
命令同时生成 criteria manifest，记录模型、官方 commit、数据和 criteria
的 SHA256。

## 5. Qwen Judge 评价

```powershell
$env:PERSONAEMP_QWEN_JUDGE_API_KEY="<key>"
$env:PERSONAEMP_QWEN_JUDGE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:PERSONAEMP_QWEN_JUDGE_MODEL="qwen3-30b-a3b-instruct-2507"

python -m src.experiments.personaemp.official_eval suite `
  --official-repo D:\codex_workspace\.references\PersonalizedEmpathy-official `
  --dataset <split-run>\evaluation_dataset.json `
  --predictions-dir <split-run>\predictions `
  --criteria <split-run>\criteria.json `
  --output-dir data/personaemp_public_reproduction/evaluation `
  --split-name random `
  --judge Qwen:PERSONAEMP_QWEN_JUDGE
```

OOD 将 `--split-name` 改为 `ood` 并换成 OOD 的运行目录。

## 6. 汇总与可视化

```powershell
python -m src.experiments.personaemp.report `
  --result random:Qwen:base_model=<path> `
  --result random:Qwen:memory=<path> `
  --result random:Qwen:rag=<path> `
  --result random:Qwen:ours=<path> `
  --result ood:Qwen:base_model=<path> `
  --result ood:Qwen:memory=<path> `
  --result ood:Qwen:rag=<path> `
  --result ood:Qwen:ours=<path> `
  --output-dir data/personaemp_public_reproduction/report
```

主报告只使用官方 Resonation、Expression、Reception、Average 1--5 分。
置信区间和 RAG Recall@3 是诊断数据，不是新的主指标。
报告目录同时保存总汇 CSV、按用户 CSV、JSON、中文 Markdown 和各设置图表；
官方 Table 1 的训练免方法另存为独立参考 CSV。

## 验收

```powershell
python -m pytest tests/test_personaemp_exp1.py `
  tests/test_personaemp_public_reproduction.py -q
git diff --check
```

正式汇总前还必须确认：

- 每个 split 的四种方法 query ID 完全一致；
- 两个 Judge 对每条 query 都有三个合法分数；
- criteria 没有错位；
- Ours 核心 prompt hash 与当前生产实现一致；
- 报告明确写明“公开数据受控复现”，不与官方绝对值拼接。
