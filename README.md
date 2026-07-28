# 用户画像对话 Agent

这是一个个性化对话系统。系统会在对话过程中读取用户画像、生成个性化回复，并异步更新画像中的长期稳定信息和近期状态。

## 当前系统架构

项目由 5 个核心部分组成：

1. `app.py`：Flask 服务入口，负责页面和后端 API。
2. `frontend/`：聊天界面和画像展示界面。
3. `src/agent.py`：对话、记忆和画像接入逻辑。
4. `src/profile_batch_updater.py`：原始对话批处理、独立画像模型调用、格式校验和字段级合并。
5. `user/*_profile.json`：本地用户画像及其待处理队列。

## 用户画像更新

画像保持现有五层结构：`core`、`regulation`、`cognition`、`identity`、`behavior`。每层包含一条 `summary`，其他属性只能来自该层的既有字段白名单。

更新只读取本批原始对话，不使用被压缩的中期或长期记忆。默认累计 8 条用户消息或等待 15 分钟，任一条件满足即在后台处理。待处理消息写入画像同目录的 `.pending.json` 文件，只有画像成功校验、合并并保存后才会移出队列。

新的原始对话批处理只用于默认的 `bayesian_online` 模式。`static` 仍保持不更新，`periodic_rebuild` 仍按原来的会话间隔重建，避免改变实验对照语义。角色工作画像只在首次选择时从数据集种子创建，后续选择和服务重启会继续复用已有画像。

模型输出必须包含完整五层结构。服务端会校验层名、字段名、置信度和本批证据消息 ID，再按字段合并，禁止整块覆盖画像。失败时在同一任务上下文中带校验错误重试一次；任务结束后不保留纠错对话。

## 环境要求

- Python 3.11+
- 可访问的 OpenAI 兼容模型服务
- Windows、macOS、Linux

## 配置

主聊天模型继续使用：

```text
API_KEY=主聊天模型密钥
BASE_URL=主聊天模型接口地址
```

独立画像模型使用：

```text
PROFILE_API_KEY=画像模型密钥
PROFILE_BASE_URL=https://api.moonshot.cn/v1
PROFILE_MODEL=kimi-k2.6
PROFILE_BATCH_MESSAGES=8
PROFILE_BATCH_SECONDS=900
```

不要把真实密钥提交到仓库。项目已忽略 `.env`。

## 启动

1. 安装依赖：`pip install -r requirements.txt`
2. 在本地 `.env` 配置主聊天模型和画像模型。
3. 检查或编辑初始画像。
4. 运行：`python app.py`
5. 访问 `http://127.0.0.1:18201`

## 测试

运行离线单测：

```bash
python -m unittest \
  tests.test_profile_batch_updater \
  tests.test_interactive_profile_simulation \
  tests.test_profile_integration_boundaries -v
```

单测覆盖五层格式约束、字段级合并、任务内纠错重试、失败时不推进待处理队列，以及交互模拟的因果边界。

### 逐轮交互画像模拟

`src/experiments/interactive_profile_simulation.py` 不预写用户的 40 条输入。测试者只提供一份不向 agent 和画像提取器公开的隐藏 persona，以及一个或多个自然话题。每一轮先由 agent 根据此前对话发言，再由用户模拟模型读取这句发言并按照隐藏 persona 即时回应；只有这条新生成的用户原话会进入画像队列。模拟器只等待生产队列自己的后台 worker 完成“画像原子写入且 pending 清空”，不会手动调用处理函数。

隐藏 persona 示例：

```json
{
  "relationship": "很在意关系安全感，容易反复揣摩细节",
  "strength": "共情和表达能力较强，愿意照顾他人感受",
  "growth": "正在练习事实核对和边界表达"
}
```

运行 40 轮、每 8 条激活一次画像：

```bash
SIMULATION_API_KEY=... PROFILE_API_KEY=... python -m src.experiments.interactive_profile_simulation \
  --persona /tmp/hidden_persona.json \
  --profile /tmp/interactive_profile.json \
  --output /tmp/interactive_simulation_record.json \
  --turns 40 \
  --batch-size 8 \
  --topic "朋友临时改变约定时如何沟通" \
  --topic "任务堆积时怎样安排优先级" \
  --topic "是否购买一件昂贵但长期使用的设备"
```

用户模拟提示词禁止直接陈述性格、优缺点、心理弱点或测试目的。若仍出现这类句式，会要求模型改写为具体选择、经历或当下反应；连续失败则终止测试，不把违规消息送入画像队列。
