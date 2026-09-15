# 模型选择指南

本项目把模型服务分成两类：Ollama 在自己的机器上运行开放模型，OpenAI-compatible API 把请求发送到 OpenAI 或其他兼容服务。服务的模型目录会变化，所以这里链接官方目录，而不复制一份会过期的清单或价格。

## 先记住两个模型

embedding model（嵌入模型）把文档片段和问题转换为向量，用于检索；chat model（对话模型）读取 Top-K 片段并组织答案。二者职责不同，不能因为某个 chat model 写作好就直接拿来替代 embedding model。更换 embedding model 后应重新为整份 PDF 建立嵌入；同一索引中的文档和问题必须使用同一个嵌入模型。

## 快速比较

| 方案 | 隐私 | 安装 | 成本 | 离线使用 | 速度 | 硬件 | 适合选择 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ollama + 本地模型 | PDF/问题留在本机（除非模型或系统另有联网行为） | 安装 Ollama、下载模型 | 无 API 按量费，但消耗磁盘/电力 | 可以，模型已下载后可断网运行 | 取决于 CPU/GPU/RAM；首次加载较慢 | 需要足够 RAM，GPU 可加速 | 学习、隐私优先、反复试验 |
| OpenAI API | 内容发送到你配置的服务，按其政策处理 | 只需 API key 和网络 | 按服务商当前定价计费 | 不可以 | 通常无需本机加载，网络和服务端延迟仍存在 | 本机硬件要求低 | 快速试用、没有本地硬件、托管部署 |
| 其他 OpenAI-compatible API | 取决于供应商 | base URL、key、模型名各不相同 | 取决于供应商 | 通常不可以 | 取决于网络和服务 | 本机硬件要求低 | 已有内部网关或其他供应商 |

表格只描述通常的权衡，不代表任何服务的隐私承诺或性能保证。

## 怎么选

第一次学习可以用 README 的 Ollama 默认组合：`nomic-embed-text` + `qwen3:4b`。它把成本和数据外传控制在本机，但需要下载模型并承受本机内存压力。内存不足时优先换更小的 chat model，或改用 API；不要只看参数量，还要确认 embedding 维度、语言覆盖和上下文窗口满足任务。

需要稳定的托管能力时，使用 `.env` 中的 OpenAI 变量，并查看 [OpenAI 模型目录](https://developers.openai.com/api/docs/models) 的当前能力、上下文窗口和价格。OpenAI SDK 使用 [Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)；兼容服务可能只支持其中一部分接口，先查该服务的文档。

Ollama 的模型安装和 API 行为以 [Ollama 官方文档](https://docs.ollama.com/quickstart) 与 [embedding API 文档](https://docs.ollama.com/api/embed) 为准。不要把网上旧文章里的模型名、显存要求或价格当作当前事实。

## 本项目的配置变量

`.env.example` 是可复制的模板：

- Ollama：`OLLAMA_BASE_URL`、`OLLAMA_EMBEDDING_MODEL`、`OLLAMA_CHAT_MODEL`；
- OpenAI-compatible：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_EMBEDDING_MODEL`、`OPENAI_CHAT_MODEL`。

`RAG_PROVIDER=ollama` 决定默认 provider；命令行的 `--provider openai` 可以覆盖它。API key 只放在未提交的 `.env` 或受保护的环境变量中，详见 README 的安全配置。
