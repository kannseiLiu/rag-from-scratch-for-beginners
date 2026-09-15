# 零基础 PDF RAG 教程仓库设计

日期：2026-09-15

## 1. 目标

把现有单文件 PDF RAG 练习整理成一个公开、可复现、适合第一次接触 RAG 的中文 GitHub 教程。读者应能在约 15 分钟内完成默认的 Ollama 本地路线，并能理解索引、切块、Embedding、相似度检索、上下文构造和受约束回答之间的数据流。

仓库拟命名为 `rag-from-scratch-for-beginners`，公开发布在 GitHub 用户 `kannseiLiu` 下。

项目的学习起点是 Xuan-Son Nguyen 的 Hugging Face 文章 [Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/make-your-own-rag)。本项目会明确致谢和链接原文，并独立扩展 PDF 读取、页码引用、配置管理、OpenAI 兼容 API、测试和中文教程，不复制原文内容。

## 2. 目标读者与边界

目标读者只需要会打开终端、复制命令，并见过最基本的 Python。教程不能假设读者理解向量、余弦相似度、环境变量或 API Key。

首版专注于帮助读者理解并跑通 RAG，不引入 LangChain、LlamaIndex、Chroma、FAISS、数据库或复杂前端。结尾可以说明这些工具解决什么问题，但不把它们纳入主路径。

不承诺处理扫描版 PDF、复杂表格、数学公式或跨页结构。扫描件没有可提取文本时，程序应给出明确的 OCR 提示。

## 3. 教学路线

教程采用渐进式结构：

1. 用一个极小文本示例解释最小 RAG。
2. 用 `pypdf` 读取论文，并保留页码元数据。
3. 按字符切块并加入 overlap，直观解释其作用和局限。
4. 通过 Embedding 模型把文档块和问题转换为向量。
5. 用 NumPy 手写余弦相似度并选出 Top-K。
6. 将检索结果整理成带页码的上下文。
7. 要求生成模型只能依据上下文回答，资料不足时明确拒答。
8. 对比 Ollama 本地路线和 OpenAI 兼容 API 路线。
9. 解释质量、速度、隐私、成本和后续工程化方向。

每一阶段都包含：先讲目的，再展示最小代码，随后解释输入、输出和常见错误。完整程序保持模块化，但不以抽象框架掩盖核心算法。

## 4. 模型路线

### 4.1 Ollama 默认路线

默认路线完全在本机运行，避免 API 费用，也适合包含私密 PDF 的场景。教程提供经过实际验证的 Embedding 模型和轻量聊天模型安装命令，并说明模型名称可通过环境变量更换。

程序启动时要能区分以下问题：Ollama 未安装、Ollama 服务未启动、模型未下载、机器内存不足。

### 4.2 OpenAI 兼容 API 路线

云端路线通过官方 SDK 和环境变量使用 OpenAI API，同时允许用户配置兼容的 `base_url`。API Key 只从环境变量或本地 `.env` 读取，`.env` 必须被 Git 忽略，示例配置只能使用占位符。

Embedding 与回答模型分别配置。教程解释为什么两者是不同职责，以及切换 Embedding 模型后必须重新生成文档向量。

README 不写死可能变化的价格或“最佳模型”结论，而是给出选择原则并链接官方模型文档。

## 5. 示例论文与许可

仓库包含 ACL Anthology 的论文：

- Vladimir Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020。
- 原文页：https://aclanthology.org/2020.emnlp-main.550/
- PDF：https://aclanthology.org/2020.emnlp-main.550.pdf
- 许可：Creative Commons Attribution 4.0 International（CC BY 4.0）。

PDF 放在 `data/` 的明确命名路径下，同时提供来源说明和归属信息。README 使用这篇论文设计可核查的问题，例如论文使用了哪些数据集、DPR 与 BM25 的比较结果是什么，并提醒模型答案必须带页码依据。

## 6. 仓库结构

```text
README.md                         主教程、快速开始与完整原理
LICENSE                           项目代码许可证
.env.example                     Ollama/API 配置模板
.gitignore                       密钥、缓存、虚拟环境和用户 PDF
pyproject.toml                   Python 版本、依赖和命令入口
data/
  README.md                      示例论文来源、许可和替换方式
  dpr-paper.pdf                  可直接运行的开放许可示例 PDF
examples/
  01_minimal_text_rag.py         最小文本 RAG
  02_pdf_rag_ollama.py           本地 PDF RAG
  03_pdf_rag_openai.py           OpenAI 兼容 API 版本
src/beginner_pdf_rag/
  config.py                      安全读取配置
  pdf_loader.py                  PDF 文本和页码
  chunking.py                    切块
  embeddings.py                 Ollama/OpenAI Embedding 适配
  retrieval.py                  余弦相似度与 Top-K
  generation.py                 上下文和回答
  cli.py                        统一命令行入口
tests/                           无需联网的确定性测试
docs/
  concepts.md                   面向零基础的概念详解
  troubleshooting.md            安装、模型、PDF、API 排错
  model-guide.md                Ollama 与 API 选择指南
```

原有 `rag.py` 作为学习草稿保留在 Git 历史或迁移到 `legacy/`，README 明确指出正式入口，避免读者运行写死本机路径的代码。

## 7. CLI 与数据流

统一命令示例：

```bash
pdf-rag ask --pdf data/dpr-paper.pdf --question "What datasets are used in this paper?"
```

配置决定使用 Ollama 或 OpenAI，但索引与检索逻辑保持一致：

```text
PDF -> 页面文本 -> 带重叠的文本块 -> 文档向量
问题 -> 问题向量 -> 余弦相似度 -> Top-K 文本块
问题 + 带页码上下文 -> 聊天模型 -> 带出处回答
```

首版向量只保存在内存中，每次运行重新计算。这样代码最容易理解。README 会说明当文档变多或启动变慢时，应升级为持久化向量库。

## 8. 错误处理与安全

程序需要给出面向用户的错误信息，而不是直接暴露长堆栈：

- PDF 路径不存在或不是 PDF；
- PDF 加密、损坏或没有可提取文本；
- chunk size、overlap 或 top-k 不合法；
- Ollama 不可用或模型缺失；
- API Key 缺失；
- Embedding 返回空值或维度不一致；
- 检索不到足够相关内容。

任何日志、示例输出和测试都不能包含真实 API Key 或用户私人 PDF 内容。仓库默认忽略 `*.env`、虚拟环境、缓存和 `data/private/`。

## 9. 测试与读者验证

自动化测试不依赖网络或真实模型，通过假 Embedding 和假生成器验证：

- PDF 页码得到保留；
- 切块 overlap 正确且参数校验明确；
- 余弦相似度排序和 Top-K 正确；
- 上下文包含页码；
- 无依据时触发拒答提示；
- 配置不会泄露或误提交密钥；
- CLI 参数和错误提示可理解。

实际冒烟测试分别覆盖 Ollama 和 OpenAI 路线；如果本机未配置 OpenAI Key，则只验证配置错误路径，不发送付费请求。

完成前进行一次“新读者测试”：让没有对话背景的审阅者只阅读仓库，回答如何安装、如何运行默认示例、如何切换 API、如何替换 PDF、如何解释页码引用及遇到扫描件怎么办。关键问题必须能从文档中直接得到答案。

## 10. GitHub 发布质量

README 首屏要在较短篇幅内说明项目解决什么问题、展示一次运行命令和预期输出，并提供清晰目录。仓库增加合适的 description、topics 和许可证，不使用虚假性能数据或夸大宣传。

发布前必须确认：从全新虚拟环境安装成功；测试通过；示例 PDF 来源可核验；所有链接有效；仓库历史没有密钥和本机绝对路径；GitHub 公共仓库创建和推送成功。

## 11. 完成标准

- 零基础读者可以只按 README 跑通 Ollama PDF 问答。
- 用户可通过 `.env` 切换到 OpenAI 兼容 API。
- 默认回答展示引用页码和检索分数。
- 示例 PDF 与 CC BY 4.0 归属信息完整。
- 自动化测试和至少一条本地真实模型冒烟路径通过。
- README、概念文档、模型选择和排错文档不存在互相矛盾的命令。
- GitHub 仓库公开可访问，description、topics 和主分支设置正确。
