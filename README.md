# Beginner PDF RAG

把一份 PDF 变成可追溯的问答对象：程序先找到相关页，再让语言模型只根据这些页回答，并显示页码来源。它是一个适合学习的最小 Retrieval-Augmented Generation（RAG，检索增强生成）示例，不保证消除幻觉，也不是生产系统。

你将看到：

- 使用 Python 3.11+ 读取 PDF、按页切分文本并保留页码；
- 使用 Ollama 在本机运行 embedding model（嵌入模型）和 chat model（对话模型），也可切换到 OpenAI-compatible API（兼容 OpenAI 接口的服务）；
- 用 cosine similarity（余弦相似度）检索 Top-K（前 K 个）片段；
- 让答案附带 `[Page N]` 页面引用，并在需要时打印检索分数。

## 目录

- [RAG 流程](#architecture)
- [15 分钟跑通](#quickstart)
- [OpenAI-compatible API](#openai)
- [替换 PDF](#replace-pdf)
- [测试](#tests)
- [局限与路线图](#roadmap)
- [归属与许可证](#attribution)
- [故障排查](#troubleshooting)

<a id="architecture"></a>
## RAG 的一条完整路径

```mermaid
flowchart LR
    A[PDF] --> B[按页提取文本]
    B --> C[1200 字符 chunk<br/>overlap 200]
    C --> D[embedding model<br/>文本 → 向量]
    D --> E[余弦相似度索引]
    Q[问题] --> F[问题向量]
    F --> E
    E --> G[Top-K 相关片段]
    G --> H[带页码 context]
    H --> I[chat model]
    I --> J[答案 + [Page N]]
```

PDF 和问题都要经过同一个嵌入模型，才能在同一向量空间中比较；对话模型只接收检索出来的上下文，而不是整本 PDF。术语的直观解释见 [`docs/concepts.md`](docs/concepts.md)。

<a id="quickstart"></a>
## 15 分钟跑通：本地 Ollama

这一条路径不会把 PDF 内容或问题发送到云端。第一次下载模型需要磁盘空间和网络；生成答案时模型在 Ollama 中运行。

### 1. 准备 Python 环境

要求 Python 3.11 或更高版本。先打开终端（Windows 请打开 PowerShell），进入一个你想保存代码的目录：

macOS/Linux：

```sh
git clone https://github.com/kannseiLiu/rag-from-scratch-for-beginners.git pdf-rag
cd pdf-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
# 等价写法：pip install -e .[test]
```

Windows PowerShell：

```powershell
git clone https://github.com/kannseiLiu/rag-from-scratch-for-beginners.git pdf-rag
Set-Location pdf-rag
py -3.11 -m venv .venv
# 若 `python` 已指向 Python 3.11+，也可用：python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
# 等价写法：pip install -e .[test]
```

如果代码已经在本地，直接 `cd` 到项目目录即可。安装 `.[test]` 会同时安装运行依赖和 pytest 测试工具；只想运行程序时，`python -m pip install -e .` 也可以。

### 2. 安装并启动 Ollama

从 [Ollama 官方下载页](https://ollama.com/download) 安装。macOS/Windows 安装桌面应用后启动它；Linux 可按官方页面的安装命令操作。然后在另一个终端窗口拉取两个模型：

```sh
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

`nomic-embed-text` 只负责把文本变成向量，`qwen3:4b` 负责写答案，二者不能互换。Ollama 默认监听 `http://127.0.0.1:11434`，与 [Ollama API 文档](https://docs.ollama.com/api/introduction) 一致。

### 3. 配置并提问

复制项目提供的配置模板（不要把真实密钥提交到 Git）：

macOS/Linux：

```sh
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

默认配置已经指向本地 Ollama：`RAG_PROVIDER=ollama`、`OLLAMA_BASE_URL=http://127.0.0.1:11434`、`OLLAMA_EMBEDDING_MODEL=nomic-embed-text`、`OLLAMA_CHAT_MODEL=qwen3:4b`。这些值与 `.env.example` 和程序的 Settings 默认值一致。运行一个有页码来源的精确命令：

```sh
pdf-rag ask --pdf data/dpr-paper.pdf --question "What datasets are used to evaluate DPR?" --show-sources
```

第一次运行会先为 61 个片段计算嵌入，可能需要几十秒到几分钟。终端输出的答案由模型生成，因此措辞会变化；成功输出的形状如下（页码分数也可能变化）：

```text
Indexed 61 chunks.
DPR is evaluated on ... [Page 2]

Sources
Page | Score
---- | -----
2 | 0.842
3 | 0.811
5 | 0.776
```

若答案不在文档上下文中，程序提示模型返回：`I cannot find this information in the document.`。`Indexed 61 chunks.` 是本仓库这份示例 PDF 在默认 1200/200 参数下的结果；更换模型不影响分块数，更换 PDF 或提取出的文本可能改变分块数。

### 刚才发生了什么？

`pdf-rag ask` 每次运行都会重新索引一次 PDF：`load_pdf` 按页提取并规范化文本，`chunk_pages` 将每页切成 1200 字符片段，相邻片段重叠 200 字符。嵌入模型将片段变成向量；程序也将问题变成向量，使用余弦相似度排序并取默认 `top-k=3` 的片段。最后，生成器收到问题、片段文本和页码标记，系统提示要求它只使用文档上下文并在支持答案时写 `[Page N]`。

这里的“引用”是检索片段的页码，不是学术引用管理器，也不是对回答真实性的证明。打开 PDF 对照引用页，是最可靠的复核方式。

## 从例子逐行学习

推荐按以下顺序阅读：

1. [`examples/01_minimal_text_rag.py`](examples/01_minimal_text_rag.py) 只有三段文字，展示 embedding、余弦相似度、Top-K 和 prompt（发给模型的指令/输入）。它使用代码中写死的 `nomic-embed-text` 与 `qwen3:4b`，先确保 Ollama 已启动。
2. [`examples/02_pdf_rag_ollama.py`](examples/02_pdf_rag_ollama.py) 将同样的想法接到 PDF；默认文件是 `data/dpr-paper.pdf`，也可用 `--pdf` 和 `--question` 覆盖。
3. [`examples/03_pdf_rag_openai.py`](examples/03_pdf_rag_openai.py) 保持 PDF 流程不变，只把 provider（模型服务提供方）切换为 OpenAI-compatible 配置。

核心模块在 `src/beginner_pdf_rag/`：`pdf_loader.py` 负责页面，`chunking.py` 负责片段，`retrieval.py` 负责相似度，`providers.py` 负责服务适配，`pipeline.py` 串起索引和提问，`cli.py` 负责命令行。你不需要先掌握这些文件，先运行命令，再逐个对照即可。

<a id="openai"></a>
## OpenAI-compatible API 路径

这条路径适合没有本地硬件、需要托管模型，或已经有兼容 OpenAI SDK 的服务。默认配置使用 `OPENAI_BASE_URL=https://api.openai.com/v1`、嵌入模型 `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` 和对话模型 `OPENAI_CHAT_MODEL=gpt-4.1-mini`。模型名称和价格可能更新，请查阅 [OpenAI 模型目录](https://developers.openai.com/api/docs/models)；API 调用形态见 [Responses API 文档](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)。

### 安全配置

1. 复制 `.env.example` 为 `.env`，只编辑 `.env`，不要把密钥写进 Python 文件、命令历史、截图或 README。
2. 把 `OPENAI_API_KEY` 改成服务商签发的密钥；`.env.example` 中的 `replace-me` 只是占位符，不是可用密钥。
3. 确认 `.gitignore` 忽略 `.env`，提交前运行 `git diff -- .env` 和 `git status`。若密钥曾经泄露，立即在服务商控制台撤销并重新生成。
4. 对 OpenAI 使用真实 API key；对其他兼容服务，将 `OPENAI_BASE_URL` 改为其官方 base URL，并只使用该服务支持的 embedding/chat 模型。兼容“接口格式”不代表模型、隐私政策或计费相同。

macOS/Linux：

```sh
cp .env.example .env
# 用编辑器打开 .env，设置 OPENAI_API_KEY=你的密钥
pdf-rag ask --provider openai --pdf data/dpr-paper.pdf --question "What datasets are used to evaluate DPR?" --show-sources
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
# 用记事本打开 .env，设置 OPENAI_API_KEY=你的密钥
pdf-rag ask --provider openai --pdf data/dpr-paper.pdf --question "What datasets are used to evaluate DPR?" --show-sources
```

程序会读取当前工作目录的 `.env`。如果只想临时设置而不保存到文件，也可以在当前 shell 设置环境变量，但不要把值贴进聊天或提交记录。OpenAI 路径会把 PDF 片段和问题发送给你配置的服务；发送前先确认文档包含的信息可以外传。

<a id="replace-pdf"></a>
## 换成自己的 PDF

`data/dpr-paper.pdf` 是示例，不是输入限制。把自己的文件放在一个明确的路径，然后替换 `--pdf` 的值：

```sh
pdf-rag ask --pdf "data/my-report.pdf" --question "这个报告的主要结论是什么？" --show-sources
```

路径含空格时用引号；Windows PowerShell 同样可用双引号。问题尽量具体，并要求文档中有答案，例如“第几页定义了 X？”或“报告列出的三个风险是什么？”。程序每次运行都会重新读取文件；文件不存在、扩展名不是 `.pdf`、加密或无法解析时会给出错误。

### 扫描件与 OCR

本项目使用 `pypdf` 提取 PDF 内已有的文字层，不识别图片里的字。扫描件通常只有页面图片，运行时会报告 `No extractable text was found. Run OCR on this PDF and try again.`。请先使用你信任的 OCR（Optical Character Recognition，光学字符识别）工具生成带文字层的副本，再把副本传给 `--pdf`；检查 OCR 结果，尤其是表格、公式、双栏排版和中文。不要因为程序显示了页码就认为 OCR 内容正确。

<a id="tests"></a>
## 测试

激活虚拟环境后运行：

```sh
python -m pytest -q
```

测试使用本地 fixture 和假的 provider，不需要下载模型或调用云端。若要只检查文档契约：

```sh
python3 -m pytest tests/test_docs_contract.py -q
```

Windows PowerShell（激活 `.venv` 后）：

```powershell
python -m pytest -q
```

<a id="roadmap"></a>
## 局限与路线图

当前实现是教学项目：只处理一个 PDF；每次提问都重新嵌入整份文档；没有持久化索引、增量更新、复杂版面解析、表格/图片理解、访问控制、评测集或生产级监控；Top-K 和分块参数是代码级默认值；生成模型可能误读上下文，引用页也不能替代人工核验。不要将它当作医疗、法律、财务或安全决策的唯一依据。

可能的后续路线图是持久化向量索引与缓存、可配置分块/Top-K、表格和 OCR 管线、检索与答案的自动评测、更多 provider，以及面向多文档的元数据过滤。路线图不是已实现功能。

<a id="attribution"></a>
## 贡献、归属与许可证

欢迎提交能复现的问题、测试和小而清晰的改动。请先运行测试，并在问题报告中说明操作系统、Python 版本、provider、模型名和完整错误信息（删掉 API key）。

仓库代码与本教程文档由本仓库作者提供，采用 [MIT License](LICENSE)。但 `data/dpr-paper.pdf` 是从 [ACL Anthology](https://aclanthology.org/2020.emnlp-main.550/) 下载的论文副本，不因仓库的 MIT 许可证而改变；它及其原始内容按 ACL 页面说明受 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。论文作者和 ACL 不为本教程背书。论文的完整书目信息、下载地址和 SHA-256 校验值见 [`data/README.md`](data/README.md)。使用论文时请保留作者、论文标题、ACL/EMNLP 出版信息及 CC BY 4.0 归属。

本教程的学习起点参考 Xuan-Son Nguyen 的 Hugging Face 文章 [Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/make-your-own-rag)。本仓库独立扩展了 PDF 读取、页码引用、配置、OpenAI-compatible API、测试和中文讲解，没有复制该文章内容。

<a id="troubleshooting"></a>
## 故障排查

遇到连接、模型、密钥、PDF、OCR 或答案质量问题，请先看 [`docs/troubleshooting.md`](docs/troubleshooting.md)（故障排查）；想理解术语看 [`docs/concepts.md`](docs/concepts.md)，想选模型看 [`docs/model-guide.md`](docs/model-guide.md)。
