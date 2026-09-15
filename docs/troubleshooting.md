# 故障排查

先确认你在项目根目录、虚拟环境已激活，并把完整错误信息中的 API key、个人路径和文档内容删掉后再分享。

## 命令和 Python

### `pdf-rag: command not found`（Windows 可能是“不是内部或外部命令”）

确认虚拟环境已激活，并重新安装：

```sh
python -m pip install -e .
```

也可以直接用模块检查当前环境：`python -m beginner_pdf_rag.cli ask --pdf data/dpr-paper.pdf --question "..."`。若该命令也找不到模块，确认当前目录是仓库根目录。

### Python 版本不支持或依赖安装失败

本项目要求 Python 3.11+：

```sh
python --version
```

macOS/Linux 若 `python` 不是 3.11+，使用 `python3 --version` 并用对应解释器重建 `.venv`；Windows 使用 `py -3.11 --version` 和 `py -3.11 -m venv .venv`。激活旧环境后再运行 `python -m pip install -e ".[test]"`。

## Ollama

### `Cannot connect to Ollama`

启动 Ollama 桌面应用，或在 Linux 按 [官方启动说明](https://docs.ollama.com/quickstart) 运行服务。确认 `.env` 的 `OLLAMA_BASE_URL` 是可访问地址（默认 `http://127.0.0.1:11434`），再试：

```sh
ollama list
```

如果 `ollama` 本身是 command not found，请从 [Ollama 下载页](https://ollama.com/download) 安装并重启终端。

### `Ollama model '...' is unavailable`

按错误信息中的模型名拉取，例如：

```sh
ollama pull nomic-embed-text
ollama pull qwen3:4b
ollama list
```

确认 `.env` 中的模型名与 `ollama list` 完全一致。embedding model 和 chat model 是两种角色，分别检查。

### 内存不足、模型退出或运行极慢

关闭占用内存的程序，先使用更小的 chat model，并确认本机满足该模型的实际要求。CPU 运行会比 GPU 慢；首次请求还包含加载模型的时间。若不能接受本机资源消耗，改用 OpenAI-compatible API，并阅读 [`docs/model-guide.md`](model-guide.md) 的隐私/成本权衡。

## OpenAI 或兼容 API

### `OPENAI_API_KEY is required` / API key 错误

仅对 `--provider openai` 设置 key。在项目根目录复制 `.env.example` 为 `.env`，填入真实 key，确认没有多余引号或不可见空格；不要把 key 放到 README、脚本或提交中。若 key 曾经泄露，立即撤销并重新生成。

### `401`

通常表示 key 无效、过期、没有权限，或 `OPENAI_BASE_URL` 指向了不接受该 key 的服务。检查服务商控制台、base URL 是否含 `/v1`，以及模型名是否属于该账户；不要在错误报告中粘贴 key。

### `429`

通常表示速率限制、余额/配额耗尽或并发过高。等待后重试，降低请求频率或检查账户配额；本示例没有自动重试，也不会替你购买额度。确认嵌入与答案请求都计入服务商的限制。

### 其他 API 连接或模型错误

确认网络、`OPENAI_BASE_URL`、`OPENAI_EMBEDDING_MODEL` 和 `OPENAI_CHAT_MODEL`。OpenAI-compatible 只表示接口形态相似，Responses API 或批量 embeddings 可能未被某个服务完整支持；查看该服务的官方文档，必要时先用默认 OpenAI 配置排除兼容性问题。

## PDF、OCR 和回答质量

### `Select an existing PDF file` 或扩展名错误

检查 `--pdf` 路径相对于当前目录是否正确；路径含空格请加引号。文件必须实际存在，并以 `.pdf`（不区分大小写）结尾。

### `Could not parse the PDF` 或答案为空

文件可能损坏、下载不完整或受密码保护。重新下载并用 PDF 阅读器打开；加密 PDF 请先在得到许可的前提下移除密码。若没有任何文字层，程序会报 `No extractable text was found`。

### 中文乱码、双栏顺序错、表格或公式不完整

`pypdf` 读取的是 PDF 的文字层，不保证复杂布局顺序。尝试从原始来源获得更好的 PDF；扫描件先运行 OCR，再人工抽查 OCR 文字、页码、表格和公式。OCR 错误会进入 embedding 和回答，模型不能可靠地自动修复它。

### 扫描 PDF / OCR 后仍提示没有文字

确认 OCR 输出的是带文字层的 PDF，而不是另一份纯图片；在 PDF 阅读器中尝试选择并复制一段文字。若仍不能复制，换 OCR 工具或导出设置，再运行命令。

### 运行很慢，尤其是 embedding 阶段

大 PDF 会产生更多 chunks；本程序每次运行都会重新嵌入整份文档。先用小 PDF 验证流程，确认 Ollama 模型已加载，或改用更快/托管的 embedding 服务。当前项目没有持久化缓存，这是已知局限。

### 检索到无关内容

问题可能太宽泛、PDF 文字顺序混乱，或 embedding model 不适合文档语言。用更具体的问题试试，暂时提高 `--top-k`（如 `--top-k 5`）并加 `--show-sources` 检查页码和分数；不要把低分片段自动当作证据。更换 embedding model 后必须重新索引。

### 模型说“无法回答”或答案不受支持

这是预期的保护性结果：Top-K 片段可能没有答案。先打开 Sources 中的页码，确认问题确实在 PDF 中；用原文中的术语重问。若片段有答案但模型仍无法回答，检查上下文窗口和模型语言能力，降低问题复杂度，并记住引用页需要人工核验。
