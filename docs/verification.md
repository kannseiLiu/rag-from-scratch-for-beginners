# 验证记录

执行日期：2026-09-16。此记录只保留可复现的命令、版本和结果；临时目录与任何密钥均未写入仓库。

## 本次修订中新执行的检查

以下命令是在本次修订中重新执行的；临时虚拟环境创建在仓库外，具体临时路径未写入记录。

### 干净安装

在仓库外创建临时虚拟环境后，从仓库根目录运行：

```sh
python -m pip install ".[test]"
pdf-rag --help
python -m pytest -q
```

- Python：3.12.1
- pip：23.2.1
- `pdf-rag --help`：退出码 0，并显示 `ask` 子命令。
- pytest：119 passed，1.53s。

### Ollama 前置条件

执行了 `ollama --version`、`ollama list`、`curl --fail --silent --show-error http://127.0.0.1:11434/api/tags`，以及对两个固定模型名的 `ollama show` 检查：

- Ollama：0.32.15
- daemon：可用（curl 退出码 0）
- embedding model：`nomic-embed-text:latest`
- chat model：`qwen3:4b`
- 没有执行 `ollama pull`。

### OpenAI-compatible 缺少密钥路径

使用 `OPENAI_API_KEY=` 运行相同的 OpenAI provider 命令，退出码为 1，并显示 `OPENAI_API_KEY is required for the openai provider`；错误发生在 provider 创建之前。没有执行真实 OpenAI API 请求：`not run: no authorized API request`。

### 冒烟脚本契约

契约测试在隔离临时仓库中使用本地 fake `ollama`、`curl` 和 `pdf-rag`，无网络请求；成功路径验证脚本最终到达 CLI，并强制传递文档规定的 daemon URL 和模型名，caller 环境与 `.env` 均无法覆盖，且不会调用 `ollama pull`。

## 保留的此前真实 Ollama CLI 证据

以下结果来自此前直接执行的 `pdf-rag ask` 命令，而不是本次修订重新执行的 `scripts/smoke_ollama.sh`：

```sh
pdf-rag ask \
  --pdf data/dpr-paper.pdf \
  --question "What datasets are used to evaluate DPR?" \
  --provider ollama \
  --top-k 3 \
  --show-sources
```

此前运行索引了 61 个 chunks；答案非空并包含 `[Page 6]`。Sources 表分数为 0.674、0.645、0.626，均为有限数值。答案正文未记录，因为模型措辞不是稳定测试契约。本次没有把这组结果冒充为 smoke script 的真实运行证据。

## 独立新读者审阅

结果：PASS，8/8。

独立审阅者只查看了教学文件 `README.md`、`docs/concepts.md`、`docs/model-guide.md`、`docs/troubleshooting.md` 和 `data/README.md`；明确排除 `docs/verification.md`，也没有读取对话。八题均能仅凭这些教学文件回答：默认运行入口、embedding/chat model 分工、无泄漏切换 OpenAI、替换 PDF、页码与 overlap 的原因、扫描件 OCR、`--show-sources` 复核方式，以及项目明确不实现的生产能力。

## 最终本地验证

完成本次修订后，执行了以下检查：

```sh
python3 -m pytest -q tests/test_docs_contract.py
# 15 passed in 0.35s

bash -n scripts/smoke_ollama.sh
# exit code 0

python3 -m pytest -q
# 119 passed in 1.15s

python3 -m compileall -q src examples legacy
# exit code 0

python3 -m build
# exit code 0; produced beginner_pdf_rag-0.1.0.tar.gz and
# beginner_pdf_rag-0.1.0-py3-none-any.whl in dist/

git diff --check
# exit code 0

git status --short
#  M docs/verification.md
#  M scripts/smoke_ollama.sh
#  M tests/test_docs_contract.py
```

`python3 -m build` 在检查成功后产生的 `dist/` 仅包含 sdist 与 wheel，随后移出工作树；只保留上述预期的脚本、文档和契约测试改动，供提交。
