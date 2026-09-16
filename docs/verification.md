# 验证记录

执行日期：2026-09-16。此记录只保留可复现的命令、版本和结果；临时目录与任何密钥均未写入仓库。

## 本次修订中新执行的检查

以下结果是在本次修订工作树中重新执行的；临时目录与任何密钥均未写入仓库。

### 安装命令（可复现）

在仓库外创建临时虚拟环境后，从仓库根目录运行：

```sh
python -m pip install ".[test]"
pdf-rag --help
python -m pytest -q
```

新环境应使用 Python 3.11 或更高版本；`.[test]` 会安装测试所需的 PyYAML、构建工具 `build` 和直接运行时依赖，包括 `httpx`。

### 真实服务状态

本次修订在最终代码上重新运行了真实本地 Ollama 路径，没有发送 OpenAI 请求。复现命令为：

```sh
ollama --version
ollama list
curl --fail --silent --show-error http://127.0.0.1:11434/api/tags
ollama show nomic-embed-text
ollama show qwen3:4b
bash scripts/smoke_ollama.sh
```

脚本固定使用 `http://127.0.0.1:11434`、`nomic-embed-text` 和 `qwen3:4b`。预检查只确认模型已经存在，不会自动执行 `ollama pull`；随后会运行完整的 `pdf-rag ask` 冒烟流程并验证答案、页码引用和 Sources 分数。

本次结果：Ollama 0.32.15，daemon 与两个固定模型均可用；索引 61 个 chunks，答案非空并包含 `[Page 6]`，Sources 分数为 0.674、0.645、0.626，脚本退出码为 0。没有执行 `ollama pull`。答案正文未写入仓库，因为模型措辞不是稳定测试契约。

### OpenAI-compatible 缺少密钥路径

使用 `OPENAI_API_KEY=` 可以在 provider 创建前验证缺少密钥路径；本次没有发送真实 OpenAI API 请求。

### 冒烟脚本契约

契约测试在隔离临时仓库中使用本地 fake `ollama`、`curl` 和 `pdf-rag`，无网络请求。脚本会捕获并原样输出 CLI 的合并输出；CLI 非零时保留该输出并返回原状态。成功时还要求答案非空、答案中含 `[Page N]` 引用、Sources 表至少有一个有限数值分数。测试同时验证固定 daemon URL/模型名不会被 caller 环境或 `.env` 覆盖，且不会调用 `ollama pull`。

## 此前直接 CLI 证据

在完善自动校验脚本前，也曾直接执行同一个 `pdf-rag ask` 命令：

```sh
pdf-rag ask \
  --pdf data/dpr-paper.pdf \
  --question "What datasets are used to evaluate DPR?" \
  --provider ollama \
  --top-k 3 \
  --show-sources
```

该次运行同样索引了 61 个 chunks，答案包含 `[Page 6]`，Sources 分数为 0.674、0.645、0.626。这组历史结果只用于和本次最终 smoke 结果交叉核验。

## 独立新读者审阅

结果：PASS，8/8。

独立审阅者只查看了教学文件 `README.md`、`docs/concepts.md`、`docs/model-guide.md`、`docs/troubleshooting.md` 和 `data/README.md`；明确排除 `docs/verification.md`，也没有读取对话。八题均能仅凭这些教学文件回答：默认运行入口、embedding/chat model 分工、无泄漏切换 OpenAI、替换 PDF、页码与 overlap 的原因、扫描件 OCR、`--show-sources` 复核方式，以及项目明确不实现的生产能力。

## 最终本地验证

完成本次修订后，执行了以下检查：

```sh
python3 -m pytest -q tests/test_docs_contract.py
# 20 passed

python3 -m pytest -q tests/test_docs_contract.py tests/test_github_metadata.py tests/test_providers.py
# 66 passed

bash -n scripts/smoke_ollama.sh
# exit code 0

bash scripts/smoke_ollama.sh
# exit code 0; 61 chunks; cited answer; three finite scores

python3 -m pytest -q
# 138 passed

python3 -m compileall -q src examples legacy
# exit code 0

python3 -m build
# exit code 0; produced beginner_pdf_rag-0.1.0.tar.gz and
# beginner_pdf_rag-0.1.0-py3-none-any.whl in dist/

git diff --check
# exit code 0

git status --short
# no output after commit
```

`python3 -m build` 在检查成功后产生的 `dist/` 仅包含 sdist 与 wheel，随后移出工作树；只保留上述预期的脚本、文档和契约测试改动，供提交。
