# 验证记录

执行日期：2026-09-16。此记录只保留可复现的命令、版本和结果；临时目录与任何密钥均未写入仓库。

## 干净安装

从公开 `main` 的干净浅克隆（提交 `509c95c`）在仓库外创建 Python 虚拟环境，随后在该克隆的根目录运行：

```sh
python -m pip install ".[test]"
pdf-rag --help
python -m pytest -q
```

- Python：3.12.1
- pip：26.2.1
- `pdf-rag --help`：退出码 0，并显示 `ask` 子命令。
- pytest：114 passed，1.62s（这是发布提交、尚未包含本次 smoke-script 契约测试时的计数）。

## 本地 Ollama

先执行 `ollama --version`、`ollama list` 和 `curl --fail --silent --show-error http://127.0.0.1:11434/api/tags`。Ollama 版本为 0.32.15，daemon 可用。已安装的文档默认模型是 embedding model `nomic-embed-text`（列表显示为 `nomic-embed-text:latest`）和 chat model `qwen3:4b`，因此没有执行 `ollama pull`。

在激活项目环境后，以下命令退出码为 0：

```sh
pdf-rag ask \
  --pdf data/dpr-paper.pdf \
  --question "What datasets are used to evaluate DPR?" \
  --provider ollama \
  --top-k 3 \
  --show-sources
```

结果索引了 61 个 chunks；答案非空并包含 `[Page 6]`。Sources 表的分数为 0.674、0.645、0.626，均为有限数值。生成答案本身没有记录，因为模型措辞不是稳定测试契约。

`scripts/smoke_ollama.sh` 对同一捆绑论文、问题和模型组合执行相同检查和查询，且从不拉取模型：缺失模型时会停止，并打印明确的 `ollama pull <model>` 手动操作提示。实际运行该脚本也以退出码 0 完成，返回非空、带 `[Page 6]` 的答案和相同的有限 Sources 分数。

## OpenAI-compatible 安全检查

使用空的 `OPENAI_API_KEY` 运行 OpenAI provider 的相同命令，程序按预期以 `OPENAI_API_KEY is required for the openai provider` 失败。该错误发生在 provider 创建之前。

没有执行真实 OpenAI API 请求：`not run: no authorized API request`。即使环境中存在 key，本次验证也没有授权付费请求。

## 新读者问题清单

独立的新读者审阅尚未在此任务中判定 PASS；以下是交接给独立审阅者的八个问题，以及仅根据仓库内容可找到的答案和位置：

1. 从新机器运行默认例子：README 的“15 分钟跑通：本地 Ollama”依次说明 clone、创建并激活环境、安装 `.[test]`、安装两个模型、复制 `.env.example`，再给出精确的 `pdf-rag ask` 命令。
2. embedding 与 chat model 的区别：[`docs/model-guide.md`](model-guide.md)“先记住两个模型”说明前者将文档/问题变成用于检索的向量，后者读取 Top-K 片段并组织答案。
3. 不提交 key 而切换 OpenAI：README“OpenAI-compatible API 路径 / 安全配置”要求把 `.env.example` 复制为受 `.gitignore` 保护的 `.env`，再设置 `OPENAI_API_KEY`。
4. 更换样例 PDF：README“换成自己的 PDF”给出替换 `--pdf` 路径的命令和带空格路径的引号规则。
5. 保留页码和 overlap 的原因：README“刚才发生了什么？”和 [`docs/concepts.md`](concepts.md)说明页码用于复核来源，overlap 减少句子跨 chunk 边界的信息损失。
6. 扫描 PDF 的处理：README“扫描件与 OCR”说明先运行 OCR，使用带文字层的副本，并检查 OCR 输出。
7. 查看支撑答案的 chunks：README 的 `--show-sources` 示例和 [`docs/troubleshooting.md`](troubleshooting.md)“检索到无关内容”说明检查 Sources 的页码和分数，并打开原 PDF 复核。
8. 项目刻意未实现的内容：README“局限与路线图”列出没有持久化索引、增量更新、复杂版面/表格/图片理解、访问控制、评测集和生产级监控等。

本次文档审计未发现需要修改的缺口；独立审阅者应在不读取本记录以外的对话上下文的条件下，逐题确认上述入口是否足够清晰。

## 最终本地验证

完成本次改动后，执行了以下检查：

```sh
python3 -m pytest -q tests/test_docs_contract.py
# 14 passed in 0.86s

bash -n scripts/smoke_ollama.sh
# exit code 0

python3 -m pytest -q
# 118 passed in 1.20s

python3 -m compileall -q src examples legacy
# exit code 0

python3 -m build
# exit code 0; produced beginner_pdf_rag-0.1.0.tar.gz and
# beginner_pdf_rag-0.1.0-py3-none-any.whl in dist/

git diff --check
# exit code 0

git status --short
#  M tests/test_docs_contract.py
# ?? docs/verification.md
# ?? scripts/
```

构建产物已在检查成功后从工作树移除；只保留上述预期的脚本、文档和契约测试改动，供提交。
