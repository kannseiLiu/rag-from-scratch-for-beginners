# Beginner PDF RAG Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a Chinese, zero-background PDF RAG tutorial that runs with Ollama by default and can switch to an OpenAI-compatible API.

**Architecture:** Keep the learning path framework-free: extract page-aware PDF text, create overlapping chunks, embed them, rank them with handwritten cosine similarity, and pass cited context to a generator. Model-specific behavior sits behind two small protocols so the same pipeline works with Ollama or OpenAI without hiding the RAG algorithm.

**Tech Stack:** Python 3.11+, pypdf, NumPy, Ollama Python SDK, OpenAI Python SDK, python-dotenv, pytest, GitHub CLI.

## Global Constraints

- The default 15-minute path uses Ollama and requires no paid API.
- Do not add LangChain, LlamaIndex, a vector database, a web framework, or a frontend in version 1.
- Preserve PDF page metadata through retrieval and show page citations in output.
- API keys come only from environment variables or an ignored `.env`; no real key or user-private PDF enters Git history.
- The bundled paper is Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, ACL Anthology ID `2020.emnlp-main.550`, distributed under CC BY 4.0 with attribution.
- Network/model integration tests must be opt-in; the default test suite is deterministic and offline.
- User-facing errors must explain the next action without exposing secrets or raw tracebacks.
- The final public repository is `kannseiLiu/rag-from-scratch-for-beginners`.

---

## File Map

- `pyproject.toml`: package metadata, Python floor, dependencies, CLI, pytest settings.
- `.env.example`: safe Ollama and OpenAI configuration placeholders.
- `.gitignore`: local environments, secrets, caches, private PDFs, generated index files.
- `src/beginner_pdf_rag/models.py`: immutable `Page`, `Chunk`, and `SearchResult` data objects.
- `src/beginner_pdf_rag/config.py`: validated runtime configuration.
- `src/beginner_pdf_rag/pdf_loader.py`: PDF extraction with page numbers and actionable errors.
- `src/beginner_pdf_rag/chunking.py`: page-aware overlapping character chunks.
- `src/beginner_pdf_rag/retrieval.py`: cosine similarity and deterministic Top-K ranking.
- `src/beginner_pdf_rag/providers.py`: Ollama and OpenAI embedding/generation adapters.
- `src/beginner_pdf_rag/pipeline.py`: provider-independent indexing, retrieval, context, answer flow.
- `src/beginner_pdf_rag/cli.py`: beginner-facing `pdf-rag ask` command.
- `examples/01_minimal_text_rag.py`: one-file learning example with no PDF.
- `examples/02_pdf_rag_ollama.py`: smallest local PDF invocation.
- `examples/03_pdf_rag_openai.py`: smallest API invocation.
- `data/dpr-paper.pdf`: reproducible CC BY 4.0 example.
- `data/README.md`: paper citation, source, license, checksum, replacement instructions.
- `README.md`: complete Chinese tutorial and first-run path.
- `docs/concepts.md`: detailed RAG concepts and formulas.
- `docs/model-guide.md`: Ollama/OpenAI decision guide and API setup.
- `docs/troubleshooting.md`: symptom → cause → command → solution guide.
- `tests/`: deterministic unit/contract tests.

---

### Task 1: Package, Data Models, and Safe Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/beginner_pdf_rag/__init__.py`
- Create: `src/beginner_pdf_rag/models.py`
- Create: `src/beginner_pdf_rag/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `Page(number: int, text: str)`, `Chunk(page: int, index: int, text: str)`, `SearchResult(chunk: Chunk, score: float)`.
- Produces: `Settings.from_env(provider: str) -> Settings` with `provider`, `ollama_base_url`, `embedding_model`, `chat_model`, `openai_api_key`, and `openai_base_url`.

- [ ] **Step 1: Write failing configuration tests**

```python
from beginner_pdf_rag.config import Settings, SettingsError


def test_ollama_defaults_do_not_require_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env("ollama")
    assert settings.provider == "ollama"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chat_model == "qwen3:4b"


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SettingsError, match="OPENAI_API_KEY"):
        Settings.from_env("openai")
```

- [ ] **Step 2: Run RED**

Run: `python3 -m pytest tests/test_config.py -q`

Expected: collection fails because `beginner_pdf_rag` does not exist.

- [ ] **Step 3: Add package metadata and safe ignore rules**

Use this dependency/entry-point shape in `pyproject.toml`:

```toml
[project]
name = "beginner-pdf-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26,<3",
  "ollama>=0.4,<1",
  "openai>=1.50,<3",
  "pypdf>=5,<7",
  "python-dotenv>=1,<2",
]

[project.optional-dependencies]
test = ["pytest>=8,<10"]

[project.scripts]
pdf-rag = "beginner_pdf_rag.cli:main"

[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Ignore `.venv/`, `.env`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `data/private/`, and `data/index/`. Do not ignore `data/dpr-paper.pdf`.

- [ ] **Step 4: Implement the immutable models and validated settings**

Use frozen dataclasses. `Settings.from_env()` must call `load_dotenv()`, accept only `ollama` or `openai`, apply the documented Ollama defaults, and require a nonblank `OPENAI_API_KEY` for `openai`. The error text must say how to copy `.env.example`.

`.env.example` must contain only placeholders:

```dotenv
RAG_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen3:4b

# Only needed for --provider openai
OPENAI_API_KEY=replace-me
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

- [ ] **Step 5: Run GREEN and secret scan**

Run:

```bash
python3 -m pytest tests/test_config.py -q
git check-ignore .env data/private/resume.pdf data/dpr-paper.pdf
rg -n 'sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]+' . --glob '!.git/**'
```

Expected: tests pass; the first two private paths are ignored, `data/dpr-paper.pdf` is not ignored; secret scan has no matches.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src tests/test_config.py
git commit -m "chore: scaffold beginner PDF RAG package"
```

---

### Task 2: Page-Aware PDF Loading and Chunking

**Files:**
- Create: `src/beginner_pdf_rag/pdf_loader.py`
- Create: `src/beginner_pdf_rag/chunking.py`
- Create: `tests/fixtures/two-pages.pdf`
- Create: `tests/test_pdf_loader.py`
- Create: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `Page`, `Chunk` from Task 1.
- Produces: `load_pdf(path: str | Path) -> list[Page]`.
- Produces: `chunk_pages(pages: Sequence[Page], chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]`.

- [ ] **Step 1: Write PDF loader tests**

```python
def test_load_pdf_preserves_one_based_page_numbers():
    pages = load_pdf(FIXTURES / "two-pages.pdf")
    assert [(page.number, page.text) for page in pages] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_image_only_pdf_has_an_ocr_instruction(tmp_path):
    path = make_blank_pdf(tmp_path / "blank.pdf")
    with pytest.raises(PdfLoadError, match="OCR"):
        load_pdf(path)
```

Generate the tiny fixture with `pypdf.PdfWriter` plus a deterministic minimal content stream in a checked-in test helper; do not depend on the tutorial paper for unit tests.

- [ ] **Step 2: Write chunking tests**

```python
def test_chunks_keep_page_and_overlap():
    chunks = chunk_pages([Page(7, "abcdefghij")], chunk_size=6, overlap=2)
    assert [(item.page, item.index, item.text) for item in chunks] == [
        (7, 0, "abcdef"),
        (7, 1, "efghij"),
        (7, 2, "ij"),
    ]


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_chunk_parameters_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], chunk_size, overlap)
```

- [ ] **Step 3: Run RED**

Run: `python3 -m pytest tests/test_pdf_loader.py tests/test_chunking.py -q`

Expected: imports fail because the loader and chunker do not exist.

- [ ] **Step 4: Implement the minimum loader and chunker**

`load_pdf()` must validate existence, `.pdf` suffix, readable/encrypted state, and nonempty extracted text. Normalize repeated whitespace without joining different pages. Wrap pypdf parsing errors in `PdfLoadError` with a short next action.

`chunk_pages()` must validate `chunk_size > 0` and `0 <= overlap < chunk_size`, skip blank pages, advance by `chunk_size - overlap`, and number chunks separately per page beginning at zero.

- [ ] **Step 5: Run GREEN**

Run: `python3 -m pytest tests/test_pdf_loader.py tests/test_chunking.py -q`

Expected: all loader and chunking tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/beginner_pdf_rag/pdf_loader.py src/beginner_pdf_rag/chunking.py tests
git commit -m "feat: load and chunk PDFs with page metadata"
```

---

### Task 3: Handwritten Vector Retrieval

**Files:**
- Create: `src/beginner_pdf_rag/retrieval.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float`.
- Produces: `rank_chunks(query_embedding: Sequence[float], chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], top_k: int) -> list[SearchResult]`.

- [ ] **Step 1: Write failing deterministic ranking tests**

```python
def test_cosine_similarity_and_top_k_ranking():
    chunks = [Chunk(1, 0, "cats"), Chunk(2, 0, "retrieval"), Chunk(3, 0, "cooking")]
    results = rank_chunks(
        [1.0, 0.0],
        chunks,
        [[0.8, 0.2], [1.0, 0.0], [0.0, 1.0]],
        top_k=2,
    )
    assert [item.chunk.text for item in results] == ["retrieval", "cats"]
    assert results[0].score == pytest.approx(1.0)


def test_zero_vectors_and_dimension_mismatch_are_actionable():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1.0], [1.0, 2.0])
```

- [ ] **Step 2: Run RED**

Run: `python3 -m pytest tests/test_retrieval.py -q`

Expected: import failure for `beginner_pdf_rag.retrieval`.

- [ ] **Step 3: Implement retrieval with NumPy**

Convert inputs to one-dimensional float arrays, reject empty/zero/dimension-mismatched vectors, compute `dot(left, right) / (norm(left) * norm(right))`, validate one embedding per chunk and positive `top_k`, and sort descending by score with original chunk order as the tie breaker.

- [ ] **Step 4: Run GREEN**

Run: `python3 -m pytest tests/test_retrieval.py -q`

Expected: all retrieval tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/beginner_pdf_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add transparent cosine retrieval"
```

---

### Task 4: Ollama and OpenAI Providers

**Files:**
- Create: `src/beginner_pdf_rag/providers.py`
- Create: `tests/test_providers.py`

**Interfaces:**
- Produces protocol `Embedder.embed(texts: Sequence[str]) -> list[list[float]]`.
- Produces protocol `Generator.generate(question: str, context: str) -> str`.
- Produces `OllamaEmbedder`, `OllamaGenerator`, `OpenAIEmbedder`, `OpenAIGenerator`.
- Produces `build_providers(settings: Settings) -> tuple[Embedder, Generator]`.

- [ ] **Step 1: Confirm current official SDK calls**

Check the installed Ollama Python package API and current official OpenAI Python documentation before coding. Use only official SDK documentation. Pin the tested version ranges in `pyproject.toml`; do not put model prices in code or README.

- [ ] **Step 2: Write provider contract tests with injected fake clients**

```python
def test_ollama_embedder_preserves_batch_order():
    client = FakeOllamaClient(embeddings=[[1.0, 0.0], [0.0, 1.0]])
    result = OllamaEmbedder(client, "nomic-embed-text").embed(["a", "b"])
    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert client.embed_call == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_openai_generator_passes_grounded_instructions():
    client = FakeOpenAIClient(output_text="Answer [Page 2]")
    answer = OpenAIGenerator(client, "gpt-4.1-mini").generate("Question?", "[Page 2]\nEvidence")
    assert answer == "Answer [Page 2]"
    request = client.responses_call
    assert "ONLY" in request["instructions"]
    assert "[Page 2]" in request["input"]
```

Also cover empty response data and SDK connection exceptions. Assert the public error says whether to start Ollama, pull a model, or check API configuration without including the API key.

- [ ] **Step 3: Run RED**

Run: `python3 -m pytest tests/test_providers.py -q`

Expected: import failure for provider classes.

- [ ] **Step 4: Implement the four adapters**

Use client injection in constructors. Ollama calls must use `client.embed(model=..., input=list(texts))` and `client.chat(...)`. OpenAI embeddings use `client.embeddings.create(model=..., input=list(texts))`; generation uses the current Responses API and returns `response.output_text`. Normalize SDK results into plain Python lists/strings at this boundary.

The grounded instruction must require: use only context, say `I cannot find this information in the document.` when unsupported, and cite `[Page N]` when the context supports an answer.

- [ ] **Step 5: Run GREEN**

Run: `python3 -m pytest tests/test_providers.py -q`

Expected: all provider tests pass without network access.

- [ ] **Step 6: Commit**

```bash
git add src/beginner_pdf_rag/providers.py tests/test_providers.py pyproject.toml
git commit -m "feat: support Ollama and OpenAI providers"
```

---

### Task 5: Provider-Independent RAG Pipeline and CLI

**Files:**
- Create: `src/beginner_pdf_rag/pipeline.py`
- Create: `src/beginner_pdf_rag/cli.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces `PdfRag(embedder: Embedder, generator: Generator, chunk_size: int = 1200, overlap: int = 200)`.
- Produces `PdfRag.index(pdf_path: str | Path) -> int` and `PdfRag.ask(question: str, top_k: int = 3) -> RagAnswer`.
- Produces `RagAnswer(text: str, sources: tuple[SearchResult, ...])`.
- Produces CLI `pdf-rag ask --pdf PATH --question TEXT [--provider ollama|openai] [--top-k N] [--show-sources]`.

- [ ] **Step 1: Write pipeline tests with fake providers**

```python
def test_pipeline_indexes_retrieves_and_builds_page_context(monkeypatch):
    monkeypatch.setattr("beginner_pdf_rag.pipeline.load_pdf", lambda _: [Page(4, "alpha beta")])
    embedder = KeywordEmbedder()
    generator = RecordingGenerator("Grounded answer [Page 4]")
    rag = PdfRag(embedder, generator, chunk_size=20, overlap=2)
    assert rag.index("paper.pdf") == 1
    answer = rag.ask("alpha", top_k=1)
    assert answer.text == "Grounded answer [Page 4]"
    assert generator.context.startswith("[Page 4 | score=")


def test_ask_before_index_is_rejected():
    with pytest.raises(RagError, match="index"):
        PdfRag(FakeEmbedder(), FakeGenerator()).ask("question")
```

- [ ] **Step 2: Write CLI tests**

Patch `Settings.from_env`, `build_providers`, and `PdfRag` so tests assert parsed provider, PDF, question, top-k, output answer, and source list. Assert invalid paths and provider errors return nonzero with concise guidance and no traceback.

- [ ] **Step 3: Run RED**

Run: `python3 -m pytest tests/test_pipeline.py tests/test_cli.py -q`

Expected: imports fail for pipeline and CLI.

- [ ] **Step 4: Implement the pipeline**

Index by loading pages, chunking them, and embedding every chunk text in one batch. Ask by embedding one question, ranking chunks, formatting each result exactly as `[Page {page} | score={score:.3f}]\n{text}`, and calling the generator. Reject blank questions and impossible top-k values.

- [ ] **Step 5: Implement the CLI**

Use `argparse`; default provider comes from `RAG_PROVIDER` and falls back to `ollama`. Print a short indexing summary, final answer, and a `Sources` table when requested. Catch only expected configuration/PDF/provider/RAG errors; return exit code 1 and a next action.

- [ ] **Step 6: Run GREEN and full offline suite**

Run:

```bash
python3 -m pytest tests/test_pipeline.py tests/test_cli.py -q
python3 -m pytest -q
python3 -m compileall -q src examples
```

Expected: all tests pass and compilation is silent.

- [ ] **Step 7: Commit**

```bash
git add src/beginner_pdf_rag/pipeline.py src/beginner_pdf_rag/cli.py tests
git commit -m "feat: add cited PDF RAG command"
```

---

### Task 6: Progressive Examples and Licensed Paper

**Files:**
- Create: `examples/01_minimal_text_rag.py`
- Create: `examples/02_pdf_rag_ollama.py`
- Create: `examples/03_pdf_rag_openai.py`
- Create: `data/dpr-paper.pdf`
- Create: `data/README.md`
- Create: `tests/test_examples_contract.py`
- Move: `rag.py` to `legacy/original_pdf_rag.py`
- Create: `legacy/README.md`

**Interfaces:**
- Examples are executable from repository root after `pip install -e .`.
- Example 1 displays the complete algorithm in one file; examples 2 and 3 use the tested package.

- [ ] **Step 1: Add failing artifact contract tests**

Test exact example filenames, absence of `/Users/` in tracked Python/Markdown/config files, PDF magic bytes `%PDF`, expected SHA-256, paper title/author/source/license attribution, and that all examples parse with `ast.parse`.

- [ ] **Step 2: Run RED**

Run: `python3 -m pytest tests/test_examples_contract.py -q`

Expected: missing examples and sample PDF failures.

- [ ] **Step 3: Download and verify the paper**

Download only from `https://aclanthology.org/2020.emnlp-main.550.pdf`, calculate SHA-256, confirm `file data/dpr-paper.pdf` reports PDF, and record the exact checksum in `data/README.md`. Include the full ACL citation, source URLs, and CC BY 4.0 link. Do not claim the authors endorse this tutorial.

- [ ] **Step 4: Write progressive examples**

Example 1 should remain under roughly 150 lines and visibly contain a tiny dataset, embedding calls, cosine similarity, Top-K, prompt, and answer. Examples 2 and 3 should configure the corresponding provider and call the same `PdfRag` pipeline. Each must use relative paths and a `main()` guard.

Move the user's original script unchanged to `legacy/original_pdf_rag.py`, then remove its hardcoded personal PDF path in a follow-up edit by accepting `sys.argv[1]`. `legacy/README.md` must explain that this is the learning draft and direct readers to the maintained CLI.

- [ ] **Step 5: Run GREEN and syntax checks**

Run:

```bash
python3 -m pytest tests/test_examples_contract.py -q
python3 -m compileall -q src examples legacy
git diff --check
```

Expected: tests pass, compilation is silent, diff check is empty.

- [ ] **Step 6: Commit**

```bash
git add examples data legacy tests/test_examples_contract.py
git commit -m "docs: add progressive RAG examples and paper"
```

---

### Task 7: Detailed Beginner Documentation

**Files:**
- Create: `README.md`
- Create: `docs/concepts.md`
- Create: `docs/model-guide.md`
- Create: `docs/troubleshooting.md`
- Create: `LICENSE`
- Create: `tests/test_docs_contract.py`

**Interfaces:**
- README is the canonical command source; supporting docs may elaborate but may not contradict it.

- [ ] **Step 1: Write failing documentation contract tests**

Require README to contain: Python 3.11+, clone/venv/install commands for macOS/Linux and Windows, Ollama installation link and model pulls, one exact `pdf-rag ask` command, expected output, OpenAI setup, `.env` warning, architecture flow, page citations, PDF replacement, scanning/OCR warning, tests, limitations, roadmap, Hugging Face attribution, ACL paper attribution, license, and troubleshooting links.

Also extract relative Markdown links and assert every referenced local file exists.

- [ ] **Step 2: Run RED**

Run: `python3 -m pytest tests/test_docs_contract.py -q`

Expected: README and supporting documents are missing.

- [ ] **Step 3: Write the README as a runnable lesson**

Use this order:

1. One-sentence value proposition and feature list.
2. A visible Mermaid RAG flow.
3. “15 分钟跑通” Ollama path with expected terminal output.
4. Explanation of what just happened.
5. Step-by-step code walkthrough linked to the three examples.
6. OpenAI-compatible API path with safe `.env` setup.
7. How to replace the PDF and ask good questions.
8. Tests, limitations, roadmap, contributing, attribution, license.

Do not promise guaranteed hallucination elimination, production readiness, or a specific star count. Use plain Chinese and define each English term on first use.

- [ ] **Step 4: Write the three supporting guides**

`concepts.md` explains token/chunk/overlap/embedding/vector/cosine similarity/Top-K/context window/prompt/hallucination and includes the cosine formula plus a two-dimensional worked example.

`model-guide.md` contains a comparison table for privacy, setup, cost, offline use, speed, hardware, and model choice. Explain that the embedding model and chat model are separate. Link to official Ollama and OpenAI model/API docs instead of copying volatile lists or prices.

`troubleshooting.md` organizes exact symptoms: command not found, Python version, Ollama connection, missing model, out of memory, API key, 401/429, empty PDF, garbled PDF, scanned PDF/OCR, slow embedding, irrelevant retrieval, and unsupported answer.

- [ ] **Step 5: Add licensing**

Use MIT for repository-authored code and documentation. Explain in README and `data/README.md` that the bundled paper retains its own CC BY 4.0 license and attribution.

- [ ] **Step 6: Run GREEN and prose hygiene checks**

Run:

```bash
python3 -m pytest tests/test_docs_contract.py -q
rg -n '/Users/|OPENAI_API_KEY\s*=\s*sk-' README.md docs examples src .env.example
rg -n 'TODO|TBD|稍后补充|待完善' README.md docs
git diff --check
```

Expected: docs tests pass; prohibited scans have no matches; diff check is empty.

- [ ] **Step 7: Commit**

```bash
git add README.md docs LICENSE tests/test_docs_contract.py
git commit -m "docs: write zero-background PDF RAG tutorial"
```

---

### Task 8: Real Smoke Tests and Fresh-Reader Review

**Files:**
- Create: `scripts/smoke_ollama.sh`
- Create: `docs/verification.md`
- Modify only if review finds gaps: `README.md`, `docs/*.md`, `src/beginner_pdf_rag/*.py`

**Interfaces:**
- `scripts/smoke_ollama.sh` runs the bundled paper with a deterministic question and never downloads models silently.

- [ ] **Step 1: Verify a clean installation**

Create a temporary virtual environment outside the repository, install `.[test]`, run `pdf-rag --help`, and run `pytest -q`. Record Python/pip versions and counts in `docs/verification.md`; do not commit absolute temporary paths.

- [ ] **Step 2: Run the real Ollama path**

Check `ollama --version`, `ollama list`, and `curl http://127.0.0.1:11434/api/tags`. If required models are absent, pull only the two names documented in README. Run:

```bash
pdf-rag ask \
  --pdf data/dpr-paper.pdf \
  --question "What datasets are used to evaluate DPR?" \
  --provider ollama \
  --top-k 3 \
  --show-sources
```

Verify the answer is nonempty, cites at least one `[Page N]`, and the displayed source scores are finite. Record the exact tested model names and outcome, not the full generated answer.

- [ ] **Step 3: Verify the OpenAI path safely**

Always run the missing-key test. Only send a live request if an existing `OPENAI_API_KEY` is already configured and the user has authorized cost; otherwise record `not run: no authorized API request` rather than treating it as a failure.

- [ ] **Step 4: Conduct fresh-reader testing**

Give a reviewer only the repository content and ask:

1. How do I run the default example from a fresh machine?
2. What is the difference between the embedding and chat models?
3. How do I switch to OpenAI without committing my key?
4. How do I replace the sample PDF?
5. Why are pages and overlap preserved?
6. What should I do with a scanned PDF?
7. How can I tell which chunks supported the answer?
8. What does this project deliberately not implement?

Treat any answer that requires prior conversation knowledge as a documentation failure. Fix gaps surgically, rerun `tests/test_docs_contract.py`, and record the review result.

- [ ] **Step 5: Run full verification**

```bash
python3 -m pytest -q
python3 -m compileall -q src examples legacy
python3 -m build
git diff --check
git status --short
```

Expected: tests/build pass; worktree contains only intended changes.

- [ ] **Step 6: Commit verification artifacts**

```bash
git add scripts/smoke_ollama.sh docs/verification.md README.md docs src tests
git commit -m "test: verify beginner RAG learning path"
```

---

### Task 9: GitHub Publication

**Files:**
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/learning_question.yml`
- Create: `.github/pull_request_template.md`
- Create: `CONTRIBUTING.md`

**Interfaces:**
- Produces the public repository `https://github.com/kannseiLiu/rag-from-scratch-for-beginners`.

- [ ] **Step 1: Add lightweight contribution templates**

Bug reports must request OS, Python version, provider, exact safe error, reproduction steps, and must warn users to remove keys/private PDF text. Learning questions must ask which tutorial step is unclear. `CONTRIBUTING.md` must show install, test, formatting expectations, and privacy rules.

- [ ] **Step 2: Run final repository safety audit**

```bash
git grep -nE '/Users/|sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]+'
git ls-files | rg '(^|/)(\.env|__pycache__|\.DS_Store)(/|$)|\.pyc$'
git status --short
git log --oneline --decorate -12
```

Expected: no personal paths, secrets, caches, or uncommitted intended files.

- [ ] **Step 3: Commit GitHub metadata**

```bash
git add .github CONTRIBUTING.md
git commit -m "docs: add contributor guidance"
```

- [ ] **Step 4: Create and push the public repository**

First verify `gh auth status` shows `kannseiLiu`. Then run:

```bash
gh repo create kannseiLiu/rag-from-scratch-for-beginners \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "从零实现 PDF RAG：Ollama 本地运行 + OpenAI API，面向初学者的中文教程"
```

If the repository already exists, stop and inspect it rather than overwriting or force-pushing.

- [ ] **Step 5: Configure discoverability**

```bash
gh repo edit kannseiLiu/rag-from-scratch-for-beginners \
  --add-topic rag \
  --add-topic pdf-rag \
  --add-topic ollama \
  --add-topic openai \
  --add-topic llm \
  --add-topic python \
  --add-topic chinese-tutorial \
  --enable-issues
```

Verify the default branch is `main`, repository visibility is public, README renders, PDF downloads, and the clone URL works in a temporary directory.

- [ ] **Step 6: Report publication**

Return the GitHub URL, tested quick-start command, test count, live Ollama outcome, whether OpenAI incurred any request, and any honest limitations. Do not claim or manufacture stars.
