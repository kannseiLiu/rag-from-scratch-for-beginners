# Task 5 Report: Provider-Independent PDF RAG Pipeline and CLI

## RED evidence

`python3 -m pytest tests/test_pipeline.py tests/test_cli.py -q` initially failed with 22 expected import failures because `beginner_pdf_rag.pipeline` and `beginner_pdf_rag.cli` did not exist.

The environment-provider regression test was also run against a temporary `ollama`-only parser default and failed as expected: it received `ollama` while `RAG_PROVIDER=openai` required `openai`.

The malformed query-embedding regression test failed as expected with an `IndexError` before the pipeline checked for exactly one query vector.

## GREEN evidence

`python3 -m pytest tests/test_pipeline.py tests/test_cli.py -q`

Result: `24 passed in 0.61s`.

## Full-suite and compilation evidence

`python3 -m pytest -q`

Result: `94 passed in 0.57s`.

`python3 -m compileall -q src`

Result: silent success.

`python3 -m compileall -q src examples`

Result: exit code 0, with `Can't list 'examples'`. This worktree does not contain an `examples/` directory; no directory or sample content was added because it is outside this task's scope.

## Self-review

- `PdfRag` batches document embeddings, embeds the query once, ranks through the existing retriever, and returns the same retrieved `SearchResult` objects as immutable answer sources.
- Context uses the required `[Page {page} | score={score:.3f}]` format and the generator receives only that ranked context.
- The CLI honors `RAG_PROVIDER` when present, falls back to Ollama when missing, validates positive integer `--top-k`, and catches only expected setup/PDF/RAG/provider runtime errors without exposing tracebacks.
- Tests use fakes and monkeypatches only; they make no paid or network calls.
