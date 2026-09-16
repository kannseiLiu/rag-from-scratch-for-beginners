#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repository_root"

readonly smoke_ollama_base_url="http://127.0.0.1:11434"
readonly smoke_embedding_model="nomic-embed-text"
readonly smoke_chat_model="qwen3:4b"
readonly required_models=("$smoke_embedding_model" "$smoke_chat_model")

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download" >&2
  exit 1
fi

ollama --version
ollama list
if ! curl --fail --silent --show-error "$smoke_ollama_base_url/api/tags" >/dev/null; then
  echo "Ollama daemon is not reachable at $smoke_ollama_base_url. Start Ollama and retry." >&2
  exit 1
fi

for model in "${required_models[@]}"; do
  if ! ollama show "$model" >/dev/null 2>&1; then
    echo "Required Ollama model '$model' is not installed. Run: ollama pull $model" >&2
    exit 1
  fi
done

smoke_output=$(mktemp "${TMPDIR:-/tmp}/pdf-rag-smoke.XXXXXX")
trap 'rm -f "$smoke_output"' EXIT

set +e
RAG_PROVIDER=ollama \
OLLAMA_BASE_URL="$smoke_ollama_base_url" \
OLLAMA_EMBEDDING_MODEL="$smoke_embedding_model" \
OLLAMA_CHAT_MODEL="$smoke_chat_model" \
pdf-rag ask \
  --pdf data/dpr-paper.pdf \
  --question "What datasets are used to evaluate DPR?" \
  --provider ollama \
  --top-k 3 \
  --show-sources >"$smoke_output" 2>&1
cli_status=$?
set -e

cat "$smoke_output"
if [ "$cli_status" -ne 0 ]; then
  echo "Smoke verification failed: pdf-rag exited with status $cli_status." >&2
  exit "$cli_status"
fi

if ! awk '
  BEGIN { answer = 0; citation = 0; valid = 0 }
  /^Sources$/ { valid = (answer && citation); exit }
  /^Indexed [0-9]+ chunks\.$/ { next }
  /^[[:space:]]*$/ { next }
  { answer = 1; if ($0 ~ /\[Page [0-9]+\]/) citation = 1 }
  END { exit(valid ? 0 : 1) }
' "$smoke_output"; then
  echo "Smoke verification failed: CLI output needs a non-empty cited answer before Sources." >&2
  exit 1
fi

if ! awk -F'|' '
  BEGIN { in_sources = 0; in_table = 0; valid = 0 }
  /^Sources$/ { in_sources = 1; next }
  in_sources && /^Page[[:space:]]+\|[[:space:]]+Score[[:space:]]*$/ { in_table = 1; next }
  in_table && /^----/ { next }
  in_table && NF >= 2 {
    score = $2
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", score)
    if (score ~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/) valid = 1
  }
  END { exit(valid ? 0 : 1) }
' "$smoke_output"; then
  echo "Smoke verification failed: Sources must contain at least one finite numeric score." >&2
  exit 1
fi
