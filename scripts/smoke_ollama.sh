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

RAG_PROVIDER=ollama \
OLLAMA_BASE_URL="$smoke_ollama_base_url" \
OLLAMA_EMBEDDING_MODEL="$smoke_embedding_model" \
OLLAMA_CHAT_MODEL="$smoke_chat_model" \
exec pdf-rag ask \
  --pdf data/dpr-paper.pdf \
  --question "What datasets are used to evaluate DPR?" \
  --provider ollama \
  --top-k 3 \
  --show-sources
