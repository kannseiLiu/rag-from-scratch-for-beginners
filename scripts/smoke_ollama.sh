#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repository_root"

required_models=("nomic-embed-text" "qwen3:4b")

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download" >&2
  exit 1
fi

ollama --version
ollama list
if ! curl --fail --silent --show-error http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Ollama daemon is not reachable at http://127.0.0.1:11434. Start Ollama and retry." >&2
  exit 1
fi

for model in "${required_models[@]}"; do
  if ! ollama show "$model" >/dev/null 2>&1; then
    echo "Required Ollama model '$model' is not installed. Run: ollama pull $model" >&2
    exit 1
  fi
done

exec pdf-rag ask \
  --pdf data/dpr-paper.pdf \
  --question "What datasets are used to evaluate DPR?" \
  --provider ollama \
  --top-k 3 \
  --show-sources
