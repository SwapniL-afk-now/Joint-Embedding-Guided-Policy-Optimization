#!/usr/bin/env bash
# Minimal setup for the DeepSeek-Math-7B compact study. Does not download the
# unused Qwen/Llama models or cross-domain datasets from the full paper setup.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PY=${PYTHON_BIN:-.venv/bin/python}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet}
mkdir -p "$EVAL_DATA_DIR"
"$PY" experiments/fepo_paper/test_paper_additions.py >/dev/null
"$PY" experiments/fepo_paper/test_dapo_accumulate.py >/dev/null
if [[ ! -f "$EVAL_DATA_DIR/probe256.parquet" ]]; then
  "$PY" experiments/fepo_paper/make_probe.py
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "MISSING training file: $TRAIN_FILE" >&2
  exit 4
fi
snapshot="$($PY - <<'PY'
from huggingface_hub import snapshot_download
try:
    print(snapshot_download("deepseek-ai/deepseek-math-7b-instruct", local_files_only=True))
except Exception:
    print("")
PY
)"
if [[ -z "$snapshot" || ! -f "$snapshot/config.json" ]]; then
  snapshot="$($PY - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("deepseek-ai/deepseek-math-7b-instruct"))
PY
)"
fi
MODEL_PATH="$snapshot" "$PY" - <<'PY'
import os, json
from transformers import AutoTokenizer, AutoConfig
path=os.environ["MODEL_PATH"]
cfg=AutoConfig.from_pretrained(path, trust_remote_code=False)
assert cfg.model_type == "llama", cfg.model_type
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=False)
assert tok.eos_token_id is not None
print(f"DeepSeek ready: {path}; model_type={cfg.model_type}; context={cfg.max_position_embeddings}; eos={tok.eos_token_id}")
PY
