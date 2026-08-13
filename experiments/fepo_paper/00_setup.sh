#!/usr/bin/env bash
# One-time setup for the FEPO paper program. Idempotent -- safe to re-run.
#
#   bash experiments/fepo_paper/00_setup.sh
#
# Downloads the two missing models, builds the probe set and the cross-domain
# parquets, and runs the self-checks. After this, every arm script is runnable.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

PY=${PYTHON_BIN:-.venv/bin/python}
MODELS=/workspace/models

# --- self-checks first: a broken addition should surface before any download ---
echo "== self-checks =="
"${PY}" experiments/fepo_paper/test_paper_additions.py | tail -1
"${PY}" experiments/fepo_paper/test_dapo_accumulate.py | tail -1

# --- probe set (Table 2) -------------------------------------------------------
echo
echo "== probe set =="
if [[ -f /workspace/jepa-grpo-cache/eval_data/probe256.parquet ]]; then
  echo "already built (delete probe256.parquet + train_minus_probe.parquet to rebuild)"
else
  "${PY}" experiments/fepo_paper/make_probe.py
fi

# --- models --------------------------------------------------------------------
# 1.5B is already local. These two are only needed by s01/s02 and x01/x02.
fetch() {
  local repo=$1 dest=$2
  if [[ -d "${dest}" ]]; then echo "have ${dest}"; return; fi
  echo "downloading ${repo} -> ${dest}"
  "${PY}" -c "
from huggingface_hub import snapshot_download
snapshot_download('${repo}', local_dir='${dest}',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
"
}

echo
echo "== models =="
fetch Qwen/Qwen2.5-Math-7B-Instruct   "${MODELS}/Qwen2.5-Math-7B-Instruct"
# Llama-3.2 is a gated repo: accept the license on its HF page and have a token in
# HF_TOKEN or ~/.cache/huggingface, or this one 401s while the rest still succeed.
fetch meta-llama/Llama-3.2-3B-Instruct "${MODELS}/Llama-3.2-3B-Instruct" || {
  echo "  Llama download failed (gated repo?). x01/x02 stay blocked; everything else runs." >&2
}

# --- DeepSeek-Math-7B compact study ------------------------------------------
# The compact plan uses this model for every retained experiment. Prefer an
# existing HF snapshot to avoid copying a second 7B checkpoint into /workspace.
echo
 echo "== DeepSeek-Math-7B =="
deepseek_snapshot="$("${PY}" - <<'PY'
from huggingface_hub import snapshot_download
try:
    print(snapshot_download("deepseek-ai/deepseek-math-7b-instruct", local_files_only=True))
except Exception:
    print("")
PY
)"
if [[ -n "${deepseek_snapshot}" && -f "${deepseek_snapshot}/config.json" ]]; then
  echo "using cached DeepSeek snapshot: ${deepseek_snapshot}"
else
  echo "downloading deepseek-ai/deepseek-math-7b-instruct"
  "${PY}" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("deepseek-ai/deepseek-math-7b-instruct")
PY
fi

# --- cross-domain data (Table 5) ----------------------------------------------
echo
echo "== cross-domain data =="
if [[ -f /workspace/jepa-grpo-cache/eval_data/ifeval.parquet ]]; then
  echo "already built"
else
  "${PY}" experiments/fepo_paper/prepare_crossdomain_data.py || {
    echo "  cross-domain prep failed; x01/x02 stay blocked, Tables 1-4 and 6-8 unaffected." >&2
  }
fi

echo
echo "== ready =="
echo "Next: experiments/fepo_paper/run_deepseek7b_compact.sh --dry-run"
