#!/usr/bin/env bash
# Push the TAFR prod run to HF, verify it landed, then delete the local copy.
#
#   HF_TOKEN=hf_xxx bash push_tafr_to_hf.sh
#   # or, to reuse the token already in the gxpo project:
#   HF_TOKEN=$(grep -oP '(?<=HF_TOKEN=)["'"'"']?\Khf_[A-Za-z0-9]+' \
#       /workspace/gradient-extrapolation-based-policy-optimization/.env) bash push_tafr_to_hf.sh
#
# Deletes nothing until every uploaded file is confirmed present in the repo listing.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source .venv/bin/activate
: "${HF_TOKEN:?set HF_TOKEN}"

REPO=${REPO:-ismamNur/Joint-Embedding-Guided-Policy-Optimization-checkpoints}
DELETE=${DELETE:-1}   # DELETE=0 to upload only

python - "$REPO" "$DELETE" <<'PY'
import os, shutil, sys
from huggingface_hub import HfApi

repo, delete = sys.argv[1], sys.argv[2] == "1"
api = HfApi(token=os.environ["HF_TOKEN"])
print("authed as", api.whoami()["name"], "-> ", repo, flush=True)

# (local dir, path in repo, delete local after verify)
jobs = [
    ("/workspace/eval_ckpts/tafr_step280_hf",  "eval_models/tafr_step280_hf",  False),
    ("/workspace/eval_ckpts/grpo_step350_hf",  "eval_models/grpo_step350_hf",  False),
    ("checkpoints/grpo-qwen-3b/tafr-prod-b05", "grpo-qwen-3b/tafr-prod-b05",   True),
]

for local, remote, drop in jobs:
    if not os.path.isdir(local):
        print(f"SKIP {local} (missing)"); continue
    print(f"\nuploading {local} -> {remote}", flush=True)
    api.upload_folder(folder_path=local, path_in_repo=remote, repo_id=repo,
                      repo_type="model", commit_message=f"add {remote}")

    # verify: every local file must appear in the remote listing at the right size
    remote_files = set(api.list_repo_files(repo))
    missing = []
    for root, _, files in os.walk(local):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, local)
            if f"{remote}/{rel}" not in remote_files:
                missing.append(rel)
    if missing:
        print(f"  VERIFY FAILED, {len(missing)} missing e.g. {missing[:3]} -- not deleting")
        continue
    n = sum(len(fs) for _, _, fs in os.walk(local))
    print(f"  verified {n} files present on HF")
    if drop and delete:
        shutil.rmtree(local)
        print(f"  deleted local {local}")
    elif drop:
        print(f"  DELETE=0, keeping {local}")
PY
df -h /workspace | tail -1
