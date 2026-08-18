#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Joint-Embedding-Guided-Policy-Optimization
CKPT_ROOT=$ROOT/checkpoints/tafr-repro-math15b
M09=fepo-m09_ngrpo_fepo-20260812_avspo_ngrpo_token_beta05
M10=fepo-m10_avspo_fepo-20260812_avspo_ngrpo_token_beta05
P09=1545443
P10=1545437
R09=ismamNur/fepo-m09-ngrpo-full
R10=ismamNur/fepo-m10-avspo-full
LOG=$ROOT/logs/posttrain_7b_pipeline.log
exec >>$LOG 2>&1
cd $ROOT
set -a; source $ROOT/.env; set +a
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
log(){ echo "[$(date -u)] $*"; }
log 'Pipeline started; waiting for both training processes.'
for p in $P09 $P10; do while kill -0 $p 2>/dev/null; do sleep 60; done; log "training PID $p ended"; done
sleep 90
for r in $M09 $M10; do [[ -d $CKPT_ROOT/$r/global_step_500 ]] || { log "missing final checkpoint $r"; exit 2; }; done
log 'Both final checkpoints present; uploading full run directories.'
HF_REPO=$R09 HF_PRIVATE=true $ROOT/scripts/push_full_checkpoint_repo.sh $CKPT_ROOT/$M09
HF_REPO=$R10 HF_PRIVATE=true $ROOT/scripts/push_full_checkpoint_repo.sh $CKPT_ROOT/$M10
log 'Uploads returned successfully; verifying expected paths via HF API.'
$ROOT/.venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ['HUGGING_FACE_HUB_TOKEN'])
for repo in ['ismamNur/fepo-m09-ngrpo-full','ismamNur/fepo-m10-avspo-full']:
 files={x.rfilename for x in api.list_repo_tree(repo,repo_type='model',recursive=True)}
 for needle in ['global_step_500','best_pass_at_1']:
  if not any(needle in f for f in files): raise SystemExit(f'{repo}: missing {needle}')
 print(repo, 'verified', len(files), 'files')
PY
log 'HF verification passed; deleting only the two completed local run directories.'
rm -rf $CKPT_ROOT/$M09 $CKPT_ROOT/$M10
log 'Local checkpoint deletion complete.'
[[ -d /workspace/models/Qwen2.5-Math-7B-Instruct ]] || { log '7B model absent; downloading before launch.'; $ROOT/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Math-7B-Instruct',local_dir='/workspace/models/Qwen2.5-Math-7B-Instruct',allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*'])
PY
}
log 'Launching verified 2-GPU 7B GRPO training.'
tmux new-session -d -s grpo_7b_train "cd $ROOT && GPUS=0,1 bash experiments/fepo_paper/s03_grpo_7b.sh"
log '7B training queued in tmux session grpo_7b_train.'
