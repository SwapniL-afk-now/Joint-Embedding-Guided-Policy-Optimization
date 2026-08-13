#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/Joint-Embedding-Guided-Policy-Optimization
LOG=$ROOT/eval_results/official_eval_pipeline.log
exec >>$LOG 2>&1
cd $ROOT
for s in official_eval_m09 official_eval_m10; do while tmux has-session -t $s 2>/dev/null; do sleep 30; done; done
echo "official final evaluations finished $(date -u)"
$ROOT/.venv/bin/python - <<'PY'
import json, os
root='/workspace/eval_results'
for bench in ['math_core.json','math_extra.json','code.json']:
 print('===',bench)
 for arm in ['fepo_m09-official','fepo_m10-official']:
  p=f'{root}/{arm}/{bench}'
  if not os.path.exists(p): print(arm,'MISSING'); continue
  d=json.load(open(p)); print(arm, {k:round(v.get('pass@1_mean',0),6) for k,v in d.get('per_benchmark',{}).items()})
PY
for arm in m09 m10; do
 case $arm in m09) RUN=fepo-m09_ngrpo_fepo-20260812_avspo_ngrpo_token_beta05;; m10) RUN=fepo-m10_avspo_fepo-20260812_avspo_ngrpo_token_beta05;; esac
 ACTOR=$ROOT/checkpoints/tafr-repro-math15b/$RUN/best_pass_at_1/actor
 MODEL=$ROOT/eval_models/$arm-best_pass_at_1
 mkdir -p "$ROOT/eval_models"
 if [[ ! -f "$MODEL/config.json" ]]; then $ROOT/.venv/bin/python scripts/legacy_model_merger.py merge --backend fsdp --local_dir "$ACTOR" --target_dir "$MODEL"; fi
done
tmux new-session -d -s best_eval_m09 "CUDA_VISIBLE_DEVICES=0 bash /tmp/run_official_eval_best.sh m09 > $ROOT/eval_results/fepo_m09-best.log 2>&1"
tmux new-session -d -s best_eval_m10 "CUDA_VISIBLE_DEVICES=1 bash /tmp/run_official_eval_best.sh m10 > $ROOT/eval_results/fepo_m10-best.log 2>&1"
