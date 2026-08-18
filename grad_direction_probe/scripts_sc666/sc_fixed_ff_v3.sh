#!/usr/bin/env bash
# sc_fixed_ff_v3 — fixed-code JEPA run. Same setup as sc_fixed_ff.sh v2
# (full FT, paper-exact anchor, llm-jepa, self_code_targets=True) PLUS:
#   - train/accuracy is now MERGED per-prompt any-correct over cot+code
#     (trainer change in _compute_train_comparison_metrics)
#   - code view prompt fixed: user message gets an explicit output-format
#     line (jepa.code_format_line, config default) — measured in the
#     code-prompt probe to lift code output-match ~0.20 -> ~0.24, fences ~97%
#   - TRAIN_SEED=31415 + VALIDATION_SEEDS matching fw7zf51z EXACTLY
#     (the baseline ran data_loader_seed=31415; sc runs wrongly used 14142,
#     which changed the prompt batches and broke step-by-step comparability)
#   - jepa.tcr_reward_beta=0 (paper-exact llm-jepa; the 0.5 default folded a
#     β·ŝ teacher-alignment term into rewards using the WRONG (dapo-keyed)
#     teacher caches — random-text similarity noise in every sc run)
#   - jepa.teacher_cache_path -> the real amcaime32k teacher cache
#     (data/amcaime32k/teacher.cot.responses.pt, keys 0..21365)
#   - ENABLE_OVERLONG_BUFFER=true to match the baseline's RL signal
# Checkpoints: checkpoints/grpo-qwen-3b/selfcode_fixedcode_v4_666
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_sc_fixedcode_v4_666.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_sc_fixedcode_v4_666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
N_COT=4 \
N_CODE=4 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
DAPO_USE_LORA=false \
ACTOR_LR=1e-6 \
TRAIN_FILE=/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet \
TRAIN_SEED=31415 \
VALIDATION_SEEDS='[31415,271828,14142,16180,299792]' \
ENABLE_OVERLONG_BUFFER=true \
NDEVICES_PER_NODE=1 \
MAX_OPTIMIZER_STEPS=666 \
TOTAL_TRAINING_STEPS=666 \
PROJECT_NAME=grpo-qwen-3b \
EXPERIMENT_NAME=selfcode_fixedcode_v4_666 \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_sc_fixedcode_v4_666.jsonl \
./run_drgrpo_jepa.sh \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  jepa.self_code_targets=True \
  jepa.paper_sigreg_lambda=0.1 \
  jepa.fuse_optimizer_step=True \
  jepa.tcr_reward_beta=0 \
  jepa.teacher_cache_path=/workspace/jepa-grpo-cache/data/amcaime32k/teacher.cot.responses.pt 2>&1 | tee /workspace/exploration/grad_direction_probe/log_sc_fixedcode_v4_666.log
