#!/usr/bin/env bash
# sc_jepa-adv-8cot-seed31415-666.sh — ADVANTAGE-ALIGNED JEPA arm of the controlled
# comparison, vs sc_jepa_4cot4code_400.sh (the audited baseline, now under the
# 8cot-400 naming).
# This arm enables the advantage-aligned JEPA fix (ADVANTAGE_ALIGNED_JEPA.md):
#   - jepa.adv_weighting=True: every sampled rollout of a prompt is an anchor;
#     pulls weighted by softmax(Â/tau) over the prompt group (tau auto = group
#     advantage std, floored at 1e-3) — shared sample support with the Dr.GRPO
#     policy gradient (AWR identity), aligning the two gradients (cos ~ 0 fix).
#   - jepa.adv_target=best: pull every anchor of a prompt toward the embedding of
#     the group's highest-advantage response (no teacher text involved).
#   - jepa.neg_pull=True: repel arm — anchors with Â < 0 are pushed away from
#     their OWN response embedding by max(0, neg_margin - cos), weighted by
#     softmax(-Â/tau) (strict Â < 0 only).
# Everything else identical to the baseline: full FT (lora_rank=0), llm-jepa,
# self_code_targets=True, TRAIN_SEED=31415, ENABLE_OVERLONG_BUFFER=true,
# tcr_reward_beta=0, no teacher targets, 8 CoT only, separate GRPO + JEPA optimizer
# steps, AUTO_OFF_ENABLE=False, GPU0 ray head 127.0.0.1:37379.
# Checkpoints: checkpoints/grpo-qwen-3b/jepa-adv-8cot-seed31415-666
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_jepa-adv-8cot-seed31415-666.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_jepa-adv-8cot-seed31415-666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
CUDA_VISIBLE_DEVICES=0 \
N_COT=8 \
N_CODE=0 \
ROLLOUT_N=8 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
AUTO_OFF_ENABLE=False \
DAPO_USE_LORA=false \
ACTOR_LR=1e-6 \
TRAIN_FILE=/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet \
TRAIN_SEED=31415 \
VALIDATION_SEEDS='[31415,271828,14142,16180,299792]' \
VAL_FILES='[/workspace/jepa-grpo-cache/eval_data/aime24.parquet,/workspace/jepa-grpo-cache/eval_data/aime25.parquet,/workspace/jepa-grpo-cache/eval_data/aime26.parquet,/workspace/jepa-grpo-cache/eval_data/amc23.parquet]' \
BEST_CKPT_SOURCES='["aime24","aime25","aime26","amc23"]' \
ENABLE_OVERLONG_BUFFER=true \
NDEVICES_PER_NODE=1 \
TOTAL_TRAINING_BATCHES=666 \
MAX_OPTIMIZER_STEPS=666 \
TOTAL_TRAINING_STEPS=666 \
VAL_BATCH_SIZE=128 \
PROJECT_NAME=grpo-qwen-3b \
EXPERIMENT_NAME=jepa-adv-8cot-seed31415-666 \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_jepa-adv-8cot-seed31415-666.jsonl \
./run_drgrpo_jepa.sh \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  actor_rollout_ref.rollout.max_num_seqs=2048 \
  jepa.self_code_targets=True \
  jepa.paper_sigreg_lambda=0.1 \
  jepa.fuse_optimizer_step=True \
  jepa.tcr_reward_beta=0 \
  jepa.teacher_cache_path="" \
  jepa.code_teacher_cache_path="" \
  jepa.adv_weighting=True \
  jepa.adv_target=best \
  jepa.adv_tau=0.0 \
  jepa.neg_pull=True \
  jepa.neg_margin=0.2 \
  +ray_kwargs.ray_init.address=127.0.0.1:37379 2>&1 | tee /workspace/exploration/grad_direction_probe/log_jepa-adv-8cot-seed31415-666.log
