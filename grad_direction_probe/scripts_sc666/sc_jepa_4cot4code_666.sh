#!/usr/bin/env bash
# sc_jepa_4cot4code_666.sh — JEPA arm of the 2-GPU controlled comparison.
# Identical to sc_fixed_ff_v3.sh EXCEPT:
#   - VALIDATION_SEEDS reduced to 3 seeds [31415,271828,14142] (eval protocol
#     shared with the GRPO arm sc_grpo_4cot4code_666.sh)
#   - fresh EXPERIMENT_NAME jepa_4cot4code_666 (val-seed change requires a
#     new name; same name would auto-resume the stale 5-seed checkpoint)
#   - AUTO_OFF_ENABLE=False: the JEPA signal must stay ON for the whole run
#     (script default True auto-disables it after auto_off_patience plateau)
# Everything else is the v4 setup: full FT, llm-jepa, self_code_targets=True,
# TRAIN_SEED=31415, ENABLE_OVERLONG_BUFFER=true, tcr_reward_beta=0,
# amcaime32k teacher cache, 4 cot + 4 code, 666 steps.
# Checkpoints: checkpoints/grpo-qwen-3b/jepa_4cot4code_666
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_jepa_4cot4code_666.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_jepa_4cot4code_666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
CUDA_VISIBLE_DEVICES=0 \
N_COT=4 \
N_CODE=4 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
AUTO_OFF_ENABLE=False \
DAPO_USE_LORA=false \
ACTOR_LR=1e-6 \
TRAIN_FILE=/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet \
TRAIN_SEED=31415 \
VALIDATION_SEEDS='[31415,271828,14142]' \
ENABLE_OVERLONG_BUFFER=true \
NDEVICES_PER_NODE=1 \
MAX_OPTIMIZER_STEPS=666 \
TOTAL_TRAINING_STEPS=666 \
PROJECT_NAME=grpo-qwen-3b \
EXPERIMENT_NAME=jepa_4cot4code_666 \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_jepa_4cot4code_666.jsonl \
./run_drgrpo_jepa.sh \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  jepa.self_code_targets=True \
  jepa.paper_sigreg_lambda=0.1 \
  jepa.fuse_optimizer_step=True \
  jepa.tcr_reward_beta=0 \
  jepa.teacher_cache_path=/workspace/jepa-grpo-cache/data/amcaime32k/teacher.cot.responses.pt \
  +ray_kwargs.ray_init.address=127.0.0.1:37379 2>&1 | tee /workspace/exploration/grad_direction_probe/log_jepa_4cot4code_666.log
