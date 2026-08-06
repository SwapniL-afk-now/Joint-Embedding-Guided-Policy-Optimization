#!/usr/bin/env bash
# sc_grpo_4cot4code_666.sh — GRPO arm of the 2-GPU controlled comparison.
# Runs on GPU1 (CUDA_VISIBLE_DEVICES=1) against a dedicated ray head
# (ray start --head --port=16379 --dashboard-port=18265) so it cannot
# collide with the JEPA arm's own head on default ports.
#
# Identical to the JEPA arm (sc_jepa_4cot4code_666.sh) EXCEPT:
#   - JEPA_ENABLE=False -> jepa.enable=False: pure DAPO-GRPO, no JEPA loss,
#     no TCR shaping, no EMA/predictor; both views (4 cot + 4 code) still
#     roll out and get verifier rewards (the trainer's supported
#     JEPA-vs-baseline comparison mode)
#   - EXPERIMENT_NAME grpo_4cot4code_666
#   - ray_kwargs.ray_init.address -> the dedicated GPU1 ray head
# Everything else identical: full FT, TRAIN_SEED=31415, 3 validation seeds,
# ENABLE_OVERLONG_BUFFER=true, tcr_reward_beta=0, 666 steps.
# Checkpoints: checkpoints/grpo-qwen-3b/grpo_4cot4code_666
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_grpo_4cot4code_666.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_grpo_4cot4code_666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
CUDA_VISIBLE_DEVICES=1 \
JEPA_ENABLE=False \
N_COT=4 \
N_CODE=4 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
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
EXPERIMENT_NAME=grpo_4cot4code_666 \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_grpo_4cot4code_666.jsonl \
./run_drgrpo_jepa.sh \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  jepa.self_code_targets=True \
  jepa.paper_sigreg_lambda=0.1 \
  jepa.fuse_optimizer_step=True \
  jepa.tcr_reward_beta=0 \
  jepa.teacher_cache_path=/workspace/jepa-grpo-cache/data/amcaime32k/teacher.cot.responses.pt \
  +ray_kwargs.ray_init.address=127.0.0.1:36379 2>&1 | tee /workspace/exploration/grad_direction_probe/log_grpo_4cot4code_666.log