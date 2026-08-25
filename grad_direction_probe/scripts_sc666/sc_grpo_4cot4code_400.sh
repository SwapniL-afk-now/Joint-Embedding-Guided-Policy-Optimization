#!/usr/bin/env bash
# sc_grpo-8cot-seed31415-666.sh — GRPO arm (v2) of the 2-GPU controlled comparison.
# Runs on GPU1 (CUDA_VISIBLE_DEVICES=1) against the dedicated GPU1 ray head
# (127.0.0.1:36379).
#
# Identical to the JEPA arm (sc_jepa_8cot_400.sh) EXCEPT:
#   - JEPA_ENABLE=False -> jepa.enable=False: pure DAPO-GRPO, no JEPA loss,
#     no TCR shaping, no EMA/predictor; both 8 CoT rollouts still
#     roll out and get verifier rewards
#   - EXPERIMENT_NAME grpo-8cot-seed31415-666
#   - ray_kwargs.ray_init.address -> the dedicated GPU1 ray head
# Everything else identical: full FT, TRAIN_SEED=31415, same five validation seeds
# [31415], VAL_FILES [aime24, aime25, aime26, amc23], vLLM
# gpu_memory_utilization 0.7 / max_num_batched_tokens 131072 / max_num_seqs
# 2048, ENABLE_OVERLONG_BUFFER=true, tcr_reward_beta=0, 400 steps,
# PROJECT_NAME=grpo-qwen-3b (new wandb project shared with the JEPA arm).
# Checkpoints: checkpoints/grpo-qwen-3b-400/grpo-8cot-seed31415-666
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_grpo-8cot-seed31415-666.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_grpo-8cot-seed31415-666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
CUDA_VISIBLE_DEVICES=1 \
JEPA_ENABLE=False \
N_COT=8 \
N_CODE=0 \
ROLLOUT_N=8 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
DAPO_USE_LORA=false \
ACTOR_LR=1e-6 \
TRAIN_FILE=/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet \
TRAIN_SEED=31415 \
VALIDATION_SEEDS='[31415,271828,14142,16180,299792]' \
VAL_FILES='[/workspace/jepa-grpo-cache/eval_data/aime24.parquet,/workspace/jepa-grpo-cache/eval_data/aime25.parquet,/workspace/jepa-grpo-cache/eval_data/aime26.parquet,/workspace/jepa-grpo-cache/eval_data/amc23.parquet]' \
ENABLE_OVERLONG_BUFFER=true \
NDEVICES_PER_NODE=1 \
MAX_OPTIMIZER_STEPS=666 \
TOTAL_TRAINING_STEPS=666 \
PROJECT_NAME=grpo-qwen-3b \
EXPERIMENT_NAME=grpo-8cot-seed31415-666 \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_grpo-8cot-seed31415-666.jsonl \
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
  +ray_kwargs.ray_init.address=127.0.0.1:36379 2>&1 | tee /workspace/exploration/grad_direction_probe/log_grpo-8cot-seed31415-666.log
