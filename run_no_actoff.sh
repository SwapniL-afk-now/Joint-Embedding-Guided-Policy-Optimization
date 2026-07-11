#!/bin/bash
# DAPO-only LoRA512, no grad-ckpt, Liger, NO activation offload (activation offload crashes with LoRA)
cd /workspace/exploration
NDEVICES_PER_NODE=2 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 MODEL_PATH=/workspace/models/Qwen2.5-3B-Instruct \
JEPA_ENABLE=False N_COT=8 N_CODE=0 LORA_RANK=512 LORA_ALPHA=1024 ENABLE_GRAD_CKPT=False \
VALIDATION_SEEDS='[31415]' EMBED_MICRO_BATCH_SIZE=40 PPO_MAX_TOKEN_LEN_PER_GPU=16384 VAL_BEFORE_TRAIN=false \
bash examples/jepa_grpo_trainer/run_qwen_3b_dapo_ray.sh \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
  trainer.val_before_train=false \
  actor_rollout_ref.model.use_liger=True \
  2>&1 | tee dapo_only_lora512_no_gc_liger_16384.log
