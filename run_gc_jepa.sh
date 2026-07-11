#!/bin/bash
# LoRA512, gradient checkpointing ON, Liger ON, JEPA ON, no activation offload
# FRESH run: simplified JEPA — 8 CoT rollouts, 0 code rollouts (policy training
# is pure DAPO, directly comparable to baseline 0daxtn9n). JEPA aux loss keeps
# align_cot + a cot->code-teacher pull (CoT anchors paired with the code teacher
# cache); student-code alignment and cos_self are gone with the code view, and
# TCR reward shaping is disabled (beta=0, Step 2.5 skipped entirely).
# Policy and JEPA gradients are now fused into ONE optimizer step per training
# step (policy update defers its step, JEPA backward accumulates on top, then
# issues the single combined step) instead of two independent full-strength
# steps on the shared LoRA weights.
cd /workspace/exploration
TS=$(date +%Y%m%d_%H%M%S)
NDEVICES_PER_NODE=2 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 MODEL_PATH=/workspace/models/Qwen2.5-3B-Instruct MODEL_NAME=Qwen2.5-3B-Instruct \
JEPA_ENABLE=True N_COT=8 N_CODE=0 JEPA_REWARD_BETA=0 SELF_CONSIST_W=0 ALPHA_WARMUP_STEPS=20 \
LORA_RANK=512 LORA_ALPHA=1024 ENABLE_GRAD_CKPT=True \
PPO_MINI_BATCH_SIZE=64 AUTO_OFF_PATIENCE=10 \
VALIDATION_SEEDS='[31415,27182,16180]' EMBED_MICRO_BATCH_SIZE=32 PPO_MAX_TOKEN_LEN_PER_GPU=16384 VAL_BEFORE_TRAIN=false SAVE_FREQ=10 \
PROJECT_NAME=dapo_deepseek_1.5B EXPERIMENT_NAME=dapo_jepa-cotonly-Qwen2.5-3B-Instruct-${TS} \
WANDB_ENTITY=ismamnurswapnil-bangladesh-university-of-engineering-and \
bash examples/jepa_grpo_trainer/run_qwen_3b_dapo_ray.sh \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=98304 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=98304 \
  trainer.val_before_train=false \
  actor_rollout_ref.model.use_liger=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8 \
  +actor_rollout_ref.actor.fuse_old_log_prob=True \
  2>&1 | tee dapo_jepa_cotonly_${TS}.log
