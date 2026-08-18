#!/bin/bash
cd /workspace/Joint-Embedding-Guided-Policy-Optimization
export WANDB_ENTITY=ismamnurswapnil-bangladesh-university-of-engineering-and
export CUDA_VISIBLE_DEVICES=0
# PLAIN GRPO on Qwen2.5-1.5B-Instruct (the GENERAL instruct model, not the Math variant).
# jepa.enable=False -> no auxiliary loss of any kind. This is a clean GRPO reference.
#
# VALIDATION IS DISABLED IN-LOOP (test_freq=-1). Instead, checkpoints are written every 10
# steps with hf_model weights, and a companion watcher on GPU1 scores them asynchronously:
#   scripts/eval_watcher.py  (tmux session `evalwatch`)
# Measured on the h-grpo run: a train step is 54.7s but an inline validation step is 330.6s
# -- 6x -- so at test_freq=10 validation was ~38% of wall-clock AND serial. Moving it to the
# second GPU removes it from the critical path entirely and puts the idle GPU to work.
#
# NOT comparable to baseline 2pk6qu21 or any other curve in this project: those all used
# Qwen2.5-Math-1.5B-Instruct. This IS comparable to the SFT / GXPO-SFT arms, which fine-tuned
# this same base -- and its step-0 score is the untrained baseline those arms never measured.
#
# save_contents includes hf_model so the watcher can load each checkpoint straight into vLLM
# with no merger step. max_actor_ckpt_to_keep=2 bounds disk (~22G each) while leaving the
# watcher ~18 min of headroom before a checkpoint rotates away.
exec /workspace/exploration/.venv/bin/python3 -m verl.experimental.jepa_grpo.main_ray \
  --config-name \
  jepa_grpo_ray_qwen25_1_5b \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  data.train_files=/workspace/jepa-grpo-cache/data/deepscaler_preview_train.parquet \
  data.val_files=[/workspace/jepa-grpo-cache/eval_data/aime24.parquet,/workspace/jepa-grpo-cache/eval_data/aime25.parquet,/workspace/jepa-grpo-cache/eval_data/aime26.parquet,/workspace/jepa-grpo-cache/eval_data/amc23.parquet] \
  data.train_batch_size=64 \
  +data.gen_batch_size=64 \
  data.val_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=/workspace/models/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=3.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.data_loader_seed=14142 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
  jepa.enable=False \
  actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
  jepa.n_cot=8 \
  jepa.n_code=0 \
  jepa.alpha=0.0015 \
  jepa.ema_decay=0.99 \
  jepa.embed_micro_batch_size=8 \
  jepa.target_max_length=4096 \
  jepa.min_valid_pairs=1 \
  jepa.loss_type=llm-jepa-triplet \
  +jepa.triplet_tau=0.1 \
  +jepa.triplet_margin=0.1 \
  +jepa.triplet_repel_weight=1.0 \
  +jepa.triplet_include_cot=false \
  jepa.triplet_sigreg_lambda=0.0 \
  jepa.predictor_k=0 \
  jepa.alpha_warmup_steps=20 \
  jepa.max_grad_norm=0.5 \
  jepa.teacher_cache_path=/workspace/jepa-grpo-cache/ds40k.cot.responses.pt \
  jepa.code_teacher_cache_path=/workspace/jepa-grpo-cache/ds40k.code.responses.pt \
  jepa.code_system_prompt= \
  jepa.tcr_match=cycle \
  jepa.max_anchors_per_prompt=2 \
  jepa.jepa_anchor_set=correct \
  jepa.auto_off_enable=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.reward_manager.name=dapo \
  +reward.reward_kwargs.overlong_buffer_cfg.enable=false \
  +reward.reward_kwargs.overlong_buffer_cfg.len=614 \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward.reward_kwargs.max_resp_len=3072 \
  trainer.balance_batch=True \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=grpo-qwen-3b \
  trainer.experiment_name=grpo-only-Qwen2.5-1.5B-Instruct-20260726 \
  trainer.default_local_dir=checkpoints/grpo-qwen-3b/grpo-only-Qwen2.5-1.5B-Instruct-20260726 \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.total_training_steps=800 \
  trainer.log_val_generations=0 \
  +trainer.rep_tsne_enable=false \
  +trainer.rep_tsne_mode=train \
  +trainer.rep_tsne_max_prompts=32 \
  +trainer.rep_tsne_freq=10 \
  +trainer.rep_tsne_seed=14142 \
  +trainer.rep_tsne_perplexity=30 \
  +trainer.rep_tsne_micro_batch_size=4 \
  trainer.rollout_data_dir=null \
  trainer.validation_data_dir=null \
  trainer.max_actor_ckpt_to_keep=2 \
  +trainer.val_only=false \
  +trainer.validation_seeds=[31415,27182,16180] \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=98304 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=98304 \
  actor_rollout_ref.model.use_liger=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto \
  +actor_rollout_ref.actor.fuse_old_log_prob=True \
  2>&1
