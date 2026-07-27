#!/bin/bash
cd /workspace/Joint-Embedding-Guided-Policy-Optimization
export WANDB_ENTITY=ismamnurswapnil-bangladesh-university-of-engineering-and
export CUDA_VISIBLE_DEVICES=0,1
# HIDDEN-STATE GRPO (loss_type=h-grpo). GRPO's group-relative surrogate with the token
# policy replaced by a group-normalised distribution over hidden-state compatibility.
#   a_i  = sg(normalize(f_theta(x_i, y_i^T)))     frozen correct-solution anchor (CoT)
#   s_ij = a_i . z_ij                             compatibility of rollout j
#   q_i  = softmax_j(s_ij)                        the representation policy
#   A_ij = (c_ij - cbar_i)/(sigma_i + eps)        group-relative advantage from the verifier
#   L    = L_GRPO + alpha * ( -mean_i sum_j A_ij q_ij )
#
# WHY THIS AND NOT THE TRIPLET. q is EXACTLY invariant to s_j -> s_j + gamma, so raising
# every cosine by the same amount is not a descent direction -- it is not a direction in
# the objective at all. That common mode is precisely how llm-jepa-triplet failed (run
# 0ntqn3zc, 170 steps at 3B): its absolute `1 - cos(p+, a)` align term was satisfied by
# dragging EVERYTHING toward the anchor -- cos_correct 0.8525->0.9393 and cos_wrong
# 0.8550->0.9445 rose in lockstep, jepa/margin stayed ~0, jepa/violation pinned at 1.00
# for every one of the 170 steps. There is no absolute attraction term here.
# It also has no trainable readout direction w -- the readout IS the teacher anchor -- so
# it cannot repeat the BCE arm's failure (run ssq4liuv: w registered at lr=1e-6, AdamW
# drift bound 0.64 degrees of rotation over 283 steps, i.e. a frozen random projection).
#
# ONE optimizer step over the total loss (fuse_optimizer_step=True), as in LLM-JEPA:
# theta -> theta - lr*Adam(g_GRPO + alpha*g_HGRPO). Two sequential Adam steps would NOT be
# equivalent -- Adam rescales each gradient by its own second moment, so alpha would stop
# controlling the relative contribution. Preconditions hold: ppo_epochs=1 and
# ppo_mini_batch_size(64) >= train_batch_size(64).
# This needed a real fix: EngineTrainModeCtx.__exit__ zeroed the optimizer unconditionally,
# wiping the deferred policy gradient before jepa_update could fuse onto it (the first
# launch of this script died on _assert_policy_grad_resident norm=0.0). train_mode now takes
# keep_grad, set from apply_step. Regression test: test_fuse_optimizer_step_on_cpu.py.
#
# predictor_k=0 IS REQUIRED. The sibling infoNCE/triplet path appends [PRED] to correct
# rows and [BAD_PRED] to wrong rows; under a group softmax that is LABEL LEAKAGE -- the
# score would separate rollouts by their appended token rather than by their content.
#
# alpha 0.05 (= the spec's beta, the ONLY coefficient this objective adds). The 3B triplet
# arm ran at jepa/grad_norm 0.0004 vs actor/grad_norm 0.10-0.16 -- 0.3% of the policy
# gradient, too weak to move anything. GATE 1 (step 25): jepa/grad_norm/actor/grad_norm ~ 0.2.
# GATE 2 (step 50, THE FALSIFIER): jepa/margin > 0 and rising, jepa/violation < 1.0,
#   jepa/hgrpo_q_correct above its step-0 value. If these stay pinned under an objective
#   that CANNOT reward the common mode, correctness is not separable along the anchor
#   direction in this representation and the line ends.
# GATE 3 (step 50): jepa/cos_cot and jepa/cos_neg must DIVERGE; effective_rank_norm must
#   not fall toward the degenerate 0.018 the 3B arm sat at.
# GATE 4 (step 100): rep/fixed_auroc_cw > 0.7514 (the plain-GRPO level) -- an independent
#   held-out readout the loss cannot game.
# GATE 5 (>=200 steps): val pass@1 slope vs the baseline's +0.0020/100 (run 2pk6qu21).
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
  actor_rollout_ref.model.path=/workspace/models/Qwen2.5-Math-1.5B-Instruct \
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
  jepa.enable=True \
  jepa.n_cot=8 \
  jepa.n_code=0 \
  jepa.alpha=0.05 \
  jepa.ema_decay=0.99 \
  jepa.embed_micro_batch_size=8 \
  jepa.target_max_length=4096 \
  jepa.min_valid_pairs=1 \
  jepa.tcr_reward_beta=0 \
  jepa.loss_type=h-grpo \
  +jepa.fuse_optimizer_step=True \
  jepa.predictor_k=0 \
  jepa.alpha_warmup_steps=20 \
  jepa.max_grad_norm=0.5 \
  jepa.teacher_cache_path=/workspace/jepa-grpo-cache/ds40k.cot.responses.pt \
  jepa.code_teacher_cache_path= \
  jepa.code_system_prompt= \
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
  trainer.experiment_name=hgrpo-a05-Qwen2.5-Math-1.5B-20260726 \
  trainer.default_local_dir=checkpoints/grpo-qwen-3b/hgrpo-a05-Qwen2.5-Math-1.5B-20260726 \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.val_before_train=false \
  trainer.total_training_steps=800 \
  trainer.log_val_generations=0 \
  +trainer.rep_tsne_enable=true \
  +trainer.rep_tsne_mode=train \
  +trainer.rep_tsne_max_prompts=32 \
  +trainer.rep_tsne_freq=10 \
  +trainer.rep_tsne_seed=14142 \
  +trainer.rep_tsne_perplexity=30 \
  +trainer.rep_tsne_micro_batch_size=4 \
  trainer.rollout_data_dir=null \
  trainer.validation_data_dir=null \
  trainer.max_actor_ckpt_to_keep=1 \
  +trainer.val_only=false \
  +trainer.best_ckpt_sources=["aime24","aime25","aime26","amc23"] \
  +trainer.best_ckpt_metrics=["pass@8","pass@1+pass@8"] \
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
