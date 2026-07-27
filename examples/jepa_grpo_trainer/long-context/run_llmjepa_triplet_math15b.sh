#!/bin/bash
cd /workspace/Joint-Embedding-Guided-Policy-Optimization
export WANDB_ENTITY=ismamnurswapnil-bangladesh-university-of-engineering-and
export CUDA_VISIBLE_DEVICES=1
# CORRECTNESS-CONDITIONED REPRESENTATION GRADIENT (llm-jepa-triplet + frozen anchor).
#   a_i = sg(f_theta(y^T_cot))                      FROZEN correct-solution anchor
#   L   = align_w * (1 - cos(p+, a)) + repel_w * tau*softplus((cos(p-,a) - cos(p+,a) + m)/tau)
# p+ = correct student read, p- = wrong student read, both response-only.
#
# THREE changes from the failed code-only arm (run 0ntqn3zc, 170 steps at 3B):
#  1. triplet_detach_teacher=true. Previously the anchor was differentiable, so the
#     cheapest descent was to drag the ANCHOR into the student cloud: cos_correct
#     0.8525->0.9393 and cos_wrong 0.8550->0.9445 rose in LOCKSTEP, jepa/margin stayed
#     ~0, jepa/violation pinned at 1.00 for all 170 steps. A frozen anchor removes that.
#  2. alpha 0.0015 -> 0.05. That run had jepa/grad_norm 0.0004 vs actor/grad_norm
#     0.10-0.16 -- the representation term was 0.3% of the policy gradient, far too weak
#     to move anything. Check jepa/grad_norm / actor/grad_norm ~ 0.2 at step 25.
#  3. triplet_include_cot=true, n_code=0: anchor is the teacher CoT, not the Code view.
#     The old arm asked a CoT read to approach a CODE read, across the modality gap that
#     dominates this representation -- so modality, not correctness, was the axis being
#     optimised. A same-modality anchor leaves correctness as the only usable axis.
#     Cost: the align arm starts near-saturated (cos~0.93). That is fine -- align is not
#     the discriminative term; the hinge on the DIFFERENCE is, and 1/tau amplifies it.
# margin=0 makes 1 - jepa/violation exactly nearest-anchor retrieval accuracy.
# SIGReg off: uniform contraction toward a FROZEN anchor drives the margin to 0 but can
# never make it positive (tau*log2 floor), so there is no degenerate optimum to guard.
# FALSIFIER: jepa/margin must rise above 0 and violation fall below 1.0 by ~step 50.
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
  jepa.loss_type=llm-jepa-triplet \
  +jepa.triplet_tau=0.1 \
  +jepa.triplet_margin=0.0 \
  +jepa.triplet_repel_weight=1.0 \
  +jepa.triplet_include_cot=true \
  +jepa.triplet_detach_teacher=true \
  jepa.triplet_sigreg_lambda=0.0 \
  jepa.predictor_k=0 \
  jepa.alpha_warmup_steps=20 \
  jepa.max_grad_norm=0.5 \
  jepa.teacher_cache_path=/workspace/jepa-grpo-cache/ds40k.cot.responses.pt \
  jepa.code_teacher_cache_path= \
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
  trainer.experiment_name=triplet-sg-cotanchor-a05-Qwen2.5-Math-1.5B-20260726 \
  trainer.default_local_dir=checkpoints/grpo-qwen-3b/triplet-sg-cotanchor-a05-Qwen2.5-Math-1.5B-20260726 \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=1 \
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
