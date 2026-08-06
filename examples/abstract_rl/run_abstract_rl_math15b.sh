#!/usr/bin/env bash
# Abstract-reasoning compression GRPO on Qwen2.5-Math-1.5B-Instruct.
#
# Config is deliberately identical to the reference GRPO run
# `tafr-repro-math15b/tafr-sweep-repro-b10-20260805` (same dataset, batch sizes,
# lr, rollout.n, seeds, val parquets, greedy val, best-ckpt metric, wandb project),
# so the arms below are comparable to it and to each other.
#
# max_response_length stays at 3072: Qwen2.5-Math-1.5B has max_position_embeddings
# 4096, and 1024 + 3072 already fills it exactly. There is no headroom to give the
# abstraction -- it has to fit inside the existing budget, and YaRN is known to
# break this model (3x pass@1 drop), so extending the context is not an option.
#
#   ARM=grpo        L_GRPO(r)                       -- ablation 1, mode disabled
#   ARM=utility     L_GRPO(U)                       -- ablation 2, no compression
#   ARM=compression L_GRPO(r) + lambda*L_comp       -- ablation 3, no usefulness
#   ARM=full        L_GRPO(U) + lambda*L_comp       -- the method
#   ARM=smoke       tiny 2-step run that dumps parsed examples
#
# MEASURED, READ BEFORE LAUNCHING: Qwen2.5-Math-1.5B-Instruct emits <abstract> in
# 0 of 416 sampled rollouts across 6 prompt placements (appended, system-only,
# no-system, imperative, schematic example, and a full 1-shot assistant turn). It
# always stops at <|im_end|> right after \boxed{}. With no valid abstraction every
# U_i = 0, every group has zero advantage, and training is a no-op -- the smoke run
# shows exactly that (valid_abstract_rate=0, zero_advantage_group_fraction=1).
# Qwen2.5-1.5B-Instruct (general, same size) does comply: 1% valid with
# INSTRUCTION=spec, 7% with INSTRUCTION=strong, which is enough for the group-relative
# advantage to bootstrap the format. Use MODEL= to pick, at the cost of comparability
# with the Math-1.5B reference run.
#
# Usage:  ARM=full GPU=0 examples/abstract_rl/run_abstract_rl_math15b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

ARM=${ARM:-full}
GPU=${GPU:-0}
KL_COEF=${KL_COEF:-0.05}
KL_CLIP=${KL_CLIP:-2.0}   # per-token cap on the compression KL (0 = k3's own +-10 clamp only)
MAX_RESPONSE=${MAX_RESPONSE:-3072}
TOTAL_STEPS=${TOTAL_STEPS:-1000}
TRAIN_BS=${TRAIN_BS:-64}
MINI_BS=${MINI_BS:-16}
ROLLOUT_N=${ROLLOUT_N:-8}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
DEBUG_PRINT=${DEBUG_PRINT:-0}
MODEL=${MODEL:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
INSTRUCTION=${INSTRUCTION:-spec}
BOOTSTRAP=${BOOTSTRAP:-0}          # graft abstractions for the first N steps (0 = off)
GRAFT_FRACTION=${GRAFT_FRACTION:-0.5}
SFT_WARMUP=${SFT_WARMUP:-0}       # first N steps: SFT the abstraction span instead of GRPO
SFT_LR=${SFT_LR:-0}               # lr during the warm-up only (0 = keep the RL lr)
SFT_CLIP=${SFT_CLIP:-0}           # grad-norm clip during the warm-up only (0 = keep the RL clip)
SFT_ACC_DROP=${SFT_ACC_DROP:-0.10} # abort the warm-up if train acc falls this far (0 = no guard)
SFT_MIN_LEN=${SFT_MIN_LEN:-0.6}    # abort if abstract length falls below this fraction of baseline
ABSTRACT_CACHE=${ABSTRACT_CACHE:-} # frozen question->abstract parquet (pregen_abstracts.py); empty = online phase-2
# TWO_PASS=1: A is always a genuine second vLLM generation from (X, Y), never a
# same-turn continuation. Y is solved with no abstract instruction at all (so its
# sampling, and r_i, are untouched by anything abstract-related -- see the
# scoring.py / config.py notes on why trailing order matters for Delta). This
# reuses bootstrap.py's graft machinery as the PRIMARY path rather than a warm-up:
# BOOTSTRAP covers the whole run and GRAFT_FRACTION=1 (every row, not a held-back
# fraction -- that fraction existed to teach single-pass emission, which is not
# what two-pass is doing).
TWO_PASS=${TWO_PASS:-0}
if [[ "$TWO_PASS" == "1" ]]; then
  # BOOTSTRAP already defaulted to "0" above (a non-empty string), so ${BOOTSTRAP:-...}
  # would never trigger here -- check the sentinel value explicitly instead.
  [[ "$BOOTSTRAP" == "0" ]] && BOOTSTRAP=$TOTAL_STEPS
  GRAFT_FRACTION=1.0
fi
LOGGER=${LOGGER:-'["console","wandb"]'}
PROJECT=${PROJECT:-tafr-repro-math15b}
EXPERIMENT=${EXPERIMENT:-abstractrl-${ARM}-$(date +%Y%m%d)}

DATA=/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet
EVAL=/workspace/jepa-grpo-cache/eval_data
VAL="[${EVAL}/math500.parquet,${EVAL}/amc23.parquet,${EVAL}/aime24.parquet,${EVAL}/aime25.parquet,${EVAL}/aime26.parquet]"

case "$ARM" in
  grpo)        ABSTRACT=(custom_abstract_rl.enable=false) ;;
  utility)     ABSTRACT=(custom_abstract_rl.enable=true custom_abstract_rl.use_utility=true  custom_abstract_rl.kl_coef=0) ;;
  compression) ABSTRACT=(custom_abstract_rl.enable=true custom_abstract_rl.use_utility=false custom_abstract_rl.kl_coef="${KL_COEF}") ;;
  full)        ABSTRACT=(custom_abstract_rl.enable=true custom_abstract_rl.use_utility=true  custom_abstract_rl.kl_coef="${KL_COEF}") ;;
  smoke)
    ABSTRACT=(custom_abstract_rl.enable=true custom_abstract_rl.use_utility=true custom_abstract_rl.kl_coef="${KL_COEF}")
    TRAIN_BS=8; MINI_BS=8; ROLLOUT_N=4; TOTAL_STEPS=2; TEST_FREQ=0; SAVE_FREQ=0
    DEBUG_PRINT=3; LOGGER='["console"]'; EXPERIMENT=abstractrl-smoke
    # BOOTSTRAP already defaulted to the "0" sentinel string above, so ${BOOTSTRAP:-2}
    # never fires here (same shell pitfall as the TWO_PASS block) -- check explicitly.
    [[ "$BOOTSTRAP" == "0" ]] && BOOTSTRAP=2 ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac

# The verifier must never see the abstraction: a \boxed{} inside <abstract> would
# otherwise become the graded answer (the math parser takes the last boxed span).
REWARD=()
# The instruction must be in the prompt BEFORE maybe_filter_out_long_prompts measures
# it; appending later overflows max_prompt_length on near-ceiling prompts and kills
# the rollout tens of steps in. The plain-GRPO arm gets neither.
DATASET=()
INJECT=()
if [[ "$ARM" != "grpo" ]]; then
  REWARD=(
    reward.custom_reward_function.path=verl/experimental/abstract_rl/reward_fn.py
    reward.custom_reward_function.name=compute_score_ignoring_abstract
  )
  if [[ "$TWO_PASS" == "1" ]]; then
    # No single-pass instruction at all: Y is a plain solve, A only ever comes
    # from the phase-2 graft below. Skips the AbstractRLDataset requirement too,
    # since that requirement only exists to fit the injected instruction before
    # prompt-length filtering runs.
    INJECT=(custom_abstract_rl.inject_instruction=false)
  else
    DATASET=(
      data.custom_cls.path=verl/experimental/abstract_rl/dataset.py
      data.custom_cls.name=AbstractRLDataset
      +data.abstract_rl_instruction="${INSTRUCTION}"
    )
  fi
fi

# The image exports RAY_ADDRESS=127.0.0.1 (no port), which ray.init rejects outright.
[[ "${RAY_ADDRESS:-}" == *:* ]] || unset RAY_ADDRESS

export CUDA_VISIBLE_DEVICES=${GPU}
exec .venv/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA}" \
  data.val_files="${VAL}" \
  data.train_batch_size="${TRAIN_BS}" \
  data.val_batch_size=128 \
  data.dataloader_num_workers=2 \
  data.max_prompt_length=1024 \
  data.max_response_length="${MAX_RESPONSE}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=3.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_loss_coef=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BS}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.data_loader_seed=3407 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_num_seqs=2048 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  critic.enable=false \
  "${ABSTRACT[@]}" \
  custom_abstract_rl.debug_print_n="${DEBUG_PRINT}" \
  custom_abstract_rl.kl_clip="${KL_CLIP}" \
  custom_abstract_rl.instruction_variant="${INSTRUCTION}" \
  custom_abstract_rl.bootstrap_steps="${BOOTSTRAP}" \
  custom_abstract_rl.bootstrap_graft_fraction="${GRAFT_FRACTION}" \
  custom_abstract_rl.bootstrap_sft_steps="${SFT_WARMUP}" \
  custom_abstract_rl.bootstrap_sft_lr="${SFT_LR}" \
  custom_abstract_rl.bootstrap_sft_clip_grad="${SFT_CLIP}" \
  custom_abstract_rl.bootstrap_sft_accuracy_drop="${SFT_ACC_DROP}" \
  custom_abstract_rl.bootstrap_sft_min_length_fraction="${SFT_MIN_LEN}" \
  custom_abstract_rl.bootstrap_abstract_cache="${ABSTRACT_CACHE}" \
  "${REWARD[@]}" \
  "${DATASET[@]}" \
  "${INJECT[@]}" \
  trainer.resume_mode=auto \
  trainer.balance_batch=True \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT}" \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.default_local_dir="checkpoints/${PROJECT}/${EXPERIMENT}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train=false \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.log_val_generations=0 \
  +trainer.best_ckpt_sources='["math500","amc23","aime24","aime25","aime26"]' \
  +trainer.best_ckpt_metrics='["pass@1"]' \
  +trainer.validation_seeds='[3407]' \
  "$@" 2>&1
