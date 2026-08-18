#!/usr/bin/env bash
# ARM A -- CORRECT-cluster LLM-JEPA, FUSED optimizer step, MARGIN objective.
#
#   L_total = L_GRPO + alpha * L_JEPA-cluster   ->  ONE Adam step at ACTOR_LR.
#
# The JEPA loss (core_algos.llm_jepa_cluster_loss) treats a prompt's CORRECT rollouts
# as views of one another:  L = mean_x mean_{i!=j in C_x} (1 - cos(z_i, z_j)) over
# prompts with |C_x| >= 2. Responses are encoded WITHOUT the prompt, predictor_k=0
# (the paper's identity read), no stop-grad, no SIGReg, no negatives.
#
# Fused is the sound way to keep a SINGLE optimizer: alpha scales the raw gradient
# before the one Adam step, so it genuinely controls the relative contribution, and
# Adam's m/v buffers are updated once per iteration. (Two sequential Adam steps on the
# same optimizer do NOT preserve either property -- alpha stops controlling the relative
# contribution and the shared m/v is driven alternately by two differently scaled
# gradients; see jepa.fuse_optimizer_step. The decoupled arm that measured this,
# cluster-jepafirst-20260729, is retired: it collapsed. Set jepa.update_before_policy
# to revive it.)
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# GRPO controls copied from run_qwen25_math_1_5b_fsdp.sh.
# JEPA fusion is the only intended difference from that reference.
export TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet}
export MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-666}
export TRAIN_SEED=${TRAIN_SEED:-14142}
export NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-False}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
export CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
export LOGPROB_MAX_TOKEN_LEN_PER_GPU=${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-98304}
export SAVE_FREQ=${SAVE_FREQ:-10}
export BEST_CKPT_METRICS=${BEST_CKPT_METRICS:-'[pass@8,pass@1+pass@8]'}
export VALIDATION_SEEDS=${VALIDATION_SEEDS:-'[31415,271828,14142,16180,299792]'}
# --- the arm -------------------------------------------------------------------
export JEPA_LOSS_TYPE=llm-jepa-cluster
export JEPA_FUSE_OPTIMIZER_STEP=True
export JEPA_UPDATE_BEFORE_POLICY=False
# alpha=0.005 made the JEPA term INVISIBLE: on cluster-fused-20260729
# jepa/grad_norm_total matched actor/grad_norm to 3 decimals (0.107 vs 0.107), i.e. the
# aux gradient was <0.5% of the update and the arm was a slow GRPO replica. Raised 20x.
# MEASURED at alpha=0.1: grad_norm_total 0.105 vs actor/grad_norm 0.0878 -> JEPA is 30% of
# total gradient energy. Healthy auxiliary weight; do not raise further without re-measuring.
export ALPHA=${ALPHA:-0.1}                   # meaningful here: fused, pre-Adam scaling
export JEPA_MIN_CORRECT_FOR_CLUSTER=2
# MARGIN objective, not attraction-only. Attraction-only has inf L = 0 at any constant
# encoder (verified: constant encoder scores 0.000 vs 0.135 for a good representation),
# and cluster-jepafirst-20260729 was descending into it -- margin 0.107->0.063,
# effective_rank 4.73->3.43 over 26 steps. repel=1 makes the objective
# cos_all_pairs - cos_correct_correct, which collapse cannot lower (both -> 1 => 0).
export JEPA_CLUSTER_REPEL=${JEPA_CLUSTER_REPEL:-1.0}
# MUST be 'correct'. With 'all', _build_jepa_batch_cluster selects every rollout, so a
# group's "correct-correct" cosine is really over all 8 responses and the objective
# degenerates to prompt-identity clustering with NO correctness signal (observed:
# pool_size=512=64x8, group_size_mean=8, all 64 groups eligible, margin flat at 0.07).
export JEPA_ANCHOR_SET=${JEPA_ANCHOR_SET:-correct}

# --- required by this loss type ------------------------------------------------
export LLM_JEPA_PREDICTOR_K=0                # identity predictor Pred(x)=x
export N_CODE=0                              # CoT rollouts only ...
export N_COT=8                               # ... n_cot + n_code must equal ROLLOUT_N
export JEPA_REWARD_BETA=0                    # no teacher-cache reward shaping
export AUTO_OFF_ENABLE=False                 # latches on cos_cot, which this loss never emits

# --- fused preconditions (engine zeroes grad each mini-batch) -------------------
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}     # must be >= TRAIN_BATCH_SIZE

# --- matched to the GRPO baseline so the benchmarks are comparable -------------
export ACTOR_LR=${ACTOR_LR:-1e-6}
export USE_LORA=${USE_LORA:-false}           # baseline is full-parameter (LORA_RANK=0)
export LORA_RANK=0
export ROLLOUT_N=${ROLLOUT_N:-8}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}     # baseline reports pass@8
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}

# Same wandb project as the GRPO baseline, so both arms and the baseline land
# on one set of charts.
export PROJECT_NAME=${PROJECT_NAME:-grpo-qwen-3b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cluster-margin-fused-20260729}

exec bash examples/jepa_grpo_trainer/run_qwen_1_5b_dapo_ray.sh "$@"
