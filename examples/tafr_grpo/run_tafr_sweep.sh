#!/usr/bin/env bash
# TAFR ablation sweep — wraps run_tafr_grpo_fsdp.sh, one arm per invocation.
#
#   bash examples/tafr_grpo/run_tafr_sweep.sh beta1.0
#   ARM_GPU=1 bash examples/tafr_grpo/run_tafr_sweep.sh anchor
#   bash examples/tafr_grpo/run_tafr_sweep.sh all          # sequential, in priority order
#
# Baseline for every arm is the prod run tafr-prod-b05 (wandb 5j6rgx1l):
# beta 0.5, sft_interval 5, checkpoint/EMA interval 10, ema_gamma 0.9, mix_eta 0.5,
# unbounded failure buffer, failure_sft_lr 1e-6 (the *config* default — note the
# launcher's own TAFR_SFT_LR default is 1e-5 and the prod run did not use it).
# Each arm changes exactly one thing against that baseline, except `anchor`, where
# the two knobs are deliberately coupled (see below).
set -euo pipefail

ARM=${1:-}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# --- shared across every arm: identical to the prod run so arms are comparable ---
export PROJECT_NAME=${PROJECT_NAME:-grpo-qwen-3b}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-50}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}   # keep RL arms on 1 GPU: the DP gradient deficit
export TRAIN_SEED=${TRAIN_SEED:-31415}
export CUDA_VISIBLE_DEVICES=${ARM_GPU:-0}

# prod-run baseline values; each arm overrides only what it is testing
export TAFR_LOSS_VERSION=failure_token_adv
export TAFR_VARIANT=full          # full | anchor_only | replay_only (replay_only = anchor KL off)
export TAFR_BETA=0.5
# Must be exported, not just assigned: run_tafr_grpo_fsdp.sh is a child process and only
# sees the environment. An arm that assigns these without them being exported here silently
# runs the launcher's defaults (p99 / 1.0 / 1.0) instead.
export TAFR_ADV_NORM=p99          # p99 | dense_token_credit
export TAFR_LAMBDA_A=1.0          # -> anchor_weight  (dense_token_credit only)
export TAFR_LAMBDA_F=1.0          # -> failure_weight (dense_token_credit only; only the a:f ratio matters)
export TAFR_DTC_DOSE=0.5          # rms(A_dense) = dose * that rollout's own |A_grpo|
export TAFR_SFT_UPDATE_INTERVAL=5
export TAFR_CHECKPOINT_INTERVAL=10
export TAFR_EMA_GAMMA=0.9
export TAFR_MIX_ETA=0.5
export TAFR_FAILURE_DATA_MAX_SIZE=null
export TAFR_FAILURE_DATA_SAMPLING=recent
export TAFR_SFT_LR=1.0e-6

TAG=$(date -u +%Y%m%d)

run_arm() {
  local name=$1 desc=$2; shift 2   # remaining args are extra hydra overrides
  export EXPERIMENT_NAME="tafr-sweep-${name}-${TAG}"
  # Fresh dir per arm. The launcher hardcodes trainer.resume_mode=auto, so a shared
  # or reused CKPTS_DIR would silently resume someone else's checkpoint.
  export CKPTS_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
  if [[ -d "${CKPTS_DIR}" && "${ALLOW_RESUME:-0}" != "1" ]]; then
    echo "REFUSING: ${CKPTS_DIR} exists — resume_mode=auto would continue it, not start fresh." >&2
    echo "Delete it or set a different TAG if you meant to rerun this arm." >&2
    echo "Set ALLOW_RESUME=1 to deliberately resume from it (e.g. after a crash)." >&2
    exit 3
  fi
  echo "=== ${EXPERIMENT_NAME} on GPU ${CUDA_VISIBLE_DEVICES} — ${desc}"
  echo "    beta=${TAFR_BETA} sft_interval=${TAFR_SFT_UPDATE_INTERVAL} ckpt_interval=${TAFR_CHECKPOINT_INTERVAL}" \
       "ema_gamma=${TAFR_EMA_GAMMA} mix_eta=${TAFR_MIX_ETA} buf=${TAFR_FAILURE_DATA_MAX_SIZE}/${TAFR_FAILURE_DATA_SAMPLING} sft_lr=${TAFR_SFT_LR}"
  [[ $# -gt 0 ]] && echo "    extra: $*"
  bash examples/tafr_grpo/run_tafr_grpo_fsdp.sh "$@" 2>&1 | tee -a "logs_${EXPERIMENT_NAME}.log"
}

case "${ARM}" in

  # 1. Is the regularizer under-dosed? beta is a target regularizer/GRPO advantage
  #    RMS ratio and the prod run hits it precisely (0.112, flat) with no sign of a
  #    stability limit. Read collapse rate and val pass@1.
  # beta is a plain multiplier, not a realized ratio: reg/GRPO rms = beta * 0.226.
  # The +-5.0 clip is applied to A_reg *before* beta (tafr_loss.py:131), so raising beta
  # never triggers it — clip_fraction is 3.9e-5 and ppo_ratio_clip_fraction is exactly 0.
  # beta=2.0 puts A_reg's p99 at 85% of a GRPO token: influential, still subordinate.
  beta1.0) TAFR_BETA=1.0; run_arm b1.0 "beta 0.5 -> 1.0 (p99 21% -> 43% of a GRPO token)" ;;
  beta2.0) TAFR_BETA=2.0; run_arm b2.0 "beta 0.5 -> 2.0 (p99 -> 85% of a GRPO token)" ;;

  # 2+3. Coupled on purpose. The failure model is refreshed by the SFT update and the
  #    anchor EMA is refreshed on the checkpoint interval, so slowing SFT to 10 while
  #    leaving the EMA at its old mix means the two halves of the anchor drift apart.
  #    Both knobs must move together for the arm to test one hypothesis: "a slower,
  #    less policy-chasing failure model, weighted harder toward the moving EMA."
  anchor)
    TAFR_SFT_UPDATE_INTERVAL=10
    TAFR_CHECKPOINT_INTERVAL=10   # aligned so SFT and EMA refresh land on the same step
    TAFR_MIX_ETA=0.8              # weight the moving EMA over the fixed reference
    run_arm anchor "sft_interval 5->10, ckpt/EMA interval aligned at 10, mix_eta 0.5->0.8"
    ;;

  # 4. The failure buffer is unbounded and sampled 'recent', so failures the policy
  #    has since fixed still train the failure model. Cap it so the model describes
  #    current failure modes.
  buffer)
    TAFR_FAILURE_DATA_MAX_SIZE=20000
    run_arm buf20k "failure buffer unlimited -> 20k (sampling stays 'recent')"
    ;;

  # 5. Lowest priority. The prod run ran the failure-SFT optimizer at 1e-6 (verified
  #    from its step-405 optimizer state), not the launcher's 1e-5 default — which is
  #    the likeliest reason failure_sft_loss never descends (flat 0.774).
  sftlr)
    TAFR_SFT_LR=1.0e-5
    run_arm sftlr1e5 "failure_sft_lr 1e-6 -> 1e-5"
    ;;

  # Coverage arm. The binding constraint is not magnitude but that >=50% of tokens sit at
  # |A_reg| ~1e-4: anchor, failure model and policy agree there, and 0 x anything is still 0.
  # So: make the anchor a purely-trained model that lags far behind (eta=1 drops the base,
  # interval 20 doubles the lag to 200 steps), push the failure model further from the policy
  # (sft_lr 1e-5), and raise the influence on the tokens that do fire (beta 2.0).
  coverage|coverage-shuffle)
    TAFR_MIX_ETA=1.0            # theta_anchor = theta_ema, no fixed-base component
    TAFR_EMA_GAMMA=0.9          # unchanged: higher gamma re-contaminates with the EMA seed
    TAFR_CHECKPOINT_INTERVAL=20 # lag = interval/(1-gamma) = 200 steps, was 100
    TAFR_SFT_UPDATE_INTERVAL=20 # keep SFT and EMA refresh on the same clock
    TAFR_SFT_LR=1.0e-5
    # beta deliberately left at the prod 0.5: this arm tests COVERAGE only, so it stays
    # comparable to the prod run on per-token magnitude. beta is the `beta2.0` arm.
    if [[ "${ARM}" == "coverage-shuffle" ]]; then
      # Negative control: identical config, A_reg shuffled within each response
      # (tafr_loss.py:110). Same magnitude distribution, token alignment destroyed.
      # If this matches `coverage`, the per-token attribution contributes nothing and
      # only the magnitude of the perturbation matters.
      run_arm coverage-shuffle "coverage config + tokens shuffled (negative control)" \
        custom_tafr_grpo.shuffle_advantage_tokens=true
    else
      run_arm coverage "pure-EMA anchor (eta=1), lag 200, sft_lr 1e-5 — coverage only, beta stays 0.5"
    fi
    ;;

  # Loss-form arm. Every failure_token_adv dose underperforms plain GRPO monotonically
  # (beta 0 -> +0.106 train/acc over 400 steps, 0.5 -> +0.032, 1.0 -> +0.000, 2.0 -> -0.007),
  # so this is not a dosing problem. A_reg-as-advantage is rescaled so its p99 matches GRPO's
  # advantage RMS (tafr_loss.py:117), which never saturates and survives any small beta.
  # k3_legacy applies the same anchor as a KL penalty instead: minimized at pi_theta =
  # pi_anchor, gradient vanishing on approach. beta is NOT comparable across the two -- here
  # it multiplies a raw KL, there a GRPO-matched advantage. 0.1 matches the good run xdl0kb1c.
  k3)
    TAFR_LOSS_VERSION=k3_legacy
    TAFR_BETA=0.1
    run_arm k3b01 "loss_version failure_token_adv -> k3_legacy, beta 0.5 -> 0.1"
    ;;

  # Same as k3 but with the anchor KL D_KL(pi_theta || pi_anchor) off, leaving only the
  # gated failure-repulsion term. variant=replay_only zeroes the anchor branch in
  # compute_tafr_grpo_auxiliary_loss (tafr_loss.py:286). Isolates which half does the work.
  k3-noanchor)
    TAFR_LOSS_VERSION=k3_legacy
    TAFR_VARIANT=replay_only
    TAFR_BETA=0.1
    run_arm k3b01-noanchor "k3_legacy beta 0.1, anchor KL off (variant=replay_only)"
    ;;

  # Advantage-based TAFR with the Dense Token Credit normalization -- NOT the k3 KL loss.
  # Replaces the p99/batch-RMS scaling (which is not zero-mean per sequence, so it shifts
  # the sequence-level GRPO preference, and which uses one global scalar that cannot be
  # subordinate to both a +1.75 and a -0.25 rollout in the same group) with residuals that
  # are zero-mean within each response and scaled by that rollout's own |A_GRPO|.
  # TAFR_BETA MUST be 1.0: lambda_a/lambda_f are applied inside the module and
  # losses.py:433 multiplies by beta on top, so beta=0.5 would silently halve the dose.
  dtc)
    TAFR_LOSS_VERSION=failure_token_adv
    TAFR_ADV_NORM=dense_token_credit
    TAFR_BETA=1.0
    TAFR_LAMBDA_A=0.5
    TAFR_LAMBDA_F=0.5
    TAFR_DTC_DOSE=0.5
    run_arm dtc-d05 "dense token credit, dose=0.5 (50% of each rollout's own |A_GRPO|), 50/50 anchor:failure mix"
    ;;

  # Pure potential-based reward shaping (Ng, Harada & Russell 1999). Coefficient 1:
  #   A_total[t] = A_grpo[i] + (log pi_anchor - log pi_old) + c_x[i]*(log pi_old - log pi_failure)
  # Zero free parameters -- no dose, no sigma, no tau, no eps_floor, no zero-mean.
  # beta MUST be 1.0 or the coefficient is no longer 1 and the telescoping breaks.
  pbrs)
    TAFR_LOSS_VERSION=failure_token_adv
    TAFR_ADV_NORM=pbrs_pure
    TAFR_BETA=1.0
    run_arm pbrs "pure PBRS, coefficient 1, zero free parameters"
    ;;

  # Reproduction of wandb run xdl0kb1c (project verl_drgrpo_dapo_math), the only TAFR
  # run measured to BEAT its GRPO baseline: +0.0154 macro avg_at_k over 38 matched evals
  # (t=12.3) against aof7yymb. Everything in the current sweep is flat or negative.
  #
  # Carried over from that run: k3_legacy loss, beta 0.1, variant full (anchor KL live --
  # its anchor_beta=0 was a dead config key, never read), mix_eta 0.5, ema_gamma 0.9,
  # ppo_mini_batch 16 (4 optimizer steps per rollout batch, not 1), failure_sft_lr 5e-7.
  #
  # Deliberately NOT carried over: that run was LoRA r=128 on the non-Math base model,
  # which is what let it use the vLLM log-prob scorer. Full fine-tuning on the Math model
  # forces logprob_backend=hf, so A_reg comes from HF forwards instead of rollout log-probs.
  # Also changed: lr 1e-6 (was 5e-7), 1000 steps (was 400), new train set, greedy eval.
  # `repro-grpo` is the same block with custom_tafr_grpo.enable=false: plain GRPO, every
  # other setting byte-identical, so the pair is a single-variable comparison. Run both
  # (one per GPU) -- the whole reason the current sweep is unreadable is that its control
  # differs from its arms in model, dataset and step count at once.
  repro|repro-grpo|repro-b05|repro-b10|repro-split)
    MODEL_PATH=/workspace/models/Qwen2.5-Math-1.5B-Instruct
    DRGRPO_USE_LORA=false
    TRAIN_FILE=/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet
    VAL_FILES="[${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}/math500.parquet,${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}/amc23.parquet,${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}/aime24.parquet,${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}/aime25.parquet,${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}/aime26.parquet]"
    BEST_CKPT_SOURCES='["math500","amc23","aime24","aime25","aime26"]'
    # Greedy n=1 leaves only pass_at_1, so the trainer's default "combined" metric
    # (0.5*(avg@k+pass@k)) can never resolve and no best/ checkpoint would be written.
    BEST_CKPT_METRICS='["pass@1"]'
    # Its own project: these two are not comparable to anything in grpo-qwen-3b
    # (different model, data, eval protocol and step count).
    PROJECT_NAME=${REPRO_PROJECT_NAME:-tafr-repro-math15b}
    # vLLM headroom. Starting the engine is NOT the binding constraint -- 0.78 starts
    # fine and then dies ~10 minutes in, when sleep mode tries to remap its allocation
    # after the actor update ("CUDA Error: out of memory at cumem_allocator.cpp:137").
    # The TAFR arm also holds the anchor and failure models, which the control does not,
    # so the peak is after the first SFT update, not at startup. 0.7 is what every
    # previous TAFR run used without OOM; treat it as the ceiling, not a default.
    ROLLOUT_GPU_MEM_UTIL=0.7
    # 1024 + 3072 = 4096 exactly, the Math model's stock context. Do NOT raise it:
    # YaRN on this model measured a 3x pass@1 drop and 66% truncation.
    MAX_PROMPT_LENGTH=1024
    MAX_RESPONSE_LENGTH=3072
    ACTOR_LR=1e-6
    PPO_MINI_BATCH_SIZE=16
    # Greedy, single-sample eval: deterministic, so an eval-to-eval move is a real
    # change rather than the sampling noise that made every arm comparison a coin flip.
    # Note this leaves only pass_at_1 -- pass_at_8/avg_at_8 need n>1.
    VAL_ROLLOUT_N=1
    VAL_DO_SAMPLE=False
    VAL_TEMPERATURE=0
    # One seed, not five: _validate() runs a full pass per seed, and greedy decoding
    # makes all five byte-identical. Five seeds here would be 5x the eval cost for
    # zero extra information. Same seed as training, by request.
    VALIDATION_SEEDS='[3407]'
    TRAIN_SEED=3407
    TAFR_LOSS_VERSION=k3_legacy
    TAFR_VARIANT=full
    TAFR_BETA=0.1
    TAFR_EMA_GAMMA=0.9
    TAFR_MIX_ETA=0.5
    TAFR_SFT_UPDATE_INTERVAL=5
    TAFR_CHECKPOINT_INTERVAL=10
    TAFR_SFT_LR=5.0e-7
    # 64 is the cap on rows per SFT step, but TAFR_SFT_MAX_TOKEN_LEN_PER_GPU (defaulted
    # to PPO_MAX_TOKEN_LEN_PER_GPU=24576 by the launcher) takes precedence: records are
    # packed greedily so rows*max_len stays under the budget. That is what keeps a batch
    # of 64 long failures from OOMing next to a resident vLLM engine.
    TAFR_SFT_BATCH_SIZE=64
    export MODEL_PATH DRGRPO_USE_LORA TRAIN_FILE VAL_FILES BEST_CKPT_SOURCES \
           BEST_CKPT_METRICS PROJECT_NAME ROLLOUT_GPU_MEM_UTIL TRAIN_SEED \
           MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH ACTOR_LR PPO_MINI_BATCH_SIZE \
           VAL_ROLLOUT_N VAL_DO_SAMPLE VAL_TEMPERATURE VALIDATION_SEEDS TAFR_SFT_BATCH_SIZE
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS_REPRO:-1000}
    export TOTAL_TRAINING_STEPS
    if [[ "${ARM}" == "repro-grpo" ]]; then
      TAFR_ENABLE=false; export TAFR_ENABLE
      run_arm repro-grpo "control: TAFR disabled, plain GRPO, identical everything else"
    elif [[ "${ARM}" == "repro-b05" ]]; then
      # beta ablation. beta scales the whole k3 term (anchor pull minus replay push);
      # everything else is identical to the repro arm, so beta is the only variable.
      TAFR_BETA=0.5
      run_arm repro-b05 "beta ablation: beta=0.5 (repro baseline is 0.1)"
    elif [[ "${ARM}" == "repro-b10" ]]; then
      TAFR_BETA=1.0
      run_arm repro-b10 "beta ablation: beta=1.0 (repro baseline is 0.1)"
    elif [[ "${ARM}" == "repro-split" ]]; then
      # The region a single beta cannot reach: strong replay push, weak anchor pull.
      # At beta 0.5/1.0 the failure clone never fits (failure_sft_loss stuck ~0.29 vs
      # 0.073 at beta 0.1), so replay is inert and only the anchor cost is paid.
      # Anchor stays at the winning 0.1; only the replay weight moves.
      TAFR_BETA=0.1
      TAFR_BETA_ANCHOR=${TAFR_BETA_ANCHOR:-0.1}
      TAFR_BETA_REPLAY=${TAFR_BETA_REPLAY:-0.5}
      export TAFR_BETA_ANCHOR TAFR_BETA_REPLAY
      run_arm repro-split "beta split: anchor=${TAFR_BETA_ANCHOR}, replay=${TAFR_BETA_REPLAY} (repro baseline is 0.1/0.1)"
    else
      run_arm repro "xdl0kb1c config, full-finetune Math model, dsr+MATH345, greedy eval, 1000 steps"
    fi
    ;;

  all)
    for a in coverage beta2.0 beta1.0 anchor buffer sftlr; do
      bash "${BASH_SOURCE[0]}" "$a"
    done
    ;;

  *)
    echo "usage: $0 {coverage|coverage-shuffle|beta1.0|beta2.0|anchor|buffer|sftlr|k3|k3-noanchor|dtc|pbrs|repro|repro-grpo|repro-b05|repro-b10|repro-split|all}" >&2
    echo "  env: ARM_GPU (default 0), TOTAL_TRAINING_STEPS (200), TRAIN_SEED (31415)" >&2
    exit 2
    ;;
esac
