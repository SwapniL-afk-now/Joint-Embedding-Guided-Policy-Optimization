#!/usr/bin/env bash
# Shared configuration for the FEPO paper experiment program.
# Sourced by every arm script; not runnable on its own.
#
# Everything here is held IDENTICAL across all 22 arms so that each arm's own
# deltas are the only variable. If you need to change a value, change it here --
# changing it in one arm script silently breaks every comparison the paper makes.
#
# Baseline is the tuned repro-split config (examples/tafr_grpo/run_tafr_sweep.sh),
# with one deliberate departure -- 500 steps, not 1000, for compute budget -- and
# otherwise matched to the reference GRPO run (wandb dxv253gl, project grpo-qwen-3b):
# greedy n=1 validation, full train.parquet, pass@1 as the reported/selection metric.
# Consequence: Table 2's mechanism analysis (per-prompt buckets, Recovery@delta) is
# NOT derivable under this protocol -- it needs sampled per-prompt success rate,
# which greedy cannot give. Tables 1/3/4/5/6/7/8's accuracy columns are unaffected.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PROBE_FILE=${PROBE_FILE:-${EVAL_DATA_DIR}/probe256.parquet}

# --- identity -----------------------------------------------------------------
# tafr-repro-math15b: matches the reference GRPO run (wandb dxv253gl) so results
# land in the same wandb project and are directly comparable.
export PROJECT_NAME=${PROJECT_NAME:-tafr-repro-math15b}
export TAG=${TAG:-$(date -u +%Y%m%d)}

# --- model / data -------------------------------------------------------------
export MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
export DRGRPO_USE_LORA=false
# Full train.parquet -- matches the reference GRPO run (dxv253gl) exactly, same data
# as the baseline being compared against. No probe held out: Table 2's mechanism
# analysis needs per-prompt success rate in [0,1], which greedy validation cannot
# produce anyway (see below), so holding data out for it here buys nothing.
export TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet}

# 1024 + 3072 = 4096 exactly, this model's stock context. Do NOT raise it: YaRN on
# Qwen2.5-Math-1.5B measured a 3x pass@1 drop and 66% truncation.
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072

# --- budget -------------------------------------------------------------------
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
export NDEVICES_PER_NODE=1        # keep RL arms on 1 GPU: measured ~3.3x gradient
                                  # deficit on 2-GPU data parallel in this fork
export TRAIN_SEED=3407            # ONE training seed for the whole program, by
                                  # design -- variance is reported over eval seeds
export CUDA_VISIBLE_DEVICES=${ARM_GPU:-0}

# --- optimizer ----------------------------------------------------------------
export ACTOR_LR=1e-6
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_GPU_MEM_UTIL=0.7   # ceiling, not a default: 0.78 starts fine and then
                                  # OOMs ~10min in when vLLM sleep mode remaps its
                                  # allocation after the actor update. FEPO arms also
                                  # hold the anchor and failure models, so the peak is
                                  # after the first SFT update, not at startup.

# --- validation: GREEDY, matching the reference GRPO run (explicit instruction) -----
# Deterministic decode, one sample per prompt. Consequence: no eval-seed variance (all
# seeds return the identical number -- never report a std from this run), and no
# per-prompt mechanism data (Table 2's buckets need sampled p in [0,1]; greedy only
# gives {0,1}). Tables 1/3/4/5/6/7/8's accuracy columns are unaffected.
export VAL_ROLLOUT_N=1
export VAL_DO_SAMPLE=False
export VAL_TEMPERATURE=0
export VAL_TOP_P=0.95
export VALIDATION_SEEDS='[3407]'  # single seed: greedy is deterministic, more seeds
                                  # would just repeat the identical number
export TEST_FREQ=10               # matches the reference GRPO run's cadence
export VAL_BEFORE_TRAIN=false     # matches the reference GRPO run
export VAL_BATCH_SIZE=128

# Exactly the reference GRPO run's val_files -- no probe set.
export VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet]"
export BEST_CKPT_SOURCES='["math500","amc23","aime24","aime25","aime26"]'
# pass@1 is EXACT under greedy n=1 (the single deterministic sample, not a 1-of-8
# coin flip) -- explicitly requested as the reporting metric.
export BEST_CKPT_METRICS='["pass@1"]'

# --- FEPO / TAFR defaults (beta0, gamma0, K0 for the Table 8 sensitivity sweep) -
# k3_legacy is the only loss_version with explicit KL terms, which is what the
# paper's math describes and what the KL columns of Tables 2 and 3 need. The
# failure_token_adv path has no KL at all and cannot fill them.
export TAFR_ENABLE=true
export TAFR_LOSS_VERSION=k3_legacy
export TAFR_VARIANT=full
export TAFR_BETA=0.1
export TAFR_BETA_ANCHOR=1.0       # beta0 centre, by explicit instruction: equal
export TAFR_BETA_REPLAY=1.0       # anchor/replay weight (1.0/1.0), not the earlier
                                  # 1.0/0.5 split. Watch grad_norm closely -- a
                                  # replay-heavier ratio than 1.0/0.5 (0.5/1.0) is
                                  # what diverged after step ~144 in wandb oyul6igu;
                                  # 1.0/1.0 is a different, not-yet-tested point.
export TAFR_EMA_GAMMA=0.9         # gamma0
export TAFR_SFT_UPDATE_INTERVAL=5 # K0
export TAFR_MIX_ETA=0.5
export TAFR_CHECKPOINT_INTERVAL=10
# 1e-5, NOT the repro run's 5.0e-7. Measured on that run: 3716 failure-clone
# updates left kl_replay at 4e-4, i.e. pi_failure never separated from pi_theta, so
# the replay term had no direction to push along and the whole escape mechanism was
# inert. 1e-5 is the value the repo's own `sftlr` sweep arm already uses, so it is a
# known-safe magnitude rather than a guess. If kl_replay still sits near zero after
# ~100 steps, the clone is the bottleneck and nothing downstream of it can work.
export TAFR_SFT_LR=1.0e-5
export TAFR_SFT_BATCH_SIZE=64     # row cap; TAFR_SFT_MAX_TOKEN_LEN_PER_GPU (24576)
                                  # takes precedence and packs greedily, which is what
                                  # keeps 64 long failures from OOMing next to vLLM
# max_size is effectively moot: the collector is cleared unconditionally after every
# failure-SFT update (ray_trainer.py:2876), so the clone always trains on exactly one
# refresh interval of failures (~800 responses at interval=5) and never on a
# accumulated pool. Widening the window means raising the interval, not the capacity.
export TAFR_FAILURE_DATA_MAX_SIZE=null
export TAFR_FAILURE_DATA_SAMPLING=recent

# --- disk ---------------------------------------------------------------------
# 47G/run at save_freq=50 against ~117G free. save_freq=100 plus the launcher's
# hardcoded max_actor_ckpt_to_keep=1 holds a run to roughly 7G. This is not
# advisory: a full disk stalls a run, and resume_mode=auto then makes the retry
# silently continue the stalled checkpoint instead of starting clean.
export SAVE_FREQ=100

# --- fail fast on missing prerequisites ---------------------------------------
# These runs cost ~13 GPU-h each. Every check below is something that otherwise
# surfaces as a crash minutes in, or worse, as a silently wrong config.

require_file() {
  [[ -e "$1" ]] || { echo "MISSING: $1 -- $2" >&2; exit 4; }
}

require_config_key() {
  # Guards the Phase-0 config additions. Without this an arm that asks for a
  # not-yet-implemented key either dies in hydra or, if the schema is permissive,
  # runs to completion with the flag silently ignored -- 13 GPU-h for nothing.
  grep -q "^\s*${1%%=*}\b" verl/experimental/tafr_grpo/config.py \
    || { echo "UNKNOWN CONFIG KEY: custom_tafr_grpo.${1%%=*} is not in TAFRGRPOConfig." >&2
         echo "  Check the spelling against verl/experimental/tafr_grpo/config.py." >&2
         exit 5; }
}

launch() {
  local name=$1 desc=$2; shift 2   # remaining args are extra hydra overrides

  # Any custom_tafr_grpo.* key passed as an extra override must actually exist.
  # Hydra's schema would otherwise either reject it late or, worse, accept it and
  # run the whole arm with the flag silently ignored -- 13 GPU-h for a wrong row.
  local arg
  for arg in "$@"; do
    [[ $arg == custom_tafr_grpo.* ]] && require_config_key "${arg#custom_tafr_grpo.}"
  done

  require_file "${MODEL_PATH}" "download it first (see RUNBOOK step 0)"
  require_file "${TRAIN_FILE}" "training data"

  export EXPERIMENT_NAME="fepo-${name}-${TAG}"
  export CKPTS_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
  export VALIDATION_DATA_DIR="val_dumps/${EXPERIMENT_NAME}"

  # resume_mode=auto is hardcoded in the launcher, so an existing dir would be
  # resumed rather than restarted -- and a resumed arm is not the arm you asked for.
  if [[ -d "${CKPTS_DIR}" && "${ALLOW_RESUME:-0}" != "1" ]]; then
    echo "REFUSING: ${CKPTS_DIR} exists -- resume_mode=auto would continue it, not start fresh." >&2
    echo "  Delete it, or set ALLOW_RESUME=1 if you are deliberately resuming a crashed run." >&2
    exit 3
  fi

  local free_gb
  free_gb=$(df -BG --output=avail "${REPO_ROOT}" | tail -1 | tr -dc '0-9')
  if (( free_gb < 30 )); then
    echo "REFUSING: only ${free_gb}G free. Evaluate and delete a finished run's" >&2
    echo "  checkpoints first (RUNBOOK: disk hygiene). A run that fills the disk" >&2
    echo "  mid-training is worse than one that never started." >&2
    exit 6
  fi

  mkdir -p "${VALIDATION_DATA_DIR}"
  echo "=== ${EXPERIMENT_NAME} on GPU ${CUDA_VISIBLE_DEVICES} -- ${desc}"
  echo "    steps=${TOTAL_TRAINING_STEPS} tafr=${TAFR_ENABLE} variant=${TAFR_VARIANT}" \
       "beta_a=${TAFR_BETA_ANCHOR} beta_r=${TAFR_BETA_REPLAY} gamma=${TAFR_EMA_GAMMA} K=${TAFR_SFT_UPDATE_INTERVAL}"
  [[ $# -gt 0 ]] && echo "    extra: $*"
  echo "    val dumps -> ${VALIDATION_DATA_DIR} (this is what analyze.py reads)"

  # A stale WANDB_RUN_ID exported in the launching shell makes wandb RESUME that run,
  # so the arm silently appends into a different arm's history (m03_gspo landed in
  # m05_dapo's run d3v51tt8 this way). Derive it from the arm name instead: unique
  # per arm, and stable so a deliberate ALLOW_RESUME=1 retry rejoins its own run.
  export WANDB_RUN_ID=$(printf '%s' "${EXPERIMENT_NAME}" | md5sum | cut -c1-8)

  unset RAY_ADDRESS RAY_ARGS
  PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python} \
    bash examples/tafr_grpo/run_tafr_grpo_fsdp.sh "$@" 2>&1 \
    | tee -a "logs/logs_${EXPERIMENT_NAME}.log"
}

# Baseline arms measure KL against an anchor they do not use. TAFR must stay
# ENABLED for the anchor and failure models to be built at all -- diagnostic_only
# is what zeroes their contribution to the loss. Setting TAFR_ENABLE=false instead
# would leave the KL columns of Tables 2 and 3 empty and cost 4 extra runs.
diagnostic_only=( +custom_tafr_grpo.diagnostic_only=true )

# --- reusable base-objective overrides ----------------------------------------
# The launcher hardcodes policy_loss.loss_mode=vanilla, but "$@" is passed last and
# hydra lets the later duplicate win, so these override cleanly as extra args.

# GSPO: sequence-level importance ratio (core_algos.py:1538).
gspo=( actor_rollout_ref.actor.policy_loss.loss_mode=gspo )

# DAPO: clip-higher + dynamic sampling + overlong soft punishment.
#
# CAVEAT, must be reflected in the paper's method text: this fork's
# _filter_dapo_groups (ray_trainer.py:976) keeps only mixed groups and truncates to
# train_batch_size, but never resamples -- max_num_gen_batches is read and ignored.
# That is group *filtering*, not upstream DAPO dynamic sampling. Either implement
# the resample loop or describe the real behaviour; do not call this stock DAPO.
dapo=(
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  # '+' required: filter_groups is absent from the runtime algorithm struct, so a
  # plain override raises "Key 'filter_groups' is not in struct" before training starts.
  +algorithm.filter_groups.enable=true
  +algorithm.filter_groups.metric=acc
  +algorithm.filter_groups.max_num_gen_batches=4
  reward_model.reward_manager=dapo
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=true
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=512
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=false
  +reward_model.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
)
