#!/usr/bin/env bash
# sc_fixed_ff — JEPA alignment fix pass (F1+F2+F3A+F5) with FULL FINETUNING.
# Fixes sc_fixed.sh (which silently inherited LORA_RANK=512 / DAPO_USE_LORA=true from
# run_drgrpo_jepa.sh defaults). Matches the GRPO baseline fw7zf51z training setup:
#   full FT (no LoRA), ACTOR_LR=1e-6, TRAIN_FILE=amcaime32k, TRAIN_SEED=14142, 1 GPU
# v2: paper-exact anchor (user view = PROBLEM ONLY + [PRED], read before the solution;
#   response appears only in the target) — finetune.py view pairing, no deviation.
# JEPA config: llm-jepa, N_COT=4 N_CODE=4, ALPHA=0.1, AUTO_OFF_MIN_COS=0.5,
#   self_code_targets=True, paper_sigreg_lambda=0.1, fuse_optimizer_step=True
# Checkpoints: checkpoints/grpo-qwen-3b/selfcode_sigreg_paper_anchor_666
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_sc_paper_anchor666.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_sc_paper_anchor666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
N_COT=4 \
N_CODE=4 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
DAPO_USE_LORA=false \
ACTOR_LR=1e-6 \
TRAIN_FILE=/workspace/jepa-grpo-cache/data/amcaime32k/train.parquet \
TRAIN_SEED=14142 \
VALIDATION_SEEDS='[14142,16180,271828,299792,31415]' \
NDEVICES_PER_NODE=1 \
MAX_OPTIMIZER_STEPS=666 \
TOTAL_TRAINING_STEPS=666 \
PROJECT_NAME=grpo-qwen-3b \
EXPERIMENT_NAME=selfcode_sigreg_paper_anchor_666 \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_sc_paper_anchor666.jsonl \
./run_drgrpo_jepa.sh \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  jepa.self_code_targets=True \
  jepa.paper_sigreg_lambda=0.1 \
  jepa.fuse_optimizer_step=True 2>&1 | tee /workspace/exploration/grad_direction_probe/log_sc_paper_anchor666.log
