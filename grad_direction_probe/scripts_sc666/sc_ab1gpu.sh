#!/usr/bin/env bash
# sc_ab1gpu — A/B leg: EXACTLY sc_fixed config but 1 GPU (NDEVICES_PER_NODE=1).
# Tests the user's hypothesis that 2-GPU training shrinks actor/grad_norm.
# Compare actor/grad_norm with 25glww16 (sc_fixed, 2 GPU): 2-GPU first5 = 0.0078/0.0051/0.0078/0.0069/0.0068.
# Everything else identical: TRAIN_FILE=dapo_math_17k (script default), LORA_RANK=512,
# ACTOR_LR=5e-7 (script default), TRAIN_SEED=31415, N_COT=4 N_CODE=4, llm-jepa, ALPHA=0.1.
# Checkpoints: checkpoints/grpo-qwen-3b/selfcode_sigreg_fixed_r01_040_1gpu
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_sc_ab1gpu40.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_sc_ab1gpu40.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
N_COT=4 \
N_CODE=4 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
NDEVICES_PER_NODE=1 \
MAX_OPTIMIZER_STEPS=40 \
TOTAL_TRAINING_STEPS=40 \
PROJECT_NAME=grpo-qwen-3b \
EXPERIMENT_NAME=selfcode_sigreg_fixed_r01_040_1gpu \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_sc_ab1gpu40.jsonl \
./run_drgrpo_jepa.sh jepa.self_code_targets=True jepa.paper_sigreg_lambda=0.1 jepa.fuse_optimizer_step=True 2>&1 | tee /workspace/exploration/grad_direction_probe/log_sc_ab1gpu40.log
