#!/usr/bin/env bash
# sc_fixed — JEPA alignment fix pass 1 (F1+F2+F3A+F5), launched 2026-08-01.
# Replaces the latched sc_dec/sc_fus pair. Changes vs sc_fus (selfcode_sigreg_fused_r01_666):
#   F1  anchor rows = [prompt; response] (paper view pairing; was response-only)
#   F2  stop-grad on the target branch (worker jepa_update llm-jepa _loss_fn)
#   F3A centered cosine in llm_jepa_paper_loss (center=True default)
#   F5  ALPHA 0.03 -> 0.1, AUTO_OFF_MIN_COS 0.0 -> 0.5 (latch gate on centered cos)
# Checkpoints: checkpoints/grpo-qwen-3b/selfcode_sigreg_fixed_r01_666
# Metrics:     /workspace/exploration/grad_direction_probe/metrics_sc_fixed666.jsonl
# Log:         /workspace/exploration/grad_direction_probe/log_sc_fixed666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
N_COT=4 \
N_CODE=4 \
JEPA_LOSS_TYPE=llm-jepa \
ALPHA=0.1 \
AUTO_OFF_MIN_COS=0.5 \
MAX_OPTIMIZER_STEPS=666 \
TOTAL_TRAINING_STEPS=666 \
PROJECT_NAME=grpo-qwen-3b \
EXPERIMENT_NAME=selfcode_sigreg_fixed_r01_666 \
LOGGER='["console","wandb","file"]' \
VERL_FILE_LOGGER_PATH=/workspace/exploration/grad_direction_probe/metrics_sc_fixed666.jsonl \
./run_drgrpo_jepa.sh jepa.self_code_targets=True jepa.paper_sigreg_lambda=0.1 jepa.fuse_optimizer_step=True 2>&1 | tee /workspace/exploration/grad_direction_probe/log_sc_fixed666.log
