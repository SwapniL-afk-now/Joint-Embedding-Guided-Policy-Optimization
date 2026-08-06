#!/usr/bin/env bash
# Reproduces the sc_fus run (tmux session "sc_fus", started 2026-08-01 01:56).
# Checkpoints: checkpoints/grpo-qwen-3b/selfcode_sigreg_fused_r01_666
# Wandb:       run-20260801_015816-b1w9lonx
# Log:         /workspace/exploration/grad_direction_probe/log_sc_fus666.log
cd /workspace/Joint-Embedding-Guided-Policy-Optimization/examples/jepa_grpo_trainer/qwen2.5-math-1.5b
./run_drgrpo_jepa.sh jepa.self_code_targets=True jepa.paper_sigreg_lambda=0.1 jepa.fuse_optimizer_step=True 2>&1 | tee /workspace/exploration/grad_direction_probe/log_sc_fus666.log
