#!/usr/bin/env bash
# Wait for a GPU to go idle, then launch an arm on it.
#
#   setsid nohup bash experiments/fepo_paper/queue_on_free_gpu.sh 1 m03_gspo.sh \
#     >> queue_gpu1.log 2>&1 &
#
# Safe to start immediately after launching the previous arm, while the GPU is still
# empty: the queue waits for the card to go BUSY before it waits for it to go idle.
# (Earlier this took the previous arm's pid instead. Every way of obtaining that pid
# -- $! of the setsid wrapper, pgrep on the arm script -- handed back something that
# had already exited, so the queue skipped its wait and fired at once. The GPU itself
# is the only signal that cannot be handed to it wrong.)
#
# setsid+nohup is not decoration: the previous attempt at queueing an arm died with
# the shell that started it, which is why the queued run never began. Detached, this
# survives the terminal, ssh drop and agent session.
#
# "Idle" is read from nvidia-smi's compute-apps list for that GPU, not from the
# trainer's exit code -- a killed run frees the GPU too, and the point of the queue
# is to use the GPU the moment it is free.
set -euo pipefail

GPU=${1:?usage: queue_on_free_gpu.sh <gpu_index> <arm_script.sh>}
ARM=${2:?usage: queue_on_free_gpu.sh <gpu_index> <arm_script.sh>}
POLL=${POLL:-30}
busy() { [[ -n "$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; }

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "experiments/fepo_paper/${ARM}" ]] || { echo "no such arm: ${ARM}" >&2; exit 2; }

echo "$(date -Is) waiting for GPU ${GPU} to free, then: ${ARM}"

# Phase 1: wait for the GPU to be BUSY. Armed right after launching the previous arm
# the card is still empty (vLLM takes minutes to allocate), and without this the queue
# reads "idle", fires at once, and puts two trainers on one GPU.
while ! busy; do
  echo "$(date -Is) GPU ${GPU} not busy yet -- waiting for the running arm to allocate"
  sleep "${POLL}"
done
echo "$(date -Is) GPU ${GPU} busy -- now waiting for it to free"

clear_checks=0
while :; do
  pids=$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || echo BUSY)
  if [[ -z "${pids}" ]]; then
    clear_checks=$((clear_checks + 1))
  else
    [[ ${clear_checks} -gt 0 ]] && echo "$(date -Is) GPU ${GPU} busy again (${pids//$'\n'/,})"
    clear_checks=0
  fi
  # Two consecutive clear polls: a torn-down run's memory takes ~30s to be released,
  # and starting into a half-freed GPU is exactly how the OOM-at-startup happens.
  (( clear_checks >= 2 )) && break
  sleep "${POLL}"
done

echo "$(date -Is) GPU ${GPU} idle -- launching ${ARM}"
exec env ARM_GPU="${GPU}" bash "experiments/fepo_paper/${ARM}"
