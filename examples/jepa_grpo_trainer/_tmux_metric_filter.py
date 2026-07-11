#!/usr/bin/env python3
"""Line filter for tmux: keep only step / loss / grad_norm / avg@8 / pass@8.

Reads the trainer's stdout on stdin, prints a compact one-liner per logged
metric step and drops everything else (ray/vllm/tqdm chatter). Full output is
preserved separately via `tee` to the run log.
"""
import sys

def want(key: str) -> bool:
    k = key.lower()
    if "grad_norm" in k:
        return True
    if "loss" in k:
        return True
    if k.endswith("avg_at_8") or k.endswith("avg_at_k"):
        return True
    if k.endswith("pass_at_8"):
        return True
    return False

for line in sys.stdin:
    line = line.rstrip("\n")
    if not line.startswith("step:"):
        continue
    toks = [t.strip() for t in line.split(" - ")]
    kept = [toks[0]]  # step:N
    for t in toks[1:]:
        if ":" not in t:
            continue
        key = t.rsplit(":", 1)[0]
        if want(key):
            kept.append(t)
    print("  " + " | ".join(kept), flush=True)
