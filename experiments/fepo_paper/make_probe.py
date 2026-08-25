#!/usr/bin/env python3
"""Build the fixed probe set that Table 2 (Recovery@delta + difficulty buckets) reads.

Why a probe set at all: Recovery@delta needs the SAME prompts scored at a reference
checkpoint and again delta steps later. Training rollouts cannot supply that -- each
step draws a different batch, so a given prompt is rarely revisited within delta=50.
A fixed set evaluated at every test_freq gives per-prompt success rate over time for
free, with no extra training and no new trainer code.

Why held out: dsr_math345 has 6601 rows and a 500-step run draws 500*64 = 32000
prompts, so every prompt is seen ~5 times during training. Scoring the probe on
prompts the model trained on would make Recovery@delta indistinguishable from
memorization. So this script writes BOTH the probe and a training file with the
probe rows removed, and the arm scripts must use the reduced training file.

    python experiments/fepo_paper/make_probe.py

Writes:
    /workspace/jepa-grpo-cache/eval_data/probe256.parquet
    /workspace/jepa-grpo-cache/data/dsr_math345/train_minus_probe.parquet
"""

import sys
from pathlib import Path

import pandas as pd

SRC = Path("/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet")
PROBE_OUT = Path("/workspace/jepa-grpo-cache/eval_data/probe256.parquet")
TRAIN_OUT = SRC.with_name("train_minus_probe.parquet")

N_PROBE = 256
SEED = 3407  # same seed as training, so the split is reproducible from the config alone


def main() -> int:
    if not SRC.exists():
        print(f"missing source parquet: {SRC}", file=sys.stderr)
        return 1

    df = pd.read_parquet(SRC)
    if len(df) <= N_PROBE * 2:
        print(f"refusing: {len(df)} rows is too few to spare {N_PROBE}", file=sys.stderr)
        return 1

    probe = df.sample(n=N_PROBE, random_state=SEED)
    train = df.drop(probe.index).reset_index(drop=True)
    probe = probe.reset_index(drop=True)

    # Relabel so the probe gets its own val/<source>/ metric namespace and, crucially,
    # stays out of BEST_CKPT_SOURCES -- checkpoint selection must not be driven by the
    # same prompts the mechanism analysis is measured on.
    probe["data_source"] = "probe"
    probe["extra_info"] = [{**r, "split": "test"} for r in probe["extra_info"]]

    PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
    probe.to_parquet(PROBE_OUT, index=False)
    train.to_parquet(TRAIN_OUT, index=False)

    print(f"probe: {len(probe):>5} rows -> {PROBE_OUT}")
    print(f"train: {len(train):>5} rows -> {TRAIN_OUT}  (was {len(df)})")
    print()
    print("Point _common.sh at the reduced training file:")
    print(f"  export TRAIN_FILE={TRAIN_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
