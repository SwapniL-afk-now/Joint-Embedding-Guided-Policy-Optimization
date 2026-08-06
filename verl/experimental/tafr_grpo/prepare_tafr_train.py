# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build the TAFR-GRPO training parquet: One-Shot-RLVR dsr_sub + MATH levels 3-5.

    python -m verl.experimental.tafr_grpo.prepare_tafr_train \
        --output /workspace/jepa-grpo-cache/data/dsr_math345/train.parquet

Two sources, normalized to one schema:

* ``ypwang61/One-Shot-RLVR-Datasets`` split ``dsr_sub`` -- already in verl format,
  but its ``data_source`` is ``deepscaler``, which is NOT a key the reward router
  knows (only ``deepscaler-preview-dataset`` is). Left alone it would fall through
  to the default scorer. Rewritten to ``math_dapo``, the same verifier the eval
  parquets use.
* ``EleutherAI/hendrycks_math`` (all 7 subjects), levels 3-5 -- has no answer
  column, only a ``solution`` with a ``\\boxed{}``. The answer is extracted from
  it; rows without a parseable box are dropped, not kept with an empty target.

Prompts are rebuilt to byte-match the eval parquets (same system prompt, same
user suffix) so the policy sees one format at train and eval time.

Every extracted answer is round-tripped through the actual reward function: a
response consisting of just ``\\boxed{answer}`` must score correct, or the row is
dropped. That catches silently unverifiable ground truths before a 1000-step run
does. Pass ``--skip-verify`` to keep them.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

# Byte-identical to the eval parquets (math500/amc23/aime*), so train and eval
# prompts differ only in the problem text.
SYSTEM_PROMPT = "You are a careful mathematical problem solver. Reason step by step and put the final answer in \\boxed{}."
USER_SUFFIX = "\nSolve the problem. Show your reasoning and put the final answer in \\boxed{}."

# The router key that reaches verl.experimental.fepo.math_parser.compute_math_reward,
# which is what every eval set here is scored with.
DATA_SOURCE = "math_dapo"
LEVELS = {"Level 3", "Level 4", "Level 5"}
SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "precalculus",
    "prealgebra",
)


def _record(problem: str, answer: str, source: str) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem + USER_SUFFIX},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": "train", "index": 0, "problem": problem, "answer": answer, "source": source},
    }


def _load_dsr_sub() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("ypwang61/One-Shot-RLVR-Datasets", split="dsr_sub")
    out = []
    for row in ds:
        user = next((m["content"] for m in row["prompt"] if m["role"] == "user"), "")
        # Strip the source's own instruction tail; ours is appended by _record.
        problem = user.split("Let's think step by step")[0].strip()
        answer = (row["reward_model"] or {}).get("ground_truth", "").strip()
        if problem and answer:
            out.append(_record(problem, answer, "dsr_sub"))
    return out


def _load_hendrycks() -> tuple[list[dict], int]:
    from datasets import load_dataset

    from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

    out, dropped = [], 0
    for subject in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        for row in ds:
            if row["level"] not in LEVELS:
                continue
            try:
                answer = remove_boxed(last_boxed_only_string(row["solution"]))
            except Exception:
                answer = None
            if not answer or not answer.strip():
                dropped += 1
                continue
            out.append(_record(row["problem"].strip(), answer.strip(), f"hendrycks/{subject}/{row['level']}"))
    return out, dropped


def verify(df: pd.DataFrame) -> pd.Series:
    """Score ``\\boxed{ground_truth}`` with the real reward fn; True where it comes back correct.

    A ground truth that cannot verify against itself can never be earned by a rollout --
    the prompt is a guaranteed zero that only adds gradient noise. One row in the current
    build fails this way (unbalanced braces in the source LaTeX).
    """
    from verl.utils.reward_score import default_compute_score

    def ok(row) -> bool:
        gt = row["reward_model"]["ground_truth"]
        res = default_compute_score(row["data_source"], f"The answer is \\boxed{{{gt}}}", gt)
        return (res.get("acc") if isinstance(res, dict) else res) in (True, 1, 1.0)

    return df.apply(ok, axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--skip-verify", action="store_true", help="skip the reward round-trip (and its row drops)")
    args = p.parse_args()

    dsr = _load_dsr_sub()
    hend, dropped = _load_hendrycks()
    print(f"dsr_sub: {len(dsr)} rows", flush=True)
    print(f"hendrycks L3-5: {len(hend)} rows ({dropped} dropped: no parseable \\boxed{{}})", flush=True)

    df = pd.DataFrame.from_records(dsr + hend)
    before = len(df)
    # Deterministic shuffle: the two sources arrive in blocks, and MATH is ordered by
    # subject. Without this the first steps see only dsr_sub and the last only precalculus.
    df["_key"] = df["extra_info"].map(lambda e: hashlib.md5(e["problem"].encode()).hexdigest())
    df = df.drop_duplicates(subset="_key").sort_values("_key", kind="mergesort")
    df = df.drop(columns="_key").reset_index(drop=True)
    print(f"deduped + shuffled: {before} -> {len(df)}", flush=True)

    if not args.skip_verify:
        good = verify(df)
        bad = df[~good]
        print(f"parse check: {int(good.sum())} ok / {len(bad)} unparseable ({100 * len(bad) / len(df):.2f}%)", flush=True)
        for _, row in bad.head(10).iterrows():
            print("  dropped ground truth:", repr(row["reward_model"]["ground_truth"]), flush=True)
        df = df[good].reset_index(drop=True)

    for i, extra in enumerate(df["extra_info"]):
        extra["index"] = i

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
