# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build a GPI-CE training parquet from zwhe99/DeepMath-103K.

    python -m verl.experimental.gpi_ce.prepare_deepmath \
        --output /workspace/jepa-grpo-cache/data/deepmath_gpi/train.parquet

Each source row carries a question, a verified final_answer, a float difficulty
(1.5-9.5) and three independent R1 solutions. One of them (``--offline-columns``)
becomes the offline candidate of that prompt's GPI-CE group.

Two decisions worth knowing about:

*Solutions are filtered, never truncated.* R1 traces are long (median ~4.5k
tokens, p90 ~10k). A truncated trace has no final answer, so the verifier scores
it 0 and the offline candidate actively poisons its own group -- the opposite of
a "verified correct" demonstration. Rows whose solution does not fit in
``--max-response-tokens`` are dropped instead.

*Rows are sorted by difficulty, easiest first.* The dataloader must then run with
``data.shuffle=False`` (SequentialSampler) for the curriculum to survive.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from verl.utils.seqlen_balancing import ceildiv

# Matches the system prompt of the existing amcaime32k parquet so prompts, the
# verifier and the eval sets stay directly comparable across runs.
SYSTEM_PROMPT = (
    "You are a careful mathematician thinking aloud in natural prose.\n\n"
    "Restate what is asked and the constraints, noting anything easy to misread "
    "(units, ranges, integer-only). Lay out the steps you expect to take -- as many "
    "as the problem needs -- and add or revise steps as the work reveals them. Carry "
    "them out, showing your work. If a step fails, say so, return to where it went "
    "wrong, and continue. Check the result against the constraints before you commit "
    "to it.\n\nPut your final answer within \\boxed{}."
)

DATA_SOURCE = "math_dapo"
BOXED = re.compile(r"\\boxed\{")


def _fits(texts: list[str], tokenizer, max_tokens: int) -> dict[str, bool]:
    """Which texts fit in ``max_tokens``, computed the cheap way.

    A BPE token always spans >= 1 character, so ``len(text) <= max_tokens`` PROVES the
    text fits and needs no tokenization at all. Only the leftovers are encoded, and
    those go through the fast tokenizer in one batched (Rust-parallel) call rather than
    one Python call each. Same answer as tokenizing everything, ~20x less work: at a
    31744-token limit almost every R1 solution clears the character bound outright.
    """
    verdict = {}
    unresolved = []
    for text in texts:
        if len(text) <= max_tokens:
            verdict[text] = True
        else:
            unresolved.append(text)
    if unresolved:
        encoded = tokenizer(unresolved, add_special_tokens=False)["input_ids"]
        for text, ids in zip(unresolved, encoded, strict=True):
            verdict[text] = len(ids) <= max_tokens
    return verdict


def build(args) -> pd.DataFrame:
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"loading {args.dataset} ...", flush=True)
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    print(f"  {len(ds)} rows", flush=True)

    offline_columns = [c.strip() for c in args.offline_columns.split(",") if c.strip()]
    records, stats = [], {"no_answer": 0, "no_boxed": 0, "too_long": 0, "difficulty": 0, "kept": 0}

    # One length verdict for every candidate text up front, batched.
    print("  checking solution lengths ...", flush=True)
    candidates = {
        text
        for row in ds
        for column in offline_columns
        if (text := (row.get(column) or "").strip()) and BOXED.search(text)
    }
    fits = _fits(sorted(candidates), tokenizer, args.max_response_tokens)
    print(f"  {len(candidates)} candidate solutions, {sum(fits.values())} fit", flush=True)

    for row_no, row in enumerate(ds):
        if row_no and row_no % 20000 == 0:
            print(f"  ... {row_no}/{len(ds)} rows, kept {stats['kept']}", flush=True)
        answer = (row.get("final_answer") or "").strip()
        if not answer:
            stats["no_answer"] += 1
            continue
        difficulty = float(row.get("difficulty") or 0.0)
        if difficulty < args.min_difficulty or difficulty > args.max_difficulty:
            stats["difficulty"] += 1
            continue

        solutions, rewards = [], []
        for column in offline_columns:
            text = (row.get(column) or "").strip()
            if not text:
                continue
            # The offline candidate is scored by the SAME rule-based verifier as the
            # online rollouts, which extracts \boxed{...}. No box => unscoreable.
            if not BOXED.search(text):
                stats["no_boxed"] += 1
                continue
            if not fits.get(text, False):
                stats["too_long"] += 1
                continue
            solutions.append(text)
            rewards.append(1.0)
            if len(solutions) >= args.offline_per_prompt:
                break

        if len(solutions) < args.offline_per_prompt:
            continue

        records.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "difficulty": difficulty,
                "offline_responses": solutions,
                "offline_rewards": rewards,
                "extra_info": {
                    "split": "train",
                    "problem": row["question"],
                    "answer": answer,
                    "difficulty": difficulty,
                    "topic": row.get("topic", ""),
                },
            }
        )
        stats["kept"] += 1

    print(f"  filtered: {stats}", flush=True)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise SystemExit("no rows survived filtering; loosen --max-response-tokens or --min-difficulty")

    # Curriculum: easiest first. Ties broken deterministically so reruns match.
    df = df.sort_values(["difficulty"], kind="mergesort").reset_index(drop=True)
    for i, extra in enumerate(df["extra_info"]):
        extra["index"] = i
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="zwhe99/DeepMath-103K")
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--offline-columns", default="r1_solution_1,r1_solution_2,r1_solution_3")
    p.add_argument("--offline-per-prompt", type=int, default=1)
    # Model max for Qwen2.5-1.5B-Instruct: 32768 context - 1024 prompt. No R1
    # solution in the corpus reaches it, so nothing is dropped for length.
    p.add_argument("--max-response-tokens", type=int, default=31744)
    p.add_argument("--min-difficulty", type=float, default=0.0)
    p.add_argument("--max-difficulty", type=float, default=10.0)
    p.add_argument("--max-rows", type=int, default=-1)
    # Keep each shard small enough that pyarrow reads every nested column as ONE chunk.
    p.add_argument("--shard-rows", type=int, default=5000)
    args = p.parse_args()

    df = build(args)
    from pathlib import Path

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for stale in out.parent.glob(f"{out.stem}-*-of-*.parquet"):
        stale.unlink()

    # Shard the output. Every interesting column here is nested (prompt,
    # offline_responses, reward_model, extra_info), and once a single parquet grows
    # large enough that pyarrow splits a column into multiple chunks, reading it back
    # dies with "Nested data conversions not implemented for chunked array outputs".
    # Keeping each shard small keeps every column a single chunk on read.
    n_shards = max(1, ceildiv(len(df), args.shard_rows)) if args.shard_rows > 0 else 1
    if n_shards == 1:
        df.to_parquet(out, index=False)
        written = [out]
    else:
        written = []
        for i in range(n_shards):
            part = out.with_name(f"{out.stem}-{i:05d}-of-{n_shards:05d}.parquet")
            df.iloc[i * args.shard_rows : (i + 1) * args.shard_rows].to_parquet(part, index=False)
            written.append(part)
        if out.exists():
            out.unlink()  # never leave a stale single-file version to be picked up
    print(f"wrote {len(written)} shard(s), {sum(p.stat().st_size for p in written) / 1e9:.2f} GB total")

    d = df["difficulty"].to_numpy()
    print(f"\nwrote {len(df)} rows -> {out.parent}")
    print(
        f"difficulty {d.min():.1f} -> {d.max():.1f} "
        f"(first 1000 rows mean {d[:1000].mean():.2f}, last 1000 {d[-1000:].mean():.2f})"
    )
    edges = np.arange(1.0, 10.5, 1.0)
    counts, _ = np.histogram(d, bins=edges)
    for lo, c in zip(edges[:-1], counts, strict=True):
        if c:
            print(f"  [{lo:.1f},{lo + 1:.1f}) {c:6d}")


if __name__ == "__main__":
    main()
