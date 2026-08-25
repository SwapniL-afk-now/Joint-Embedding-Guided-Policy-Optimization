#!/usr/bin/env python3
"""Build the stage-1 solved-trace cache from open-r1/OpenR1-Math-220k.

Two-stage JEPA training, stage-1 data: for every problem, keep ONE verified-correct,
reasoning-complete response (DeepSeek-R1 style trace) whose token count is within
[MIN_RESPONSE_TOKENS, MAX_RESPONSE_TOKENS]. The long original script
(build_openr1_long_trace_subset.py) selects traces of >= 4096 tokens for the
overlong-trace arm; this one is the <= 3072-token arm so traces fit the
Qwen2.5-Math 4096-token context window together with the problem prompt.

Outputs (all keyed by a common row index):
  solved_traces_openr1_max3072.pt   {index: [problem_text, trace_text]}  (pretrain_offline.py format)
  train.parquet                     verl-schema rows (prompt/reward_model/extra_info)
  metadata.json                     build config + counts
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from verl.experimental.fepo.math_parser import compute_math_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-questions", type=int, default=17_000)
    p.add_argument("--max-response-tokens", type=int, default=3072)
    p.add_argument("--min-response-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--tokenizer", default="/workspace/models/Qwen2.5-Math-1.5B-Instruct")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # One response per problem (JEPA needs one complete target view). Stream the
    # source; stop once the requested subset is filled.
    selected: list[tuple[dict, str, int]] = []
    scanned = 0
    skipped_no_candidates = 0
    for row in load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True):
        scanned += 1
        problem = row["problem"]
        answer = row["answer"]
        candidates = sorted(
            (
                response
                for response, is_complete, is_math_correct in zip(
                    row["generations"], row["is_reasoning_complete"], row["correctness_math_verify"]
                )
                if is_complete and is_math_correct
            ),
            key=len,
            reverse=True,
        )
        if not candidates:
            skipped_no_candidates += 1
            continue
        # Longest verified-correct complete response that still fits the budget.
        chosen: tuple[int, str] | None = None
        for response in candidates:
            n_tokens = len(tokenizer(response, add_special_tokens=False)["input_ids"])
            if not (args.min_response_tokens <= n_tokens <= args.max_response_tokens):
                continue
            if compute_math_reward(response, answer, dataset_kind="math").is_correct:
                chosen = (n_tokens, response)
                break
        if chosen is None:
            continue
        selected.append((row, chosen[1], chosen[0]))
        if scanned % 500 == 0:
            print(f"scanned {scanned} rows, selected {len(selected)}", flush=True)
        if len(selected) >= args.n_questions:
            break

    if len(selected) < args.n_questions:
        print(f"WARNING: only {len(selected)} rows met the budget (requested {args.n_questions})")

    selected.sort(key=lambda x: str(x[0]["uuid"] or ""))
    rows = []
    trace_cache: dict[int, list[str]] = {}
    lengths = []
    for index, (source, response, n_tokens) in enumerate(selected):
        rows.append({
            "data_source": "dapo-math-17k",
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a careful mathematical problem solver. Reason step by step and put the final answer in \\boxed{}.",
                },
                {"role": "user", "content": source["problem"] + "\nSolve the problem. Show your reasoning and put the final answer in \\boxed{}."},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": source["answer"]},
            "extra_info": {
                "split": "train",
                "index": index,
                "source_dataset": "open-r1/OpenR1-Math-220k:default",
                "source_uuid": source["uuid"],
                "problem": source["problem"],
                "teacher_response_tokens": n_tokens,
            },
        })
        trace_cache[index] = [source["problem"], response]
        lengths.append(n_tokens)

    train_path = out_dir / "train.parquet"
    cache_path = out_dir / "solved_traces_openr1_max3072.pt"
    meta_path = out_dir / "metadata.json"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    tmp_cache = cache_path.with_name(cache_path.name + ".tmp")
    torch.save(trace_cache, tmp_cache)
    os.replace(tmp_cache, cache_path)
    metadata = {
        "n_rows": len(rows),
        "n_scanned": scanned,
        "skipped_no_candidates": skipped_no_candidates,
        "max_response_tokens": args.max_response_tokens,
        "min_response_tokens": args.min_response_tokens,
        "source": "open-r1/OpenR1-Math-220k:default",
        "tokenizer": args.tokenizer,
        "lengths": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"wrote {len(rows)} rows -> {train_path}")
    print(f"wrote trace cache -> {cache_path}")
    print(f"metadata: {json.dumps(metadata)}")
    # Streaming datasets + tokenizers leave ragged thread state at interpreter
    # teardown; files are already flushed above, so exit hard.
    os._exit(0)


if __name__ == "__main__":
    main()
