#!/usr/bin/env python3
"""Build a text-only, long-trace OpenR1 subset in the local VERL schema.

Only the public problem, ground-truth answer, and generated response text are
consumed. Candidate responses are independently checked with the project's
math verifier; no teacher embeddings, logits, or hidden states are used.
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
    p.add_argument("--min-response-tokens", type=int, default=4096)
    # Difficulty gate: keep only problems whose teacher pass-rate
    # (correct generations / total) is >= this. 1.0 = fully-solved (easiest)
    # bucket, so the student gets non-zero reward and a real advantage signal.
    p.add_argument("--min-pass-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Keep one response per problem: JEPA needs one complete long target view,
    # not repeated teacher samples.  The streamed source order is deterministic;
    # stop once the requested subset is filled rather than spending CPU scanning
    # the entire 94k-problem source split.
    selected: list[tuple[dict, str, int]] = []
    scanned = 0
    for row in load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True):
        scanned += 1
        problem = row["problem"]
        answer = row["answer"]
        # Difficulty gate: skip problems the teacher solved less often than
        # requested. Keeps the easy portion so rewards aren't uniformly zero.
        verdicts = row["correctness_math_verify"] or []
        if not verdicts or (sum(bool(v) for v in verdicts) / len(verdicts)) < args.min_pass_rate:
            continue
        # Try the longest complete text first.  The quick character gate avoids
        # tokenizing/verifying short responses that cannot meet the requested
        # token threshold; every retained candidate is still independently
        # checked locally below.
        # OpenR1 publishes per-response completion and math-verifier outcomes.
        # These are used only to avoid parsing known failures; the final selected
        # text is independently checked again below before it enters the cache.
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
        chosen: tuple[int, str] | None = None
        for response in candidates:
            # A long teacher view must contain a finished thinking section.
            if "<think>" not in response or "</think>" not in response:
                continue
            if len(response) < args.min_response_tokens * 2:
                continue
            n_tokens = len(tokenizer(response, add_special_tokens=False)["input_ids"])
            if n_tokens < args.min_response_tokens:
                continue
            if compute_math_reward(response, answer, dataset_kind="math").is_correct:
                chosen = (n_tokens, response)
                break
        if chosen is None:
            continue
        n_tokens, response = chosen
        record = (row, response, n_tokens)
        selected.append(record)
        if len(selected) >= args.n_questions:
            break

    selected.sort(key=lambda x: x[0]["uuid"])
    rows = []
    teacher_cache: dict[int, list[str]] = {}
    lengths = []
    for index, (source, response, n_tokens) in enumerate(selected):
        rows.append({
            # Reuse the registered Math-Verify reward route used by the DAPO
            # baseline; provenance stays in extra_info.source_dataset below.
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
        teacher_cache[index] = [response]
        lengths.append(n_tokens)

    if len(rows) < args.n_questions:
        raise RuntimeError(
            f"Only found {len(rows)} verified complete responses with >= {args.min_response_tokens} tokens; "
            f"requested {args.n_questions}."
        )
    train_path = out_dir / "train.parquet"
    cache_path = out_dir / "teacher_targets.responses.pt"
    meta_path = out_dir / "metadata.json"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    tmp_cache = cache_path.with_name(cache_path.name + ".tmp")
    torch.save(teacher_cache, tmp_cache)
    os.replace(tmp_cache, cache_path)
    metadata = {
        "source": "open-r1/OpenR1-Math-220k:default",
        "n_questions": len(rows),
        "min_response_tokens": args.min_response_tokens,
        "min_pass_rate": args.min_pass_rate,
        "mean_teacher_response_tokens": sum(lengths) / len(lengths),
        "median_teacher_response_tokens": sorted(lengths)[len(lengths) // 2],
        "max_teacher_response_tokens": max(lengths),
        "scanned_source_questions": scanned,
        "verified_qualifying_source_questions": len(selected),
        "selection_seed": args.seed,
        "verification": "local compute_math_reward on public response text",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
