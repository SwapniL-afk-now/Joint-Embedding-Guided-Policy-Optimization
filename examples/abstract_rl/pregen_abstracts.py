# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pre-generate the bootstrap abstractions once, offline, from a frozen model.

Why this exists: generating the warm-up targets with the policy being trained is a
self-distillation loop with no anchor. Measured on Qwen2.5-Math-1.5B-Instruct, mean
abstraction length fell 79 -> 3.7 tokens in three steps at lr 1e-5 and 80 -> 42 in
five at 3e-6, taking train accuracy down with it. Lower learning rates only slow it.

A frozen source fixes it outright, and generating from the QUESTION alone (rather
than question + that rollout's solution) also lets the whole thing move offline --
so training regains the spec's "no extra generation stage" property.

Usage:
    .venv/bin/python examples/abstract_rl/pregen_abstracts.py \\
        --train-files /workspace/jepa-grpo-cache/data/dsr_math345/train.parquet \\
        --model /workspace/models/Qwen2.5-1.5B-Instruct \\
        --out /workspace/jepa-grpo-cache/data/dsr_math345/abstracts.parquet
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

PROMPT = (
    "Here is a mathematics problem.\n\n"
    "Problem:\n{question}\n\n"
    "Do NOT solve it and do NOT state the answer. In AT MOST two short sentences, name the "
    "general reasoning strategy for problems of this kind: the reusable method, not the "
    "specific numbers. Be terse. Start directly with the method.\n\n"
    "Strategy:"
)


# The model echoes the prompt's framing ("Strategy:", "General Reasoning Strategy:")
# before the content. Left in, every abstraction in training starts with a label.
PREAMBLE_RE = re.compile(r"^\s*(?:the\s+)?(?:general\s+)?(?:reasoning\s+)?strategy\s*(?:is)?\s*:\s*", re.I)


def strip_preamble(text: str) -> str:
    return PREAMBLE_RE.sub("", text).strip()


def answer_of(row) -> str:
    extra = row.get("extra_info") or {}
    answer = extra.get("answer") if hasattr(extra, "get") else None
    return str(answer).strip() if answer is not None else ""


def leaks_answer(text: str, answer: str) -> bool:
    """Reject an abstraction that states the answer.

    Not a grading concern -- the verifier already truncates at the open tag -- but
    an abstraction containing the answer makes Delta (the likelihood gain from
    conditioning on A) trivially large for the wrong reason, which would poison
    the whole utility signal.
    """
    if not answer or len(answer) < 2:  # single digits match far too much prose
        return False
    return re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", text) is not None


def is_truncated(text: str) -> bool:
    """A target cut off mid-sentence teaches the policy to cut off mid-sentence."""
    return not text.rstrip().endswith((".", "!", "?"))


def question_of(row) -> str:
    """The bare question, matching what scoring.py's _question_text uses as the key."""
    extra = row.get("extra_info") or {}
    problem = extra.get("problem") if hasattr(extra, "get") else None
    if problem:
        return str(problem)
    prompt = row.get("prompt")
    if prompt is not None and len(prompt):
        return str(prompt[-1]["content"])
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--model", default="/workspace/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    frames = [pd.read_parquet(f) for f in args.train_files]
    df = pd.concat(frames, ignore_index=True)
    pairs = [(question_of(r), answer_of(r)) for _, r in df.iterrows()]
    answers = {q: a for q, a in pairs if q}
    keep = [q for q in dict.fromkeys(q for q, _ in pairs) if q]
    print(f"{len(df)} rows -> {len(keep)} unique questions")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=q)}], tokenize=False, add_generation_prompt=True
        )
        for q in keep
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    # Greedy: this file is generated once and reused, so reproducibility beats variety.
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    from verl.experimental.abstract_rl.bootstrap import clean_abstract, is_degenerate

    records = []
    drops = {"degenerate": 0, "truncated": 0, "leaked": 0}
    for q, o in zip(keep, outs, strict=True):
        text = strip_preamble(clean_abstract(o.outputs[0].text))
        if not text or is_degenerate(text):
            drops["degenerate"] += 1
        elif is_truncated(text):
            drops["truncated"] += 1
        elif leaks_answer(text, answers.get(q, "")):
            drops["leaked"] += 1
        else:
            records.append({"question": q, "abstract": text})

    out = pd.DataFrame.from_records(records)
    out.to_parquet(args.out, index=False)
    lens = out["abstract"].str.split().str.len()
    print(f"wrote {len(out)}/{len(keep)} abstracts to {args.out}; dropped {drops}")
    print(f"words: mean {lens.mean():.1f}, p10 {lens.quantile(0.1):.0f}, p90 {lens.quantile(0.9):.0f}")
    for r in records[:3]:
        print(f"\n--- {r['question'][:90]}\n    {r['abstract']}")


if __name__ == "__main__":
    main()
