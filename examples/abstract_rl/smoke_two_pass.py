# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Throwaway diagnostic: does a genuine two-pass (solve, then ask for the
abstract as a fresh generation) get real content out of the Math model?

Not wired into training -- this only answers "is the two-pass idea worth the
engineering," on a handful of problems. Pass 1 is a plain solve (no abstract
instruction at all, so r_i is untouched by anything abstract-related). Pass 2
prompts fresh with the full transcript and asks for the strategy.

    .venv/bin/python examples/abstract_rl/smoke_two_pass.py --n 8
"""

from __future__ import annotations

import argparse

import pandas as pd

SOLVE_SUFFIX = "\n\nThink step by step and provide the final answer in \\boxed{...}."
# v1 ("state the reasoning strategy") -> full paraphrase, 262 words, every
# number repeated. v2 (word cap + "no numbers" + long rule list) -> ignored
# outright, same drift into re-deriving the answer. Long, multi-part
# instructions don't stick on a 1.5B model. Going short and imperative instead
# -- one plain ask, one explicit "stop" cue -- since that's what this model's
# size can actually hold onto.
ABSTRACT_PROMPT = (
    "{question}{solve_suffix}\n\n{solution}\n\n"
    "In one short sentence, name the type of problem. In one more, give the general method, "
    "with no numbers. Then stop."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--data", default="/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--abstract-max-tokens", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.experimental.abstract_rl.reward_fn import compute_score_ignoring_abstract

    df = pd.read_parquet(args.data).head(args.n)
    questions = [r["problem"] for r in df.extra_info]
    answers = [str(r["answer"]) for r in df.extra_info]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=4096)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def chat(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

    # pass 1: plain solve, no abstract instruction anywhere
    solve_prompts = [chat(q + SOLVE_SUFFIX) for q in questions]
    solutions = [o.outputs[0].text for o in llm.generate(solve_prompts, greedy)]
    scores = [
        compute_score_ignoring_abstract("math_dapo", s, a, None)["acc"] for s, a in zip(solutions, answers, strict=True)
    ]

    # pass 2: fresh generation asking for the strategy, given the real transcript
    abstract_prompts = [
        chat(ABSTRACT_PROMPT.format(question=q, solve_suffix=SOLVE_SUFFIX, solution=s))
        for q, s in zip(questions, solutions, strict=True)
    ]
    abstract_sampling = SamplingParams(temperature=0.0, max_tokens=args.abstract_max_tokens)
    abstracts = [o.outputs[0].text.strip() for o in llm.generate(abstract_prompts, abstract_sampling)]

    n_empty = sum(1 for a in abstracts if not a.strip())
    n_words = [len(a.split()) for a in abstracts if a.strip()]
    n_digits = sum(1 for a in abstracts if any(c.isdigit() for c in a))
    print(f"\n{'=' * 88}\nsolved {sum(scores)}/{len(scores)} correct   empty abstracts: {n_empty}/{len(abstracts)}")
    print(f"abstracts containing a digit (number leak): {n_digits}/{len(abstracts)}")
    if n_words:
        print(f"abstract length: mean {sum(n_words) / len(n_words):.1f} words, min {min(n_words)}, max {max(n_words)}")

    for i in range(len(questions)):
        print(f"\n{'-' * 88}\nQUESTION : {questions[i][:200]}")
        print(f"SOLUTION (tail): ...{solutions[i][-300:]}")
        print(f"r={scores[i]:.0f}")
        print(f"ABSTRACT : {abstracts[i]}")


if __name__ == "__main__":
    main()
