# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MATH-P-Simple transfer eval: does the learned abstraction carry the method?

X is a level-5 MATH problem, X' its MATH-P-Simple perturbation -- a non-essential
edit solvable by the same underlying method. So an abstraction that captured the
*method* should help on X', while one that memorised X's numbers should not.

    A_ours      the policy's own <abstract> block, produced while solving X
    A_summary   a length-matched ordinary summary of the same solution
    A_shuffled  A_ours from a DIFFERENT problem (cyclic permutation)

A_summary controls for "any relevant text at the top of the prompt helps".
A_shuffled controls for "any abstraction-shaped text helps" and should be ~0 or
negative -- it is the null. The decisive result is

    Delta_ours > Delta_summary > Delta_shuffled ~ 0

with Delta_c = Pass@1(X', A_c) - Pass@1(X').

    .venv/bin/python examples/abstract_rl/eval_transfer.py \\
        --model <merged-hf-checkpoint> \\
        --pairs /workspace/jepa-grpo-cache/eval_data/math_perturb_pairs.parquet \\
        --out /workspace/logs/transfer_<name>.json
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

SOLVE_SUFFIX = "\n\nThink step by step and provide the final answer in \\boxed{...}."

# Same wording as the abstraction instruction so the two conditions differ in the
# CONTENT of the block, not in how it is introduced.
CONDITIONED_TEMPLATE = "Strategy for this kind of problem:\n{abstract}\n\n{question}" + SOLVE_SUFFIX

SUMMARY_TEMPLATE = (
    "Here is a problem and a worked solution.\n\nProblem:\n{question}\n\nSolution:\n{solution}\n\n"
    "Summarise the solution in at most {words} words. Do not add anything that is not in the "
    "solution above.\n\nSummary:"
)


def chat(tokenizer, text: str) -> str:
    msg = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def pass_at_1(scores: list[list[float]]) -> float:
    """Mean over problems of the mean over samples -- i.e. mean@k, the usual pass@1."""
    return float(np.mean([np.mean(s) for s in scores if s]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instruction", default="strong", choices=["spec", "strong"])
    ap.add_argument("--n", type=int, default=4, help="samples per condition for pass@1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.experimental.abstract_rl.config import INSTRUCTIONS
    from verl.experimental.abstract_rl.reward_fn import compute_score_ignoring_abstract

    pairs = pd.read_parquet(args.pairs)
    if args.limit:
        pairs = pairs.head(args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # --- stage 1: solve X with the abstraction instruction, harvest A_ours -----
    instruction = INSTRUCTIONS[args.instruction]
    solve_x = llm.generate([chat(tokenizer, f"{q}\n\n{instruction}") for q in pairs.original], greedy)
    solutions = [o.outputs[0].text for o in solve_x]

    tag = re.compile(r"<abstract>(.*?)</abstract>", re.S)
    abstracts, kept = [], []
    for i, sol in enumerate(solutions):
        m = tag.search(sol)
        if m and m.group(1).strip():
            abstracts.append(m.group(1).strip())
            kept.append(i)
    print(f"valid abstractions on X: {len(kept)}/{len(pairs)}")
    if not kept:
        raise SystemExit("no valid abstractions -- the policy never emits the tag; nothing to transfer")

    pairs = pairs.iloc[kept].reset_index(drop=True)
    solutions = [solutions[i] for i in kept]

    # --- stage 2: length-matched ordinary summaries ---------------------------
    summary_prompts = [
        chat(
            tokenizer,
            SUMMARY_TEMPLATE.format(
                question=r.original,
                # strip the abstraction so the summariser sees only the solution
                solution=tag.sub("", solutions[i]).strip(),
                words=max(8, len(abstracts[i].split())),
            ),
        )
        for i, r in enumerate(pairs.itertuples())
    ]
    summaries = [o.outputs[0].text.strip() for o in llm.generate(summary_prompts, greedy)]

    # A cyclic shift keeps the marginal distribution of abstractions identical and
    # only destroys the pairing with the problem -- that is the whole null.
    shuffled = abstracts[1:] + abstracts[:1]

    # --- stage 3: solve X' under the four conditions --------------------------
    def conditioned(blocks):
        return [
            CONDITIONED_TEMPLATE.format(abstract=a, question=q)
            for a, q in zip(blocks, pairs.perturbed, strict=True)
        ]

    conditions = {
        "bare": [q + SOLVE_SUFFIX for q in pairs.perturbed],
        "ours": conditioned(abstracts),
        "summary": conditioned(summaries),
        "shuffled": conditioned(shuffled),
    }
    sampling = SamplingParams(temperature=args.temperature, n=args.n, max_tokens=args.max_tokens)

    results, per_problem = {}, {}
    for name, prompts in conditions.items():
        outs = llm.generate([chat(tokenizer, p) for p in prompts], sampling)
        scores = [
            [
                float(
                    compute_score_ignoring_abstract("math_dapo", c.text, str(ans), None)["acc"]
                )
                for c in o.outputs
            ]
            for o, ans in zip(outs, pairs.perturbed_answer, strict=True)
        ]
        per_problem[name] = scores
        results[name] = pass_at_1(scores)
        print(f"pass@1 {name:9s} {results[name]:.4f}")

    deltas = {k: results[k] - results["bare"] for k in ("ours", "summary", "shuffled")}
    ordered = deltas["ours"] > deltas["summary"] > deltas["shuffled"]

    print("\n" + "\n".join(f"delta_{k:9s} {v:+.4f}" for k, v in deltas.items()))
    print(f"\ndecisive ordering (ours > summary > shuffled ~ 0): {'HOLDS' if ordered else 'DOES NOT HOLD'}")

    payload = {
        "model": args.model,
        "n_problems": len(pairs),
        "n_samples": args.n,
        "temperature": args.temperature,
        "pass_at_1": results,
        "deltas": deltas,
        "ordering_holds": bool(ordered),
        "abstract_words_mean": float(np.mean([len(a.split()) for a in abstracts])),
        "summary_words_mean": float(np.mean([len(s.split()) for s in summaries])),
        "per_problem": {k: [float(np.mean(s)) for s in v] for k, v in per_problem.items()},
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
