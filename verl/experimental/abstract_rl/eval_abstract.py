# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Held-out theory-validation diagnostics for abstract-reasoning compression.

Answers the question the training-time proxy cannot: does the likelihood gain
Delta correspond to actual solving utility?

    abstract_eval/pass1_with_abstract     solve X given the model's own abstraction
    abstract_eval/pass1_without_abstract  solve X cold
    abstract_eval/delta_pass1             the difference

plus copy/redundancy checks and a per-example ``(KL, Delta, |A|, |Y|, r)`` dump
for the rate-utility frontier.

This is the *only* place extra generation happens; it runs offline against a
checkpoint, never inside the training step.

    python -m verl.experimental.abstract_rl.eval_abstract \\
        --model checkpoints/.../huggingface --data /workspace/jepa-grpo-cache/eval_data/math500.parquet
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.experimental.abstract_rl.config import PRIOR_TEMPLATE, ROLLOUT_INSTRUCTION, SUFFICIENCY_TEMPLATE
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.reward_score import default_compute_score

OPEN_TAG, CLOSE_TAG = "<abstract>", "</abstract>"


def split_response(text: str) -> tuple[str, str]:
    """(Y, A) from a generated response, mirroring the training-time parser."""
    close = text.rfind(CLOSE_TAG)
    open_ = text.rfind(OPEN_TAG, 0, close) if close >= 0 else text.rfind(OPEN_TAG)
    if open_ < 0:
        return text, ""
    y = text[:open_]
    a = text[open_ + len(OPEN_TAG) : close] if close > open_ else ""
    return y, a


def _chat(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def _target_log_probs(llm, tokenizer, prompts: list[str], targets: list[str]) -> list[list[float]]:
    """Teacher-forced per-token log-probs of `target` given `prompt`.

    Uses vLLM prompt-logprobs (a forward pass, not decoding), so the whole
    diagnostic needs a single engine. The prompt boundary is taken from the
    tokenizer; the chat template ends on an assistant header, a stable token
    boundary, so the prompt segmentation is unaffected by the appended target.
    """
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([p + t for p, t in zip(prompts, targets, strict=True)], sp)
    n_prompt = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    scores = []
    for out, skip in zip(outs, n_prompt, strict=True):
        lp = out.prompt_logprobs or []
        row = []
        for entry in lp[skip:]:
            if entry:
                row.append(float(next(iter(entry.values())).logprob))
        scores.append(row)
    return scores


def _pass1(texts, meta) -> float:
    return float(np.mean([default_compute_score(ds, t, gt)["acc"] for t, (ds, gt) in zip(texts, meta, strict=True)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="held-out parquet in verl format")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=3584)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", default="abstract_eval.json")
    ap.add_argument("--dump", default="abstract_eval_tuples.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).head(args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=args.max_tokens + 2048)
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)

    questions, plain_prompts, abstract_prompts, meta = [], [], [], []
    for _, row in df.iterrows():
        messages = [dict(m) for m in row["prompt"]]
        plain_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] = f"{msg['content']}\n\n{ROLLOUT_INSTRUCTION}"
                break
        abstract_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        extra = row.get("extra_info") or {}
        questions.append(extra.get("problem") or messages[-1]["content"])
        meta.append((row["data_source"], row["reward_model"]["ground_truth"]))

    # 1. rollouts in the trained format -> Y, A, r
    rollouts = [o.outputs[0].text for o in llm.generate(abstract_prompts, sp)]
    ys, abstracts, rewards = [], [], []
    for text, (ds, gt) in zip(rollouts, meta, strict=True):
        y, a = split_response(text)
        ys.append(y)
        abstracts.append(a.strip())
        rewards.append(float(bool(default_compute_score(ds, y, gt)["acc"])))

    # 2. solver with and without the abstraction (extra generation, eval only)
    cold = [o.outputs[0].text for o in llm.generate(plain_prompts, sp)]
    pass1_without = _pass1(cold, meta)

    with_prompts = [
        _chat(tokenizer, SUFFICIENCY_TEMPLATE.format(question=q, abstract=a))
        for q, a in zip(questions, abstracts, strict=True)
    ]
    warm = [o.outputs[0].text for o in llm.generate(with_prompts, sp)]
    pass1_with = _pass1(warm, meta)

    # 3. Delta and compression KL on the same rollouts
    valid = [i for i, a in enumerate(abstracts) if a]
    cond_prompts = [
        _chat(tokenizer, SUFFICIENCY_TEMPLATE.format(question=questions[i], abstract=abstracts[i])) for i in valid
    ]
    base_prompts = [_chat(tokenizer, SUFFICIENCY_TEMPLATE.format(question=questions[i], abstract="")) for i in valid]
    prior_prompts = [_chat(tokenizer, PRIOR_TEMPLATE.format(question=questions[i])) for i in valid]
    post_prompts = [abstract_prompts[i] + ys[i] + OPEN_TAG for i in valid]

    cond = _target_log_probs(llm, tokenizer, cond_prompts, [ys[i] for i in valid])
    base = _target_log_probs(llm, tokenizer, base_prompts, [ys[i] for i in valid])
    prior = _target_log_probs(llm, tokenizer, prior_prompts, [abstracts[i] for i in valid])
    post = _target_log_probs(llm, tokenizer, post_prompts, [abstracts[i] for i in valid])

    rows = []
    for k, i in enumerate(valid):
        n_y = max(len(cond[k]), 1)
        delta = (sum(cond[k]) - sum(base[k])) / n_y
        n = min(len(prior[k]), len(post[k]))
        kl = 0.0
        if n:
            kl = float(
                kl_penalty(
                    logprob=torch.tensor(post[k][:n]),
                    ref_logprob=torch.tensor(prior[k][:n]),
                    kl_penalty="low_var_kl",
                ).mean()
            )
        y_ids = tokenizer(ys[i], add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(abstracts[i], add_special_tokens=False)["input_ids"]
        matched = sum(b.size for b in SequenceMatcher(None, a_ids, y_ids).get_matching_blocks())
        gt = str(meta[i][1])
        rows.append(
            {
                "index": i,
                "kl": kl,
                "delta": delta,
                "abstract_tokens": len(a_ids),
                "reasoning_tokens": len(y_ids),
                "reward": rewards[i],
                "token_overlap": len(set(a_ids) & set(y_ids)) / max(len(set(a_ids)), 1),
                "lcs_ratio": matched / max(len(a_ids), 1),
                "answer_copied": float(bool(gt) and gt in abstracts[i]),
            }
        )

    with open(args.dump, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    def _mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    summary = {
        "abstract_eval/pass1_with_abstract": pass1_with,
        "abstract_eval/pass1_without_abstract": pass1_without,
        "abstract_eval/delta_pass1": pass1_with - pass1_without,
        "abstract_eval/valid_abstract_rate": len(valid) / max(len(abstracts), 1),
        "abstract_eval/token_overlap_with_reasoning": _mean("token_overlap"),
        "abstract_eval/lcs_ratio": _mean("lcs_ratio"),
        "abstract_eval/answer_copy_rate": _mean("answer_copied"),
        "abstract_eval/delta_mean": _mean("delta"),
        "abstract_eval/kl_mean": _mean("kl"),
        "abstract_eval/compression_ratio": (
            float(np.mean([r["abstract_tokens"] / max(r["reasoning_tokens"], 1) for r in rows]))
            if rows
            else float("nan")
        ),
    }
    if len(rows) > 2:
        kls = np.array([r["kl"] for r in rows])
        deltas = np.array([r["delta"] for r in rows])
        lens = np.array([r["abstract_tokens"] for r in rows], dtype=float)
        summary["abstract_eval/corr_kl_delta"] = float(np.corrcoef(kls, deltas)[0, 1])
        summary["abstract_eval/corr_len_delta"] = float(np.corrcoef(lens, deltas)[0, 1])

    for key, value in summary.items():
        print(f"{key:46} {value:.4f}")
    json.dump({"model": args.model, "data": args.data, "n": len(df), **summary}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}  (per-example tuples -> {args.dump})")


if __name__ == "__main__":
    main()
