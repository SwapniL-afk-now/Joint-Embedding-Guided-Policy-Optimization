import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score


EVAL_DIR = Path("/workspace/jepa-grpo-cache/eval_data")
DEFAULT_BENCHMARKS = ["aime24", "aime25", "aime26", "amc23"]
DEFAULT_SEEDS = [31415, 27182, 16180]


def score_outputs(outputs, meta, n):
    correct = []
    for out, (data_source, ground_truth) in zip(outputs, meta):
        row = []
        for sample in out.outputs[:n]:
            row.append(bool(default_compute_score(data_source, sample.text, ground_truth)["acc"]))
        correct.append(row)
    return np.asarray(correct, dtype=bool)


def summarize(correct):
    first8 = correct[:, :8]
    first16 = correct[:, :16]
    return {
        "pass@1": float(correct[:, 0].mean()),
        "avg@8": float(first8.mean()),
        "pass@8": float(first8.any(axis=1).mean()),
        "avg@16": float(first16.mean()),
        "pass@16": float(first16.any(axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    meta = []
    spans = {}
    for benchmark in benchmarks:
        df = pd.read_parquet(EVAL_DIR / f"{benchmark}.parquet")
        start = len(prompts)
        for _, row in df.iterrows():
            prompts.append(
                tokenizer.apply_chat_template(
                    list(row["prompt"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            meta.append((row["data_source"], row["reward_model"]["ground_truth"]))
        spans[benchmark] = (start, len(prompts))

    batched_prompts = []
    batched_meta = []
    batched_keys = []
    sampling_params = []
    for seed in seeds:
        params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=seed,
        )
        for prompt, item_meta in zip(prompts, meta):
            batched_prompts.append(prompt)
            batched_meta.append(item_meta)
            batched_keys.append(seed)
            sampling_params.append(params)

    print(
        f"loaded {len(prompts)} prompts; evaluating {len(seeds)} seeds together "
        f"as {len(batched_prompts)} vLLM requests, n={args.n}"
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=False,
    )
    outputs = llm.generate(batched_prompts, sampling_params)
    correct = score_outputs(outputs, batched_meta, args.n)

    per_seed = {}
    for seed_idx, seed in enumerate(seeds):
        offset = seed_idx * len(prompts)
        seed_correct = correct[offset : offset + len(prompts)]
        per_seed[str(seed)] = {}
        for benchmark, (start, end) in spans.items():
            per_seed[str(seed)][benchmark] = summarize(seed_correct[start:end])

    per_benchmark = {}
    metric_names = ["pass@1", "avg@8", "pass@8", "avg@16", "pass@16"]
    for benchmark in benchmarks:
        per_benchmark[benchmark] = {"n_q": spans[benchmark][1] - spans[benchmark][0]}
        for metric in metric_names:
            values = np.asarray([per_seed[str(seed)][benchmark][metric] for seed in seeds], dtype=float)
            per_benchmark[benchmark][f"{metric}_mean"] = float(values.mean())
            per_benchmark[benchmark][f"{metric}_std"] = float(values.std())
            per_benchmark[benchmark][f"{metric}_seeds"] = values.tolist()

    macro = {}
    for metric in metric_names:
        values = np.asarray([per_benchmark[b][f"{metric}_mean"] for b in benchmarks], dtype=float)
        macro[metric] = float(values.mean())

    result = {
        "model": args.model,
        "benchmarks": benchmarks,
        "seeds": seeds,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "per_seed": per_seed,
        "per_benchmark": per_benchmark,
        "macro": macro,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    print("\nbenchmark      pass@1        avg@8         pass@8        avg@16        pass@16")
    for benchmark in benchmarks:
        row = per_benchmark[benchmark]
        print(
            f"{benchmark:12} "
            f"{row['pass@1_mean']:.4f}+/-{row['pass@1_std']:.4f}  "
            f"{row['avg@8_mean']:.4f}+/-{row['avg@8_std']:.4f}  "
            f"{row['pass@8_mean']:.4f}+/-{row['pass@8_std']:.4f}  "
            f"{row['avg@16_mean']:.4f}+/-{row['avg@16_std']:.4f}  "
            f"{row['pass@16_mean']:.4f}+/-{row['pass@16_std']:.4f}"
        )
    print(
        f"{'MACRO':12} "
        f"{macro['pass@1']:.4f}        {macro['avg@8']:.4f}        "
        f"{macro['pass@8']:.4f}        {macro['avg@16']:.4f}        {macro['pass@16']:.4f}"
    )
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
