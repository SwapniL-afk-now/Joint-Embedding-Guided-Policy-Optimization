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
            # Most scorers return {"score","acc",...}; gsm8k returns a bare float.
            res = default_compute_score(data_source, sample.text, ground_truth)
            row.append(bool(res["acc"] if isinstance(res, dict) else res > 0))
        correct.append(row)
    return np.asarray(correct, dtype=bool)


METRIC_NAMES = ["avg@32", "pass@1", "pass@8", "pass@16", "pass@32"]


def summarize(correct):
    n = correct.shape[1]
    out = {"pass@1": float(correct[:, 0].mean())}
    for k in (8, 16, 32):
        if k <= n:
            out[f"avg@{k}"] = float(correct[:, :k].mean())
            out[f"pass@{k}"] = float(correct[:, :k].any(axis=1).mean())
    return out


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
    parser.add_argument("--max-num-seqs", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
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
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
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
    metric_names = list(per_seed[str(seeds[0])][benchmarks[0]].keys())
    for benchmark in benchmarks:
        per_benchmark[benchmark] = {"n_q": spans[benchmark][1] - spans[benchmark][0]}
        for metric in metric_names:
            values = np.asarray([per_seed[str(seed)][benchmark][metric] for seed in seeds], dtype=float)
            per_benchmark[benchmark][f"{metric}_mean"] = float(values.mean())
            per_benchmark[benchmark][f"{metric}_std"] = float(values.std())
            per_benchmark[benchmark][f"{metric}_seeds"] = values.tolist()

    # Macro = mean over benchmarks, computed per seed first, so the reported std is
    # the spread across seeds (what varies) rather than across benchmarks.
    macro = {}
    for metric in metric_names:
        per_seed_macro = [
            float(np.mean([per_seed[str(seed)][b][metric] for b in benchmarks])) for seed in seeds
        ]
        macro[f"{metric}_mean"] = float(np.mean(per_seed_macro))
        macro[f"{metric}_std"] = float(np.std(per_seed_macro))
        macro[f"{metric}_seeds"] = per_seed_macro

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

    header = "benchmark    n_q  " + "  ".join(f"{m:^17}" for m in metric_names)
    print("\n" + header)
    for benchmark in benchmarks:
        row = per_benchmark[benchmark]
        cells = "  ".join(f"{row[m+'_mean']:.4f}+/-{row[m+'_std']:.4f}" for m in metric_names)
        print(f"{benchmark:12} {row['n_q']:4d}  {cells}")
    cells = "  ".join(f"{macro[m+'_mean']:.4f}+/-{macro[m+'_std']:.4f}" for m in metric_names)
    print(f"{'MACRO':12} {'':4}  {cells}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
