import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


EVAL_DIR = Path("/workspace/jepa-grpo-cache/eval_data")
WORKER = Path(__file__).with_name("code_judge_worker.py")
DEFAULT_BENCHMARKS = ["humanevalplus", "mbppplus", "livecodebench"]
DEFAULT_SEEDS = [31415, 27182, 16180]
MAXLEN = 4096
JOB_TIMEOUT = 30
FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def passk(c, n, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def extract_code(text):
    matches = FENCE.findall(text)
    return matches[-1] if matches else text


def make_job(ds, text, ground_truth):
    spec = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    completion = extract_code(text)
    if ds in ("humanevalplus", "mbppplus"):
        entry_point = spec["entry_point"]
        code = completion if f"def {entry_point}" in completion else f"{spec.get('prompt', '')}\n{completion}"
        return {
            "kind": "assert",
            "code": code,
            "test": spec["test"],
            "entry_point": entry_point,
        }
    return {
        "kind": "stdio",
        "code": completion,
        "inputs": list(spec["inputs"]),
        "outputs": list(spec["outputs"]),
    }


def score_one(ds, text, ground_truth, extra_info):
    try:
        job = make_job(ds, text, ground_truth)
    except Exception:
        return 0
    try:
        result = subprocess.run(
            [sys.executable, str(WORKER)],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=JOB_TIMEOUT,
        )
        return 1 if result.returncode == 0 else 0
    except Exception:
        return 0


def summarize(correct):
    metrics = {"pass@1": float(correct[:, 0].mean())}
    if correct.shape[1] >= 8:
        first8 = correct[:, :8]
        metrics["avg@8"] = float(first8.mean())
        metrics["pass@8"] = float(np.mean([passk(int(c), 8, 8) for c in first8.sum(axis=1)]))
    if correct.shape[1] >= 16:
        first16 = correct[:, :16]
        metrics["avg@16"] = float(first16.mean())
        metrics["pass@16"] = float(np.mean([passk(int(c), 16, 16) for c in first16.sum(axis=1)]))
    return metrics


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
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    prompt_lens = []
    meta = []
    spans = {}
    for benchmark in benchmarks:
        df = pd.read_parquet(EVAL_DIR / f"{benchmark}.parquet")
        start = len(prompts)
        for _, row in df.iterrows():
            text = tokenizer.apply_chat_template(
                list(row["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)
            prompt_lens.append(len(tokenizer(text)["input_ids"]))
            extra_info = row["extra_info"]
            if hasattr(extra_info, "keys"):
                extra_info = dict(extra_info)
            meta.append((row["data_source"], row["reward_model"]["ground_truth"], extra_info))
        spans[benchmark] = (start, len(prompts))

    per_max = [max(256, min(args.max_tokens, MAXLEN - prompt_len - 8)) for prompt_len in prompt_lens]
    n_capped = sum(1 for value in per_max if value < args.max_tokens)

    batched_prompts = []
    batched_meta = []
    sampling_params = []
    for seed in seeds:
        for prompt, item_meta, max_tokens in zip(prompts, meta, per_max):
            batched_prompts.append(prompt)
            batched_meta.append(item_meta)
            sampling_params.append(
                SamplingParams(
                    n=args.n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            )

    print(
        f"loaded {len(prompts)} prompts across {len(benchmarks)} code benchmarks; "
        f"{n_capped} prompts capped below {args.max_tokens}; evaluating {len(seeds)} seeds "
        f"together as {len(batched_prompts)} vLLM requests, n={args.n}"
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAXLEN,
        max_num_seqs=args.max_num_seqs,
    )
    outputs = llm.generate(batched_prompts, sampling_params)

    jobs = []
    for output, (ds, ground_truth, extra_info) in zip(outputs, batched_meta):
        for sample in output.outputs[: args.n]:
            jobs.append((ds, sample.text, ground_truth, extra_info))

    workers = os.cpu_count() or 16
    print(f"scoring {len(jobs)} generated programs with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = list(pool.map(lambda item: score_one(*item), jobs))
    correct = np.asarray(flags, dtype=bool).reshape(len(batched_prompts), args.n)

    per_seed = {}
    for seed_idx, seed in enumerate(seeds):
        offset = seed_idx * len(prompts)
        seed_correct = correct[offset : offset + len(prompts)]
        per_seed[str(seed)] = {}
        for benchmark, (start, end) in spans.items():
            per_seed[str(seed)][benchmark] = summarize(seed_correct[start:end])

    metric_names = list(per_seed[str(seeds[0])][benchmarks[0]].keys())
    per_benchmark = {}
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

    print("\nbenchmark          " + "        ".join(metric_names))
    for benchmark in benchmarks:
        row = per_benchmark[benchmark]
        values = [f"{row[f'{metric}_mean']:.4f}+/-{row[f'{metric}_std']:.4f}" for metric in metric_names]
        print(f"{benchmark:16} " + "  ".join(values))
    print(f"{'MACRO':16} " + "        ".join(f"{macro[metric]:.4f}" for metric in metric_names))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
