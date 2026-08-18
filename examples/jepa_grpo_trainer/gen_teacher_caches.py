#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate REAL verified teacher CoT + Code caches with one generator model.

Replaces precompute_teacher_targets.py for large train files. Three differences
that matter at 40k prompts:

  * DATA parallel, not tensor parallel. A 14B model fits in one 96GB card, so two
    independent single-GPU vLLM processes beat tp_size=2 (no cross-GPU collective
    per token). Launch one process per GPU with --shard i --num-shards N.
  * RESUMABLE. Results are flushed to the shard file every --flush-every prompts
    and completed dataset indices are skipped on restart. precompute_teacher_targets
    holds everything in memory and saves once at the end, so any interruption in a
    multi-hour run loses all of it.
  * BOTH VIEWS per model load, and the Code view is verified by ACTUALLY RUNNING
    the program (subprocess, stdout -> the same math checker), not by string-matching
    the source. Verifying code with compute_math_reward on the raw source only passes
    when the source happens to contain the answer, which selects for programs that
    hardcode it.

Output: two files per shard, each a {dataset_index (int): [text, ...]} dict in the
exact format jepa.teacher_cache_path / jepa.code_teacher_cache_path expect. Merge
the shards with --merge.
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE vllm is imported. Importing verl (for compute_math_reward)
# initialises CUDA in this process, and vLLM v1's default fork start method then
# fails with "Cannot re-initialize CUDA in forked subprocess" when it spawns its
# engine core. The execution pool below is unaffected -- its workers only shell
# out to a fresh interpreter and never touch CUDA.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

from verl.experimental.fepo.math_parser import compute_math_reward

# Kept identical to JEPARayConfig.code_system_prompt / precompute_teacher_targets.
CODE_SYSTEM_PROMPT = (
    "You are a Python programming expert. "
    "Solve the following math problem by writing a complete, executable Python program "
    "that prints the answer. Do not include any natural language explanation outside comments."
)

# Appended to the question itself (not as a system turn, which would stack ahead of
# the dataset's own framing). Targets are capped at --*-max-tokens; a generation that
# hits the cap is unusable as a teacher view -- it is truncated mid-reasoning and can
# never contain a verifiable answer -- so ask for brevity rather than paying for
# tokens that get thrown away.
CONCISE_COT = (
    "\n\nSolve concisely: give only the essential reasoning steps, no restatement of "
    "the problem and no exploration of dead ends, then the final answer in \\boxed{}."
)
CONCISE_CODE = (
    "\n\nKeep the program short and direct: compute the answer and print it, "
    "with no exploratory output."
)

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

_T0 = time.time()


def _hms(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "?"
    s = int(max(0, seconds))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _extract_code(text: str) -> str:
    m = _FENCE.findall(text)
    return m[-1] if m else text


def _run_code(args) -> bool:
    """Execute a candidate program and math-check its stdout. Used in a subprocess pool.

    Returns True only if the program RUNS and prints something the math checker
    accepts. A program that merely contains the answer in a comment fails here,
    which is the point: it filters out answer-hardcoding that source-text matching
    would happily keep.
    """
    text, ground_truth, dataset_kind, timeout = args
    code = _extract_code(text)
    if not code.strip():
        return False
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], input="", capture_output=True,
            text=True, timeout=timeout,
        )
    except Exception:
        return False
    out = (proc.stdout or "").strip()
    if not out:
        return False
    # Compare the printed value with the SAME checker the CoT view uses, so
    # 34 / 34.0 / \boxed{34} all agree.
    try:
        return bool(compute_math_reward(out, ground_truth, dataset_kind=dataset_kind).is_correct)
    except Exception:
        return False


def _messages(prompt, system_prompt: str, concise: str = "") -> list[dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in prompt]
    if concise:
        # Append to the LAST user turn so the instruction lands with the question.
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] = m["content"] + concise
                break
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--out-prefix", required=True,
                   help="Shard files are <prefix>.cot.shard<i>.pt / <prefix>.code.shard<i>.pt")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=4, help="Samples drawn per prompt per view")
    p.add_argument("--n-targets", type=int, default=2, help="Max VERIFIED targets kept per prompt per view")
    p.add_argument("--max-rows", type=int, default=-1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--cot-max-tokens", type=int, default=3072)
    p.add_argument("--code-max-tokens", type=int, default=3072)
    p.add_argument("--gpu-mem-frac", type=float, default=0.90)
    p.add_argument("--chunk-prompts", type=int, default=512,
                   help="Prompts per generate() call; also the resume/flush granularity")
    p.add_argument("--exec-timeout", type=float, default=6.0)
    p.add_argument("--exec-workers", type=int, default=16)
    p.add_argument("--retry-rounds", type=int, default=2,
                   help="Extra passes over prompts still short of --n-targets. 0 = single pass.")
    p.add_argument("--retry-temp-step", type=float, default=0.15,
                   help="Temperature added per retry round (redrawing at the same T mostly repeats the same failure)")
    p.add_argument("--retry-extra-samples", type=int, default=2,
                   help="Extra samples per prompt added per retry round")
    p.add_argument("--views", default="cot,code")
    p.add_argument("--merge", action="store_true", help="Merge <prefix>.*.shard*.pt into final caches and exit")
    return p.parse_args()


def _load(path: str) -> dict:
    if os.path.exists(path):
        try:
            return {int(k): list(v) for k, v in torch.load(path, map_location="cpu", weights_only=False).items()}
        except Exception:
            pass
    return {}


def do_merge(args) -> None:
    import glob
    for view in args.views.split(","):
        merged: dict[int, list[str]] = {}
        shards = sorted(glob.glob(f"{args.out_prefix}.{view}.shard*.pt"))
        for s in shards:
            merged.update(_load(s))
        out = f"{args.out_prefix}.{view}.responses.pt"
        torch.save(merged, out)
        n_t = sum(len(v) for v in merged.values())
        print(f"[merge] {view}: {len(shards)} shards -> {len(merged)} prompts, "
              f"{n_t} targets ({n_t / max(1, len(merged)):.2f}/prompt) -> {out}")


def main() -> None:
    args = parse_args()
    if args.merge:
        do_merge(args)
        return

    df = pd.read_parquet(args.train_file)
    if args.max_rows > 0:
        df = df.iloc[: args.max_rows]
    rows = df.to_dict("records")
    # Strided shards so each process sees the same difficulty mix (contiguous
    # slices of a sorted-by-source parquet would not be comparable).
    rows = rows[args.shard :: args.num_shards]
    views = [v for v in args.views.split(",") if v]
    print(f"[gen] shard {args.shard}/{args.num_shards}: {len(rows)} prompts, views={views}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem_frac, trust_remote_code=True)

    pool = ProcessPoolExecutor(max_workers=args.exec_workers)
    for view in views:
        out_path = f"{args.out_prefix}.{view}.shard{args.shard}.pt"
        cache = _load(out_path)
        # A prompt is "done" only once it has n_targets verified targets; one that
        # came back with a single correct sample is still eligible for a retry round.
        todo = [r for r in rows if len(cache.get(int(r["extra_info"]["index"]), [])) < args.n_targets]
        print(f"[gen] view={view}: {len(cache)} solved on disk, {len(todo)} short of "
              f"{args.n_targets} targets -> {out_path}", flush=True)
        sys_prompt = CODE_SYSTEM_PROMPT if view == "code" else ""
        concise = CONCISE_CODE if view == "code" else CONCISE_COT
        max_tok = args.code_max_tokens if view == "code" else args.cot_max_tokens

        def _verify_chunk(chunk, gen) -> None:
            """Fold newly verified samples into `cache`, never exceeding n_targets.

            APPENDS rather than overwrites: a retry round must be able to top up a
            prompt that already has one correct target from an earlier round.
            """
            if view == "code":
                # Execution dominates wall time here, so run every candidate of the
                # whole chunk through the pool at once rather than per prompt.
                jobs, owner = [], []
                for r, g in zip(chunk, gen):
                    for o in g.outputs:
                        jobs.append((o.text, r["reward_model"]["ground_truth"],
                                     r.get("data_source"), args.exec_timeout))
                        owner.append(int(r["extra_info"]["index"]))
                flags = list(pool.map(_run_code, jobs, chunksize=4))
                for (text, *_), idx, ok in zip(jobs, owner, flags):
                    if ok and len(cache.setdefault(idx, [])) < args.n_targets:
                        cache[idx].append(text)
            else:
                for r, g in zip(chunk, gen):
                    idx = int(r["extra_info"]["index"])
                    gt, kind = r["reward_model"]["ground_truth"], r.get("data_source")
                    for o in g.outputs:
                        if len(cache.setdefault(idx, [])) >= args.n_targets:
                            break
                        try:
                            ok = compute_math_reward(o.text, gt, dataset_kind=kind).is_correct
                        except Exception:
                            ok = False
                        if ok:
                            cache[idx].append(o.text)
            # Drop keys that ended up with nothing so `cache` stays a solved-only map.
            for k in [k for k, v in cache.items() if not v]:
                del cache[k]

        pending = todo
        for rnd in range(args.retry_rounds + 1):
            if not pending:
                break
            # Retries raise temperature: the first round already sampled this prompt
            # at args.temperature, so redrawing from the same distribution mostly
            # repeats the same failure. Also widen n on retries, since the prompts
            # that survive to round N are by construction the hard ones.
            temp = args.temperature + rnd * args.retry_temp_step
            n = args.n_samples + rnd * args.retry_extra_samples
            sampling = SamplingParams(n=n, temperature=temp, top_p=args.top_p, max_tokens=max_tok)
            tag = "gen" if rnd == 0 else f"retry{rnd}"
            print(f"[{tag}] {view} shard{args.shard}: {len(pending)} prompts, n={n}, T={temp:.2f}",
                  flush=True)
            t_round = time.time()
            for start in range(0, len(pending), args.chunk_prompts):
                chunk = pending[start : start + args.chunk_prompts]
                texts = [
                    tok.apply_chat_template(_messages(r["prompt"], sys_prompt, concise),
                                            tokenize=False, add_generation_prompt=True)
                    for r in chunk
                ]
                _verify_chunk(chunk, llm.generate(texts, sampling))
                torch.save(cache, out_path)
                done = start + len(chunk)
                n_t = sum(len(v) for v in cache.values())
                full = sum(1 for v in cache.values() if len(v) >= args.n_targets)
                rate = done / max(1e-9, time.time() - t_round)      # prompts/s this round
                eta = (len(pending) - done) / rate if rate > 0 else float("nan")
                print(f"[{tag}] {view} shard{args.shard}: {done}/{len(pending)} | "
                      f"{len(cache)} solved ({full} at {args.n_targets}) | {n_t} targets | "
                      f"{n_t / max(1, len(cache)):.2f}/solved | "
                      f"{_hms(time.time() - _T0)} elapsed | round ETA {_hms(eta)}", flush=True)
            before = len(pending)
            pending = [r for r in pending
                       if len(cache.get(int(r["extra_info"]["index"]), [])) < args.n_targets]
            print(f"[{tag}] {view} shard{args.shard}: round done in {_hms(time.time() - t_round)}, "
                  f"{before - len(pending)} prompts reached {args.n_targets}, "
                  f"{len(pending)} still short | {_hms(time.time() - _T0)} total", flush=True)
    pool.shutdown()
    print(f"[gen] shard {args.shard} complete", flush=True)


if __name__ == "__main__":
    main()
