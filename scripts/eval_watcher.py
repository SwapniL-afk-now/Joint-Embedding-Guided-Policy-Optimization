#!/usr/bin/env python3
"""Score training checkpoints on a dedicated GPU, asynchronously from the trainer.

The trainer (GPU0) runs with test_freq=-1 -- it never validates in-loop -- and just writes
`global_step_N/actor/huggingface/` every save_freq steps. This watcher (GPU1) polls for those,
scores each with the existing eval_math_multiseed_batched.py, logs to W&B, and keeps the best
weights. Inline validation cost 330.6s against a 54.7s train step, so moving it here takes
~38% off wall-clock and puts the otherwise-idle second GPU to work.

Retention: two independent champions, `best_pass_at_1` and `best_pass_at_k`. A stored
checkpoint is deleted ONLY when a strictly better one replaces it, so at most two weight sets
(~3.1G each) are ever kept. Every score is appended to results.jsonl regardless of whether its
weights are retained -- that file, not the weights, is the record the slope is computed from.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_watcher.py \
        --ckpt-dir checkpoints/grpo-qwen-3b/<exp> --exp <exp>
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "eval_math_multiseed_batched.py"
BENCHMARKS = ["aime24", "aime25", "aime26", "amc23"]
SEEDS = [31415, 27182, 16180]
# eval_math_multiseed_batched.py reports pass@1 = correct[:,0].mean() -- the FIRST sample only.
# That is exactly verl's own val/*/pass_at_1 (`any(vals[:1])`, ray_trainer.py:924), so it is the
# metric that matches every previous run. avg@8 uses the same 8 samples but all of them, so it
# has ~1/8 the variance; it is logged alongside and is the better statistic for a slope test.
HEADLINE, LOWVAR, PASSK = "pass@1", "avg@8", "pass@8"


def discover(ckpt_dir: Path, done: set[int]) -> list[tuple[int, Path]]:
    """Completed, unscored checkpoints, oldest first. Partial writes are skipped."""
    out = []
    for d in ckpt_dir.glob("global_step_*"):
        try:
            step = int(d.name.split("_")[-1])
        except ValueError:
            continue
        if step in done:
            continue
        hf = d / "actor" / "huggingface"
        # A checkpoint mid-write has the dir (tokenizer/config land first) but no weights yet.
        if hf.is_dir() and any(hf.glob("*.safetensors")):
            out.append((step, hf))
    return sorted(out)


def run_eval(model: Path, out_json: Path, n: int, max_tokens: int, gpu_mem: float) -> dict | None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EVAL),
        "--model", str(model),
        "--out", str(out_json),
        "--benchmarks", ",".join(BENCHMARKS),
        "--seeds", ",".join(map(str, SEEDS)),
        "--n", str(n),
        "--max-tokens", str(max_tokens),
        "--temperature", "0.6",
        "--top-p", "0.95",
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    print(f"[eval] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"[eval] FAILED rc={r.returncode}", flush=True)
        return None
    return json.loads(out_json.read_text()) if out_json.exists() else None


def flatten(res: dict) -> dict:
    """eval_math_multiseed_batched.py's JSON -> flat wandb keys.

    That script emits {"per_benchmark": {bench: {"<metric>_mean"/"_std"/"_seeds": ...}},
    "macro": {metric: mean-across-benchmarks}, ...}. `macro` is already the 4-benchmark
    mean, so it is used directly rather than recomputed.

    The @16 metrics are dropped when n < 16: summarize() slices correct[:, :16], so with
    n=8 avg@16 is numerically identical to avg@8 and would just be a confusing duplicate.
    """
    n = int(res.get("n", 0))
    keep = [m for m in ("pass@1", "avg@8", "pass@8", "avg@16", "pass@16")
            if n >= 16 or not m.endswith("@16")]
    flat = {}
    for bench, row in res.get("per_benchmark", {}).items():
        for m in keep:
            if f"{m}_mean" in row:
                key = m.replace("@", "_at_")
                flat[f"val/{bench}/{key}"] = float(row[f"{m}_mean"])
                flat[f"val/{bench}/{key}_std"] = float(row[f"{m}_std"])
    for m in keep:
        if m in res.get("macro", {}):
            flat[f"val/mean/{m.replace('@', '_at_')}"] = float(res["macro"][m])
    return flat


def promote(src: Path, store: Path, step: int, score: float, current: dict | None) -> dict:
    """Copy src into store as the new champion and drop the previous one."""
    tmp = store.parent / f".{store.name}.incoming"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    shutil.rmtree(store, ignore_errors=True)          # delete the beaten champion
    tmp.rename(store)
    (store / "CHAMPION.json").write_text(json.dumps({"step": step, "score": score}, indent=2))
    prev = f" (was step {current['step']} @ {current['score']:.4f})" if current else ""
    print(f"[keep] {store.name}: step {step} @ {score:.4f}{prev}", flush=True)
    return {"step": step, "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--exp", required=True, help="training experiment name (wandb group)")
    ap.add_argument("--project", default="grpo-qwen-3b")
    ap.add_argument("--store", default=None, help="default: <ckpt-dir>/../<exp>-eval-store")
    ap.add_argument("--n", type=int, default=8, help="samples/prompt; 8 matches previous runs")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--base-model", default=None,
                    help="score this as step 0 first (the untrained baseline)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    store = Path(args.store) if args.store else ckpt_dir.parent / f"{args.exp}-eval-store"
    store.mkdir(parents=True, exist_ok=True)
    results_path = store / "results.jsonl"

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.project, name=f"{args.exp}-eval", group=args.exp,
                         job_type="eval", config=vars(args))

    done: set[int] = set()
    champs: dict[str, dict | None] = {"best_pass_at_1": None, "best_pass_at_k": None}
    if results_path.exists():                      # resume without rescoring
        for line in results_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["step"])
        print(f"[init] resuming, {len(done)} steps already scored", flush=True)
    for name in champs:
        meta = store / name / "CHAMPION.json"
        if meta.exists():
            champs[name] = json.loads(meta.read_text())

    pending = []
    if args.base_model and 0 not in done:
        pending.append((0, Path(args.base_model)))

    print(f"[init] watching {ckpt_dir} -> store {store}", flush=True)
    while True:
        for step, hf in pending + discover(ckpt_dir, done):
            if step in done:
                continue
            print(f"\n[step {step}] scoring {hf}", flush=True)
            t0 = time.time()
            res = run_eval(hf, store / "raw" / f"step_{step}.json",
                           args.n, args.max_tokens, args.gpu_memory_utilization)
            done.add(step)
            if res is None:
                with results_path.open("a") as fh:
                    fh.write(json.dumps({"step": step, "failed": True}) + "\n")
                continue

            flat = flatten(res)
            elapsed = time.time() - t0
            row = {"step": step, "elapsed_s": round(elapsed, 1), **flat}
            with results_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            if run is not None:
                run.log({**flat, "eval/elapsed_s": elapsed}, step=step)

            h = flat.get(f"val/mean/{HEADLINE.replace('@', '_at_')}")
            lv = flat.get(f"val/mean/{LOWVAR.replace('@', '_at_')}")
            pk = flat.get(f"val/mean/{PASSK.replace('@', '_at_')}")
            print(f"[step {step}] pass@1 {h:.4f}  avg@8 {lv:.4f}  pass@8 {pk:.4f}  ({elapsed:.0f}s)",
                  flush=True)

            # Retention: promote only on a strict improvement, so a stored checkpoint is
            # removed only when a better one takes its place.
            for name, score in (("best_pass_at_1", h), ("best_pass_at_k", pk)):
                if score is None:
                    continue
                cur = champs[name]
                if cur is None or score > cur["score"]:
                    if hf.exists():
                        champs[name] = promote(hf, store / name, step, score, cur)
                    else:      # trainer rotated it away before we could copy
                        print(f"[keep] step {step} beat {name} but weights are gone; "
                              f"score kept in results.jsonl", flush=True)
        pending = []
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
