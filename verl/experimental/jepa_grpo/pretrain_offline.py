#!/usr/bin/env python3
"""Stage-1 offline JEPA + SIGReg pretraining on solved traces (official LLM-JEPA loss).

Two-stage recipe, stage 1: given (problem, solved-trace) pairs, train the LM backbone
so the predictor read p = Enc(problem + [PRED]) predicts the stopgrad embedding
z = Enc(trace) under the official flat-mean pull

    L = mean(1 - cos(p, z)) + lambda * SIGReg(p)

This is exactly the official repo loss (llm_jepa_paper_loss called with
weights=None / neg_weights=None => byte-for-byte the flat-mean behavior; no
advantage weighting, no repulsion/negative arm).

No RL, no rollout, no vLLM, no ray: a plain single-GPU PyTorch loop. Output is a
merged HF model dir (+ tokenizer with <|predictor_1|>..<|predictor_k|> + the
trainable predictor_delta) written every SAVE_INTERVAL steps and at the end, so
stage 2 (plain GRPO) can init from the `latest` dir via MODEL_PATH.

All settings come from env vars (set by run_two_stage_jepa_training.sh).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from verl.experimental.jepa_grpo.core_algos import llm_jepa_paper_loss

ENV = os.environ.get


def env_float(name: str, default: float) -> float:
    v = ENV(name)
    return float(v) if v is not None else default


def env_int(name: str, default: int) -> int:
    v = ENV(name)
    return int(v) if v is not None else default


def env_bool(name: str, default: bool) -> bool:
    v = ENV(name)
    return v.lower() in ("1", "true", "yes") if v is not None else default


def load_pairs(shard_paths: list[str]) -> list[tuple[int, str, str]]:
    """Load {index: [problem_text, trace_text]} shards (ds40k.cot.*.pt format).

    Shards are keyed by their own local row indices, which can overlap across
    files, so every shard after the first is offset by the cumulative size of
    the previous shards (indices stay globally unique, per-shard order kept).
    """
    pairs: dict[int, tuple[str, str]] = {}
    offset = 0
    for p in shard_paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        max_key = -1
        for idx, val in d.items():
            if len(val) < 2:
                continue
            problem, trace = val[0], val[1]
            if isinstance(problem, str) and isinstance(trace, str) and trace.strip():
                pairs.setdefault(offset + int(idx), (problem, trace))
                max_key = max(max_key, int(idx))
        offset += max_key + 1
    return sorted((i, a, b) for i, (a, b) in pairs.items())


def main() -> None:
    device = ENV("DEVICE", "cuda:0")
    model_path = ENV("MODEL", "/workspace/models/Qwen2.5-Math-1.5B-Instruct")
    shards = [s.strip() for s in ENV("DATA_SHARDS", "").split(",") if s.strip()]
    assert shards, "DATA_SHARDS must list the solved-trace shard .pt files (comma-separated)"
    out_dir = Path(ENV("OUT_DIR", "/workspace/checkpoints/jepa-pretrain/ds40k-2ep-lr1e-5"))
    metrics_path = ENV("METRICS_PATH", "")
    predictor_k = env_int("PREDICTOR_K", 1)
    sigreg_lambda = env_float("SIGREG_LAMBDA", 0.1)
    epochs = env_int("EPOCHS", 2)
    batch_size = env_int("BATCH_SIZE", 32)
    lr = env_float("LR", 1e-5)
    warmup_steps = env_int("WARMUP_STEPS", 50)
    max_prompt_len = env_int("MAX_PROMPT_LEN", 512)
    max_trace_len = env_int("MAX_TRACE_LEN", 3072)  # keep the TAIL of long traces
    grad_clip = env_float("GRAD_CLIP", 1.0)
    seed = env_int("SEED", 31415)
    save_interval = env_int("SAVE_INTERVAL", 100)
    log_interval = env_int("LOG_INTERVAL", 10)
    max_steps = env_int("MAX_STEPS", -1)
    max_pairs = env_int("MAX_PAIRS", -1)
    keep_saves = env_int("KEEP_SAVES", 2)
    log_file = ENV("LOG_FILE", "")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if log_file:
        log_handle = open(log_file, "w", buffering=1)
    else:
        log_handle = None

    def log(msg: str) -> None:
        line = f"[pretrain_offline] {msg}"
        print(line, flush=True)
        if log_handle is not None:
            log_handle.write(line + "\n")

    log(f"loading model+tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for i in range(1, predictor_k + 1):
        tokenizer.add_tokens([f"<|predictor_{i}|>"])
    pred_ids = [tokenizer.convert_tokens_to_ids(f"<|predictor_{i}|>") for i in range(1, predictor_k + 1)]
    pred_ids = list(reversed(pred_ids))  # append order k..1; last (read) token = <|predictor_1|>
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    emb_rows = model.get_input_embeddings().weight.shape[0]
    assert max(pred_ids) < emb_rows, (
        f"predictor token id {max(pred_ids)} >= embedding rows {emb_rows}: no spare rows"
    )

    # Trainable predictor-token embeddings, same init as the RL worker (randn*0.0066,
    # deterministic CPU seed 1234), added to the base embedding by a forward hook.
    g = torch.Generator(device="cpu").manual_seed(1234)
    hidden = model.get_input_embeddings().weight.shape[1]
    predictor_delta = torch.nn.Parameter(
        (torch.randn(len(pred_ids), hidden, generator=g) * 0.0066).to(device)
    )
    row_of = {tid: r for r, tid in enumerate(pred_ids)}
    embed_mod = model.get_input_embeddings()

    def _predictor_embed_hook(_mod, _inp, _out):
        input_ids = _inp[0]
        extra = None
        for tid, row in row_of.items():
            mask = input_ids == tid
            if bool(mask.any()):
                contrib = mask.unsqueeze(-1).to(_out.dtype) * predictor_delta[row].to(_out.dtype)
                extra = contrib if extra is None else extra + contrib
        return _out if extra is None else _out + extra

    embed_mod.register_forward_hook(_predictor_embed_hook)
    log(f"predictor tokens {pred_ids} (k={predictor_k}), delta shape {tuple(predictor_delta.shape)}")

    # Capture ONLY the last layer's hidden states (output_hidden_states=True would
    # retain all 28 layers' outputs: ~10GB+ per pass and the OOM source).
    last_layer = model.model.layers[-1]
    captured: dict[str, torch.Tensor] = {}

    def _capture_last_hidden(_mod, _inp, _out):
        captured["h"] = _out[0] if isinstance(_out, tuple) else _out
        return _out

    last_layer.register_forward_hook(_capture_last_hidden)

    log(f"loading solved-trace pairs from {len(shards)} shards")
    pairs = load_pairs(shards)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    log(f"{len(pairs)} solved traces")

    log("tokenizing (prompt head-truncated, trace tail-kept)")
    rows: list[tuple[int, list[int], list[int]]] = []
    for idx, problem, trace in pairs:
        p_ids = tokenizer(problem, add_special_tokens=False, truncation=True, max_length=max_prompt_len)["input_ids"]
        t_ids = tokenizer(trace, add_special_tokens=False)["input_ids"][-max_trace_len:]
        if not t_ids:
            continue
        rows.append((idx, p_ids + pred_ids, t_ids))
    log(f"{len(rows)} usable rows")
    rows.sort(key=lambda r: len(r[1]) + len(r[2]))
    n_steps_per_epoch = math.ceil(len(rows) / batch_size)
    total_steps = max_steps if max_steps > 0 else epochs * n_steps_per_epoch
    log(f"epochs={epochs} batch_size={batch_size} steps/epoch={n_steps_per_epoch} total_steps={total_steps}")

    params = list(model.parameters()) + [predictor_delta]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(warmup_steps, total_steps), num_training_steps=total_steps)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    trainable_ids = [i for i in range(len(rows))]
    saved: list[Path] = []
    step = 0
    global_start = time.time()

    def save_ckpt(step_: int, loss_: float) -> Path:
        d = out_dir / f"step_{step_:05d}"
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d)
        tokenizer.save_pretrained(d)
        torch.save({"predictor_delta": predictor_delta.data.detach().cpu().float(), "predictor_ids": pred_ids}, d / "predictor_delta.pt")
        manifest = {
            "step": step_, "loss": loss_, "lr": float(sched.get_last_lr()[0]),
            "model": model_path, "epochs": epochs, "batch_size": batch_size,
            "lr_init": lr, "sigreg_lambda": sigreg_lambda, "predictor_k": predictor_k,
            "max_prompt_len": max_prompt_len, "max_trace_len": max_trace_len,
            "n_rows": len(rows), "seed": seed, "shards": shards,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        latest = out_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(d.name, target_is_directory=True)
        saved.append(d)
        while len(saved) > keep_saves:
            rm = saved.pop(0)
            if rm.exists():
                log(f"pruning old save {rm}")
                import shutil
                shutil.rmtree(rm)
        return d

    model.train()
    block = max(1, 4 * batch_size)
    chunk_tokens = env_int("CHUNK_TOKENS", 180_000)

    def run_chunk(ch: list) -> tuple:
        b_anchor, b_target = [r[1] for r in ch], [r[2] for r in ch]
        a_max = max(len(x) for x in b_anchor)
        t_max = max(len(x) for x in b_target)
        a_ids = torch.full((len(ch), a_max), pad_id, dtype=torch.long)
        t_ids = torch.full((len(ch), t_max), pad_id, dtype=torch.long)
        for i, (a, t) in enumerate(zip(b_anchor, b_target)):
            a_ids[i, a_max - len(a):] = torch.tensor(a, dtype=torch.long)
            t_ids[i, t_max - len(t):] = torch.tensor(t, dtype=torch.long)
        a_ids, t_ids = a_ids.to(device), t_ids.to(device)

        with torch.no_grad():
            model(t_ids, use_cache=False)
            z = F.normalize(captured["h"][:, -1].float(), dim=-1).detach()
        model(a_ids, use_cache=False)
        p = F.normalize(captured["h"][:, -1].float(), dim=-1)

        return llm_jepa_paper_loss(p, z, sigreg_lambda=sigreg_lambda, target_view="cot")

    for epoch in range(epochs):
        rng = random.Random(seed + epoch * 7919)
        blocks = [rows[i : i + block] for i in range(0, len(rows), block)]
        rng.shuffle(blocks)
        order: list = []
        for bl in blocks:
            rng.shuffle(bl)
            order.extend(bl)
        for b in range(0, len(order), batch_size):
            if step >= total_steps:
                break
            t0 = time.time()
            batch = order[b : b + batch_size]
            total_tok = sum(len(r[1]) + len(r[2]) for r in batch)
            if total_tok > chunk_tokens and len(batch) > 1:
                chunks, cur, acc = [], [], 0
                for r in sorted(batch, key=lambda r: len(r[1]) + len(r[2]), reverse=True):
                    tl = len(r[1]) + len(r[2])
                    if cur and acc + tl > chunk_tokens:
                        chunks.append(cur)
                        cur, acc = [], 0
                    cur.append(r)
                    acc += tl
                if cur:
                    chunks.append(cur)
            else:
                chunks = [batch]
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            metrics = None
            for ch in chunks:
                loss, m = run_chunk(ch)
                (loss / len(chunks)).backward()
                loss_sum += float(loss.detach().cpu())
                metrics = m
            metrics["jepa/llm_jepa_loss"] = loss_sum / len(chunks)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            sched.step()

            step += 1
            dt = time.time() - t0
            if step % log_interval == 0 or step == 1 or step == total_steps:
                line = (
                    f"step {step}/{total_steps} | epoch {epoch + 1} | loss {metrics['jepa/llm_jepa_loss']:.4f} "
                    f"| cos {metrics['jepa/cos_pred_target']:.4f} | sigreg {metrics['jepa/paper_sigreg_loss']:.4f} "
                    f"| lr {float(sched.get_last_lr()[0]):.2e} | {dt:.1f}s/step | {total_tok / dt:.0f} tok/s | "
                    f"eta {(total_steps - step) * dt / 60:.1f}min"
                )
                log(line)
                if metrics_path:
                    with open(metrics_path, "a") as fh:
                        rec = {"step": step, "lr": float(sched.get_last_lr()[0]), "time_s": dt,
                               **{k.replace("jepa/", "jepa/"): v for k, v in metrics.items()}}
                        fh.write(json.dumps(rec) + "\n")
            if step % save_interval == 0:
                save_ckpt(step, float(metrics["jepa/llm_jepa_loss"]))

    final_dir = save_ckpt(step, float(loss.detach().cpu()))
    log(f"done: {step} steps in {(time.time() - global_start) / 60:.1f} min | final save {final_dir}")
    if log_handle is not None:
        log_handle.close()


if __name__ == "__main__":
    main()
