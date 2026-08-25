# SDC implementation bottlenecks and improvement roadmap

Consolidated from six independent code audits plus manual review, at commit `a7f2eaa` (branch `sdc`).
Constraint honored throughout: **the locked loss algebra is never changed**
`beta * sum(ratio * (logp_F - logp_S)) / active_failed_token_count`, references detached,
token-mean normalization. Every item below is classified by exactness:

- **BIT**: bit-exact (copies, scheduling, control flow, dead code)
- **ULP**: mathematically identical; final scalar sums may differ <= 1 ulp from reduction-order changes
- **TOL**: visible numeric change; requires an explicit empirical gate (flagged, off by default)

---

## 0. Correctness must-fixes (do first; block silent corruption / hangs)

| # | Issue | Where | Fix |
|---|-------|-------|-----|
| C1 | Non-rank-0 vLLM servers return `[]`; client writes exact-zero success/failure logprobs silently -> corrupts SDC term | `vllm_async_server.py:622-623`, `ray_trainer.py:_sdc_rows_to_tensor` | Raise RuntimeError instead of returning `[]`; assert per-row width equals expected response length |
| C2 | naive/dapo reward managers fall back to shaped reward (`score.get("acc", reward)`); partial credit (e.g. 0.5) poisons `sdc_outcome` and crashes the run hours in via `validate_sdc_outcome` | `reward_manager/naive.py:95`, `dapo.py` equivalent | Populate `sdc_outcome` only from explicit acc/correct keys; leave unset otherwise so failure is fast and clear |
| C3 | If rank 0's `torch.save` fails inside `_sdc_write_checkpoint`, ranks 1..N hang forever at the barrier | `transformer_impl.py:1149-1192` | try/except around rank-0 writes; broadcast a status object before the barrier |

## 1. Tier 1 -- highest wall-clock impact

### T1. Stop embedding the outcome buffer in every per-step actor batch [BIT]
`_sdc_config_dict()` embeds `success_buffer_state` + `failure_buffer_state` (up to 8192
records x ~4k token IDs as Python lists). Rebuilt and pickled EVERY training step and shipped
to every actor rank; the loss reads only 3 scalars from this dict.
Fix: `include_buffer_states=False` flag; pass True only at checkpoint save/load call sites.
(`ray_trainer.py:2140-2191, 2082-2083`)

### T2. Sidecar residency policy: kill the per-step GPU<->CPU ping-pong [BIT]
Every scoring call moves BOTH full clones H2D then D2H + `empty_cache()` (device-wide sync);
4 x model-size PCIe transfers per step. `_sdc_prepare_models()` is even called twice per scoring step.
Fix: keep clones resident while vLLM is asleep (config-gated); move `empty_cache()` to the
sleep/wake transition only; call prepare once per scoring pass. Optional: prefetch the failure
clone on a side stream during the success forward. (`transformer_impl.py:832-851, 869-901, 990-1006`)

### T3. Ship only the consumed record prefix for SFT refresh [BIT]
Every 5 steps ALL buffered records go to EVERY rank (`ONE_TO_ALL`); each discards all but the
deterministic 8-record prefix that rank 0 trains on. Fix: slice trainer-side using the same
prefix rule as `_sdc_match_records`. (`ray_trainer.py:2334-2336`, `engine_workers.py:764-767`)

### T4. Overlap vLLM scoring with the rest of the step [BIT]
Scoring blocks synchronously between old-log-prob and ref-policy work, but its inputs are fully
known right after DAPO annotation. Fix: launch scoring as an asyncio task right after
`_sdc_annotate_actor_batch`, await at the current consumption site. Same inputs -> same outputs.
(`ray_trainer.py:2617, 2741-2742`)

### T5. Loss hot-path cleanup [BIT except where noted]
Per micro-batch inside `compute_sdc_loss`:
- ~10 blocking `.item()`/`.cpu()` syncs BEFORE backward; Metric would convert later anyway.
  Fix: one stacked detached tensor, single `.cpu()`, or convert once per fwd/bwd batch.
- Full pipeline runs even when the micro-batch has zero failed rows; early-exit is only checked
  AFTER building everything. Fix: host-side check first (precomputed count already known).
- Dead `count = local_count.clone()` whenever the precomputed global count exists.
- Gather-active-rows rewrite (`rows = failure_mask.nonzero()` once, then operate on selected rows):
  shrinks all elementwise ops AND the ~5 autograd-saved [bsz, resp_len] fp32 buffers to n_active
  rows. [ULP]
(`sdc_loss.py:86-201`, `losses.py:140-162`)

## 2. Tier 2 -- solid mid-size wins

### T6. vLLM scoring RPC shape [BIT]
- One Ray task per (sequence, adapter) = 2N tasks; sequential `_acquire_server` round-trips
  before submission starts. Fix: concurrent acquire / acquire-once-per-server; batched
  server-side method scoring many sequences per task.
- Server returns prompt-logprobs for the WHOLE sequence; response span sliced driver-side.
  Fix: slice server-side before return (byte-identical floats, payload ~ total_len/response_len smaller).
- `vllm_score_micro_batch_size` is validated but never used -- wire it up or delete it.
(`llm_server.py:283-337`, `vllm_async_server.py:611-663`, `vllm_scoring.py:29-41`)

### T7. Vectorize tensor marshalling [BIT]
`_sdc_build_vllm_score_inputs` and `_sdc_collect_outcomes` duplicate identical per-row mask +
tolist work twice per step; `_sdc_rows_to_tensor` does N tiny H2D copies.
Fix: compute lengths/active indices once per step and share; build one pinned CPU tensor,
single non-blocking transfer. (`ray_trainer.py:2229-2320`)

### T8. Chunked scoring forwards [BIT]
HF scoring runs one dense forward over ALL active rows padded to global max length ->
activation spike / OOM risk against the colocated vLLM reserve. Row-independent no-grad
forwards chunk along batch dim give bitwise-identical per-row logprobs.
(`transformer_impl.py:989-1007`)

### T9. Sidecar gradient checkpointing [BIT]
Sidecars inherit actor GC (`_build_module` enables it globally) so their SFT backward pays
activation recomputation although VRAM is free while vLLM sleeps. Recompute-or-not is
numerically identical. Config-gate GC for sidecars. (`transformer_impl.py:318-327`)

### T10. Checkpoint slimming [BIT]
Full mode writes ~7x model bytes per checkpoint (2 models + 2 optimizer states + a THIRD full
model: the immutable base snapshot). Fixes:
- Drop the embedded base snapshot; `sdc_load` already re-clones it from the restored actor.
  Guard with a stored-step equality check. Saves 1xN per checkpoint.
- `torch.load` `sdc_state.pt` on rank 0 only (nonzero ranks currently deserialize then discard).
- Save the base snapshot once ever (it never changes), not every interval.
- Deduplicate best-checkpoint writes via copytree like the actor path.
- Remove the redundant post-save `_sdc_sync_vllm_adapters()` on steps where the SFT refresh
  already synced (every checkpoint step under current intervals).
(`transformer_impl.py:1149-1192, 1477-1497`, `sdc_state.py:47-73`, `ray_trainer.py:1568-1590, 1675`)

### T11. Sync/broadcast path [BIT]
- Layout mismatch guard uses `all_gather_object` over thousands of tuples PER refresh.
  Fix: order-sensitive hash compared via one tiny tensor collective; validate layout once, cache.
- Persistent pinned staging buffer instead of fresh 64 MB `torch.cat` allocations per flush;
  consider raising chunk bytes to 256 MB on NVLink systems.
(`transformer_impl.py:1046-1147`)

### T12. Base mixing staging [BIT]
Non-LoRA mixing streams the entire frozen base CPU->GPU per tensor, twice (both branches),
with fp32 temporaries. Fix: pre-stage the immutable base on GPU once, or mix on CPU and issue
one bulk transfer; keep exact per-tensor fp32 accumulate order. (`sdc_state.py:23-52`)

### T13. Adapter export fan-out [BIT]
Rank 0 ships the same LoRA tensor dict through Ray per replica; trainer also retains the payload
forever. Fix: ray.put once / shared path via LoRARequest; `del` the retained reference.
Skip export when update counts unchanged. (`llm_server.py:473-491`, `ray_trainer.py:2198-2215`)

## 3. Tier 3 -- structural (biggest wins, biggest diffs)

### T14. De-duplicate FSDP1/FSDP2 SDC lifecycle code [BIT]
~291 byte-identical lines duplicated across engine classes; drift ALREADY happened
(AdamW `step` placement differs between `_sdc_move_optimizer_state` implementations).
Extract an SdcEngineMixin FIRST so every other fix lands once. Delete dead
`_sdc_ensure_optimizer_state`.

### T15. Shared frozen backbone + two named LoRA adapters
Replaces two full replicated sidecars with one backbone + adapter-sized state everywhere:
mixing, sync, checkpoints, exports become KB-MB operations. This is Phase 2 of
SDC_FSDP_PERFORMANCE_AUDIT_AND_PLAN.md and removes P0-1/P0-3/T11-class costs structurally.

### T16. Real FSDP2-sharded sidecars for full fine-tuning
For workloads that require full-parameter S/F updates (Phase 3 of the repo plan):
sidecar-safe `_build_fsdp_module` variant, sharded init, distributed checkpoints,
delete the post-SFT broadcast entirely. Correctness gate: distributed SFT step must match the
single-rank reference within BF16 tolerance.

### T17. Buffer storage as tensors [BIT]
Store `prompt_ids`/`response_ids` as numpy arrays / torch tensors instead of Python int lists
(~10x per-token overhead cut); make `data_max_size=None` loud-warn or default finite;
cache incremental min-step for `buffer_age` instead of scanning metadata dicts each step.

## 4. Explicitly rejected (would change numerics)
- Storing the two reference logprob tensors as bf16: truncation shifts the contrast term at
  ~2^-8 relative error. Keep fp32. (Would be a TOL-level experiment behind a flag at most.)
- Any change to denominator semantics, clamp values, or aggregation mode: locked.

## 5. Suggested execution order
1. C1-C3 correctness guards (small diffs, protect the run)
2. T14 dedup mixin (so subsequent fixes land once)
3. T1, T3 (stop moving buffers through Ray) + T5 (hot-path syncs)
4. T2 residency + T9 sidecar GC (measure together; both free VRAM headroom)
5. T4 overlap + T6 RPC shape (only matters once vLLM backend is enabled)
6. T7-T13 incremental
7. T15-T16 architecture per profiling evidence, following the benchmark matrix in
   SDC_FSDP_PERFORMANCE_AUDIT_AND_PLAN.md (no speedup claims without Phase 0 baseline timers)
