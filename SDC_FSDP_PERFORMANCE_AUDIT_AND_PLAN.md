# SDC FSDP performance audit and implementation plan

Date: 2026-08-24
Branch audited: `sdc` (`origin/agent/drgrpo-jepa-training-diagnostics`)
Audit type: static source audit; no GPU benchmark was run for this report.

## Executive conclusion

The largest SDC cost is not a missing attention kernel. The SDC sidecars bypass
the actor's FSDP wrapping, so every actor rank owns two complete success/failure
models and two optimizers. Every rank then receives the same SFT records, runs
the same success and failure updates serially, and finally discards all but rank
0 by broadcasting both complete model states and optimizer states tensor by
tensor.

This is especially wasteful in LoRA mode: the synchronization and checkpoint
paths still traverse or save the frozen backbones, although only adapter weights
can change. Base mixing also streams the complete frozen base from CPU to GPU for
both branches after each SFT refresh.

The recommended fast architecture is:

1. one frozen SDC backbone with two named LoRA adapters;
2. data-parallel adapter training, or one FSDP2-sharded shared backbone when the
   backbone cannot remain replicated;
3. adapter-only base mixing, synchronization, vLLM export, and checkpointing;
4. no diagnostic collectives inside actor microbatches;
5. token-ID replay buffers with no decode/re-tokenize cycle.

For experiments that require exact full-parameter S/F fine-tuning, both
sidecars should become real FSDP/FSDP2 modules, consume different data shards,
and use distributed checkpoints. The current replicated sidecar path should
not remain the production full-fine-tuning implementation.

No speedup percentage should be claimed before profiling. The go/no-go target
for this work is at least a 50% reduction in SDC-added wall time and at least a
20% end-to-end step-time reduction on the target workload, provided SDC is at
least 40% of the measured baseline step.

## Current critical path

```text
rollout generation
  -> sleep vLLM
  -> optional old-log-prob forward
  -> score failures under success model
  -> score failures under failure model
  -> actor forward/backward
  -> every N steps:
       success SFT on every rank
       failure SFT on every rank
       CPU-base mix of both complete models
       broadcast both complete models and optimizer states
       export/reload both vLLM adapters
       periodically save two models + two optimizers + full base snapshot
```

The existing placement of vLLM sleep before actor work and conditional reuse of
rollout log-probabilities are good and must be preserved.

## Audit findings

| Priority | Finding | Evidence | Performance consequence |
|---|---|---|---|
| P0 | S/F sidecars are not FSDP-wrapped | `_sdc_clone_from_state()` calls `_build_module()` and moves it directly to the device (`verl/workers/engine/fsdp/transformer_impl.py:680-696` and the active LM-head override at `1599-1615`). The actor's wrapper is `_build_fsdp_module()` at `355-444`, but the sidecars never call it. | Two complete model replicas and two optimizer states live on every actor GPU. Actor FSDP does not shard this memory. |
| P0 | SFT work is redundantly repeated on every rank | The worker RPC is `Dispatch.ONE_TO_ALL` (`verl/workers/engine_workers.py:764-767`). Success and failure SFT execute serially (`transformer_impl.py:1857-1872`). | N ranks repeat the same records instead of collaborating on one distributed update; dropout can make the discarded nonzero-rank results differ. |
| P0 | Full state is broadcast after every paired refresh | `_sdc_sync_model_and_optimizer()` broadcasts every parameter, buffer, and optimizer tensor separately (`transformer_impl.py:849-859`), and is called for both branches (`1881-1887`). | Thousands of launch-sized collectives plus two model-sized broadcasts. In LoRA mode, frozen backbone tensors are still broadcast. |
| P0 | Base mixing performs two full CPU-to-GPU model passes | The immutable base is a complete CPU state dictionary. `mix_state_with_base()` transfers and FP32-mixes each tensor (`verl/experimental/sdc/sdc_state.py:37-51`) for both branches. | Two model-sized host-to-device streams and temporary FP32 allocations every SFT interval. |
| P0 | Diagnostics synchronize every actor microbatch | `compute_sdc_loss()` performs the loss-count all-reduce, metric all-reduce, max all-reduce, and `all_gather_object()` for quantiles (`verl/experimental/sdc/sdc_loss.py:100-145`). It also calls `.item()`/`.cpu()` in the loss path. | At least four distributed synchronizations per microbatch; the Python object gather stages data through CPU and scales poorly with world size and active-token count. |
| P1 | LoRA checkpoints still save full models | `_sdc_write_checkpoint()` saves each `model.state_dict()` and optimizer, then stores the full base again in `sdc_state.pt` (`transformer_impl.py:861-890`). | Large synchronous serialization and disk traffic every checkpoint, despite adapter-only trainable state. |
| P1 | Adapter export runs on all actor workers | `sdc_export_vllm_adapters` is `ONE_TO_ALL` (`engine_workers.py:774-777`); each worker clones adapter tensors to CPU before the trainer selects one payload (`ray_trainer.py:2170-2201`). | Redundant device-to-host copies, object creation, and Ray return traffic. |
| P1 | Samples are decoded and then tokenized repeatedly | Collection copies tensors to CPU and decodes every prompt/response (`ray_trainer.py:2283-2300`). Token counting tokenizes the records once, and SFT tokenizes them again (`transformer_impl.py:833-847`, `1771-1798`). | Python/tokenizer overhead, repeated allocations, and many small host-to-device copies. |
| P1 | HF mode adds two serial sidecar forwards and usually an old-policy forward | The active failure rows are scored by `_sdc_success` and then `_sdc_failure` (`transformer_impl.py:1713-1728`). The launcher defaults to HF for full fine-tuning and disables rollout-log-prob reuse (`examples/sdc_grpo/run_sdc_grpo_fsdp.sh:126-152`). | Three avoidable or structurally expensive forwards may be added before the actor update. |
| P2 | SDC lifecycle code is duplicated in the base engine and LM-head subclass | Matching SDC methods exist around `transformer_impl.py:670-1104` and `1589-1937`; the LM-head definitions override the base methods for actor runs. | Fixes can be applied to a non-active implementation, and the two paths can drift. |
| P2 | Sidecars inherit actor gradient checkpointing and miss available fused paths | `_build_module()` enables gradient checkpointing for all modules (`transformer_impl.py:318-319`). The launcher leaves Liger/fused kernels disabled and defaults to FlashAttention 2. | Sidecar SFT recomputes activations even when vLLM sleeping has freed enough memory; full logits are materialized for chosen-token log-probs and SFT cross-entropy. |

## Recommended target architectures

### Fast default: shared backbone with two LoRA adapters

Use one immutable SDC backbone and register `success` and `failure` as named
adapters. Keep independent optimizer moments and update counters for the two
adapter parameter sets.

For the current 1.5B-class workload, benchmark these modes in order:

1. replicated frozen backbone per GPU plus DDP-reduced adapter gradients;
2. one FSDP2-sharded frozen backbone plus sharded/replicated LoRA adapters;
3. success/failure adapters on two SDC submeshes if serial branch execution is
   still a major fraction of the step.

The first mode is likely to be fastest when one frozen backbone comfortably
fits on each GPU because it avoids parameter all-gathers. The second is the
memory-safe default for larger models. Selection must be based on measured
step time, not on sharding alone.

The base-mix equation becomes adapter-only. If `A0` is the frozen initial
adapter and `A` is the updated adapter, the exact operation is:

```text
A <- A0 + gamma * (A - A0)
```

No frozen backbone tensor needs to move or change. If `A0` is zero, this reduces
to scaling the adapter delta by `gamma`.

### Exact full-fine-tuning path: real FSDP sidecars

When SDC must update every S/F parameter:

- build each sidecar through a sidecar-safe variant of `_build_fsdp_module()`;
- initialize from sharded actor state rather than materializing a full actor
  state on every rank;
- partition SFT data across the data-parallel ranks;
- let FSDP gradient reduction keep replicas consistent;
- remove `_sdc_sync_model_and_optimizer()` entirely;
- use FSDP/DCP sharded state dictionaries for save and restore;
- benchmark FSDP2 first, then FSDP1 only as a compatibility fallback.

Do not wrap the current raw sidecars with FSDP while retaining the manual full
broadcast. That would pay both communication schemes.

## Implementation roadmap

### Phase 0 — establish a trustworthy baseline

1. Consolidate the shared SDC lifecycle into one implementation used by
   `FSDPEngineWithLMHead`; remove the duplicate override or make it explicitly
   delegate to the base helper.
2. Add timers and peak-memory metrics for:
   - S/F scoring;
   - SFT CPU preparation;
   - success forward/backward/optimizer;
   - failure forward/backward/optimizer;
   - base mixing;
   - distributed synchronization;
   - adapter export and vLLM reload;
   - SDC checkpoint save/load.
3. Record per-step SDC active examples/tokens, microbatch count, collective
   count/bytes, CUDA allocated/reserved peak, and host RSS.
4. Run at least 20 steady-state steps after warm-up with fixed data and seeds.
   Measure SDC disabled, current HF SDC, and current LoRA/vLLM SDC.

Deliverable: a baseline table with median and p95 phase times. Do not start
topology work until it proves which fraction is scoring, SFT, synchronization,
checkpointing, and rollout.

### Phase 1 — remove synchronization and data-path stalls

#### 1A. Move diagnostic collectives out of the microbatch loss

- Precompute the global active failed-token denominator once per logical actor
  minibatch and pass it through `global_batch_info`.
- Make `compute_sdc_loss()` return local sufficient statistics without any
  distributed call or host scalar conversion.
- Aggregate sums once after all microbatches.
- Replace exact per-step `all_gather_object()` quantiles with either:
  - a fixed-bin GPU histogram followed by one all-reduce; or
  - exact quantiles only every `sdc.metric_interval` steps.
- Keep the exact global denominator for the gradient. Diagnostic throttling must
  not alter SDC loss normalization.

Expected result: collective count becomes O(1) per actor update rather than
O(number of microbatches).

#### 1B. Store token IDs in the outcome buffers

- Store trimmed prompt IDs, response IDs, response length/mask, outcome, UID,
  and policy step directly.
- Remove decode in `_sdc_collect_outcomes()` and remove chat-template
  tokenization from `_sdc_response_token_count()` and `_sdc_sft_model()`.
- Bucket examples by total length and create one pinned CPU batch per
  microbatch, followed by one nonblocking device transfer.
- Preserve backward checkpoint compatibility by accepting old string records
  and converting them once on load.

#### 1C. Make LoRA state operations adapter-only

- Mix only trainable adapter tensors against their frozen initial values.
- Synchronize only adapter tensors and their optimizer state until the DDP/FSDP
  path in Phase 2 removes post-step synchronization entirely.
- Export adapters on rank 0 only.
- Save adapter state, adapter optimizer state, counters, collector state, base
  checkpoint identity, and configuration. Do not save two frozen backbones.
- Use atomic/versioned checkpoint directories and retain the old loader for one
  migration release.

This phase is semantics-preserving and should be implemented before tuning
FlashAttention, compile, or optimizer kernels.

### Phase 2 — build the shared-adapter fast path

1. Replace `_sdc_success` and `_sdc_failure` full modules with one frozen
   backbone and two named PEFT adapters.
2. Preserve independent AdamW state by using two optimizers or two isolated
   parameter groups with explicit adapter selection.
3. Shard each matched SFT batch across actor data-parallel ranks and all-reduce
   only adapter gradients. Normalize by the global number of valid sequences or
   response tokens so uneven shards remain exact.
4. Delete the post-SFT model/optimizer broadcast once distributed gradient
   synchronization is active.
5. Keep the vLLM multi-adapter scoring backend and rollout-log-prob reuse as the
   default fast configuration.
6. Replace the 2 x sequence count of Ray scoring RPCs with a batched server RPC
   per rollout replica if RPC scheduling is visible in the profile.
7. Reload only changed adapter tensors into vLLM. Reuse the existing weight-sync
   transport when possible instead of returning Python tensor dictionaries from
   every actor worker.

Initial parity runs must keep the existing LoRA rank (`512`). Rank
32/64/128 sweeps are a later algorithm-capacity experiment, not a pure
performance optimization.

### Phase 3 — implement the full-parameter FSDP path

1. Add an `SDCSidecarEngine` helper that owns its module, optimizer, scaler,
   state-dict policy, and device mesh without mutating the actor engine's
   `_autocast_dtype`, scaler, or offload flags.
2. Support `fsdp2` first:
   - apply the actor wrap policy to transformer blocks;
   - load a sharded actor snapshot through DTensor/FSDP state-dict APIs;
   - use BF16 parameters and benchmark BF16 versus FP32 gradient reduction;
   - keep CPU offload disabled for the speed profile;
   - checkpoint with `torch.distributed.checkpoint`.
3. Add an FSDP1 compatibility path only if required by the target cluster.
4. Benchmark sidecar `reshard_after_forward=false` when memory allows. Paired
   forwards may benefit from retaining gathered parameters; reject it if peak
   memory or wake-up headroom becomes unsafe.
5. Benchmark `forward_prefetch=true` and a sidecar-specific `fsdp_size` rather
   than inheriting `-1` blindly.
6. Remove full-state CPU materialization on every rank. Rank 0 may hold metadata,
   but model and optimizer payloads remain sharded.

Correctness gate: one distributed SFT step must match the current single-rank
reference update within BF16 tolerance, including base mixing and optimizer
resume.

### Phase 4 — overlap the two independent branches

Only after Phases 1-3:

- split the actor world into success and failure SDC submeshes;
- run success and failure scoring/SFT concurrently;
- gather only response log-prob outputs and scalar metrics;
- return all ranks to the actor collective schedule before actor update or
  rollout weight sync.

This can remove the serial S/F critical path, but each branch gets fewer GPUs.
Adopt it only if the profile shows lower wall time than running both branches on
the full mesh. Add timeout/fail-closed checks so a branch exception cannot leave
the other group blocked in a collective.

### Phase 5 — kernel and configuration tuning

Apply these only after structural waste is removed:

- disable gradient checkpointing for SDC SFT when the measured memory headroom
  permits it; keep actor checkpointing independently configurable;
- enable fused AdamW for CUDA adapter/full-parameter optimizers when supported;
- enable Liger SwiGLU and a fused or chunked response-only cross-entropy path;
- avoid materializing full `[batch, sequence, vocabulary]` logits when only
  chosen-token log-probabilities are needed;
- benchmark FlashAttention 3 on Hopper-class GPUs and retain FlashAttention 2
  elsewhere;
- compile stable SDC scoring/SFT kernels only after bucketed sequence shapes
  stop graph-recompile churn;
- tune SFT token buckets, `sft_batch_size`, and sidecar gradient accumulation
  from measured token throughput;
- tune vLLM memory utilization after confirming that the sleeping engine and
  sidecar peaks do not overlap unexpectedly.

Do not enable actor/sidecar parameter or optimizer CPU offload in the speed
profile. Offload is a memory escape hatch and normally adds host-device traffic.

## Configuration changes to introduce

Suggested fields under `custom_sdc`:

```yaml
execution_mode: shared_lora       # shared_lora | fsdp_full
metric_interval: 10
exact_quantile_interval: 0        # 0 disables exact all-gather quantiles
sidecar_fsdp_size: -1
sidecar_reshard_after_forward: true
sidecar_forward_prefetch: false
sidecar_gradient_checkpointing: false
sidecar_fused_optimizer: true
parallel_branches: false
checkpoint_adapter_only: true
token_id_replay_buffer: true
```

Keep safe fallbacks until parity tests and checkpoint migration pass. Invalid
combinations should fail during configuration validation rather than silently
falling back to replicated full models.

## Benchmark matrix

Use the same model, prompt batch, generation count, prompt/response limits,
seeds, and validation policy for every row.

| Run | Change from previous accepted run | Question answered |
|---|---|---|
| B0 | Current HF/full SDC | Baseline full-fine-tuning cost |
| B1 | Current LoRA/vLLM SDC | Existing fast-path baseline |
| A1 | Deferred metrics + no object gather | How much time is collective latency? |
| A2 | Token-ID buffers + length buckets | How much time is CPU preparation/H2D? |
| A3 | Adapter-only mix/sync/checkpoint/export | How much time is full-state movement? |
| A4 | Shared backbone + two adapters | How much memory/time do duplicate backbones cost? |
| A5 | Data-parallel adapter SFT | How much time is redundant all-rank SFT? |
| F1 | FSDP2 full sidecars | Is full-parameter SDC viable without replication? |
| F2 | F1 + tuned reshard/prefetch/fsdp_size | Best full-parameter topology |
| P1 | Parallel S/F submeshes | Does branch concurrency beat full-mesh serial work? |
| K1 | Best architecture + kernels/config | Incremental kernel gain |

Report both steady-state steps without checkpoints and amortized steps including
validation/checkpoints.

Required metrics:

- end-to-end seconds/step and generated tokens/second;
- SDC score, SFT, mix, sync, adapter reload, and checkpoint time;
- actor update and old-log-prob time;
- peak GPU allocated/reserved memory by phase and host RSS;
- NCCL collective count, bytes, and time;
- SFT response tokens/second and padding fraction;
- vLLM preemptions and score-request latency;
- success/failure log-prob parity and training/evaluation quality.

## Correctness and rollout gates

Before enabling any phase by default:

1. Existing CPU SDC algebra, masking, outcome, and base-mixing tests pass.
2. Add a 2-GPU distributed test proving:
   - one SFT update matches a single-rank reference;
   - S/F optimizer states remain independent;
   - nonzero ranks do not hold full checkpoint payloads;
   - no manual model broadcast occurs after FSDP/DDP synchronization.
3. Add FSDP1 and FSDP2 save/resume round trips for both LoRA and full modes.
4. Compare S/F token log-probabilities before and after refactoring on a fixed
   batch.
5. Verify the SDC actor gradient and global failed-token denominator across
   uneven rank-local batches and multiple microbatches.
6. Run a short deterministic training parity job, then a statistically powered
   quality run. Performance changes that alter LoRA rank, response length,
   staleness, or numeric precision require separate quality approval.
7. Roll out behind configuration flags in this order: metrics/data path,
   adapter-only lifecycle, shared adapter model, FSDP full path, parallel
   submeshes, kernels.

## Changes not recommended as the first move

- Do not start with CPU parameter/optimizer offload; it trades memory for PCIe
  traffic and does not remove redundant SDC computation.
- Do not expect FlashAttention or `torch.compile` to compensate for two full
  replicated models and model-sized broadcasts.
- Do not remove the global active-token normalization from the loss. Move its
  collective out of the microbatch path while preserving exact math.
- Do not lower LoRA rank or response length in the parity benchmark; both can
  change model quality.
- Do not introduce stale asynchronous SDC references until the synchronous
  implementation is fast and correct.

## Proposed implementation sequence

1. PR 1: consolidate duplicate engine code; add phase timers and memory/NCCL
   instrumentation.
2. PR 2: one-per-minibatch loss aggregation; sampled/histogram diagnostics;
   token-ID replay buffers.
3. PR 3: adapter-only mix, sync, export, and checkpoint with migration tests.
4. PR 4: shared backbone + two adapters + distributed adapter SFT; make this the
   recommended 1.5B fast path.
5. PR 5: FSDP2 full-parameter sidecars and distributed checkpoints.
6. PR 6: optional success/failure submeshes and kernel/config tuning.

The first implementation milestone should be PRs 1-4. They attack all observed
model-sized redundant work while preserving the existing SDC objective and
refresh schedule.

## Implemented slice

The first safe performance slice is now implemented on the `sdc` branch:

- `forward_backward_batch()` computes the exact global failed-response-token
  denominator once per logical actor batch and passes it through
  `global_batch_info`; the loss no longer launches that denominator reduction
  once per microbatch.
- SDC diagnostic reductions and exact quantile gathering are opt-in through
  `custom_sdc.distributed_metrics` (default `false`). Normal actor metrics are
  returned locally and aggregated by the existing worker path.
- The public SFT entrypoint now executes the replicated `ONE_TO_ALL` SFT work
  on rank zero, broadcasts the small result state, and reuses the coalesced
  model/optimizer synchronization for the other ranks. Rank-zero exceptions
  are propagated collectively instead of leaving workers in a deadlock.
- Full-finetuning synchronization now validates the complete parameter,
  buffer, and optimizer layout across ranks before communication, follows
  deterministic parameter-group order, and broadcasts bounded 64 MiB chunks.
  This removes model-sized temporary flattening while still splitting a single
  large tensor when necessary. CPU AdamW step tensors are staged through CUDA
  for NCCL and restored to CPU on receiving ranks.
- Full-model actor clones and checkpoint restores now require exact state-dict
  matches. Missing or renamed full-finetuning tensors fail immediately instead
  of leaving randomly initialized sidecar parameters. Standard AdamW step
  tensors also remain on CPU after checkpoint load; fused/capturable optimizers
  retain device-local steps.
- Outcome buffers preserve trimmed prompt and response token IDs. New records
  skip decode, chat-template reconstruction, and re-tokenization; old string
  checkpoints still use the compatibility path.
- SFT batches are assembled on pinned CPU memory and transferred once per
  batch, rather than copying each example independently to the GPU.
- LoRA base mixing, synchronization, export, checkpoint save, and checkpoint
  restore use adapter-only state. Broadcasts are coalesced by device/dtype and
  rank-zero-only export avoids duplicate payload construction.
- Regression coverage was added for the precomputed denominator and token-ID
  replay path, plus two-rank full-model parameter/buffer/optimizer
  synchronization, forced multi-chunk transfer, strict full-state restore, and
  AdamW state placement.
- The example launcher now sets `lora_rank=0` explicitly whenever
  `DRGRPO_USE_LORA=false`, and continues to force the actor/ref strategy to
  FSDP1.

The following planned work is intentionally not enabled by this slice: shared
backbone/two-adapter SFT, data-sharded SFT, true FSDP2 sidecar modules and
distributed sidecar checkpoints, branch submeshes, length bucketing, and
histogram/interval quantiles. Those changes alter the sidecar topology and
need distributed parity and GPU benchmarks before activation.

## Verification of the implemented slice

- Bytecode compilation passed for all modified Python modules.
- The modified FSDP transformer imports successfully.
- 26 no-fixture SDC regression functions passed through direct execution,
  including a two-process Gloo synchronization test for the full-finetuning
  model, buffers, and AdamW state.
- The repository environment does not include `pytest`; `uv run` could not
  synchronize because the unrelated legacy `pyext==0.7` dependency fails on
  Python 3.12 (`inspect.getargspec` removed). A full CUDA FSDP1 end-to-end
  smoke test was not run because the available GPUs were occupied by the
  ongoing GXPO training job and were deliberately left undisturbed.

The verified compatibility path is an FSDP1 `FULL_SHARD` actor with replicated
full-parameter success/failure SDC sidecars. The sidecars are synchronized
correctly after rank-zero SFT, but they are not themselves FSDP-sharded; the
true sharded-sidecar architecture remains Phase 3.
