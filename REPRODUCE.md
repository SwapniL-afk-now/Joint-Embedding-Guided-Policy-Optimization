# Reproducing this project on a fresh server

RL post-training research (verl, FSDP) on **Qwen2.5-Math-1.5B-Instruct**: a **DAPO**
baseline vs. the "ours" **JEPA / TCR-dual** auxiliary objective. Read this end-to-end
before running anything; it is the single source of truth for setup, layout, and
conventions on a machine that only has this git clone.

> This doc is committed so any agent/human on a new server can bootstrap without the
> original machine's local notes. The heavy artifacts live on HuggingFace (see §3).

---

## 1. Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt          # verl + vLLM + torch stack
# secret: never committed. Create it by hand:
echo "HUGGINGFACE_API_KEY=hf_xxx" > .env     # used by training scripts (source .env) and HF pushes
```

This server is a **2× NVIDIA RTX PRO 6000 Blackwell Max-Q** box (≈96 GB each, sm_120,
**no NVLink** — the two cards talk over PCIe only). Single-GPU work is trivial; getting
**both GPUs** used for one training job, and getting **FlashInfer** to run in vLLM, each
took a specific set of workarounds — documented in **§8**. If you only need single-GPU,
you can ignore §8, but read the gotchas below.

### Gotchas that will bite on a new server
- **`unset RAY_ADDRESS`** before any verl launch. A shell with `RAY_ADDRESS=127.0.0.1`
  (no port) makes Ray die with `Invalid address format: 127.0.0.1`.
- **vLLM v1** needs eval scripts wrapped in `if __name__ == "__main__": main()`.
- **LoRA merge:** verl `model_merger` writes `adapter_config.json` with `lora_alpha=0`;
  set it to **1024** before PEFT `merge_and_unload` or the LoRA contribution is zeroed.
- Reward import path is `from verl.utils.reward_score import default_compute_score`.
- **Teacher caches:** the Ray JEPA trainer loads the **`.responses.pt`** (text) caches,
  NOT the bare `.pt` (latent tensors). Pointing `TEACHER_CACHE`/`CODE_TEACHER_CACHE` at a
  `.pt` file crashes with `Boolean value of Tensor ... is ambiguous`. Defaults in the run
  script are already the `.responses.pt` files.
- **vLLM attention backend:** on the *stock* stack (torch 2.7 / vLLM 0.10) FlashInfer is
  **not** installed and its API mismatches vLLM 0.10, so training scripts here fall back to
  `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. To actually run FlashInfer you must upgrade the whole
  vLLM/torch stack and add an ABI shim — see **§8.3**.

---

## 2. Directory layout to recreate

Everything below `/workspace/jepa-grpo-cache` and `/workspace/models` is data/weights,
NOT in git — pull from HF (§3).

```
/workspace/
  exploration/                       # THIS repo (clone of jepa-in-rl)
  models/Qwen2.5-Math-1.5B-Instruct  # base model (public HF download)
  jepa-grpo-cache/
    data/        *.parquet           # training data (HF dataset repo)
    eval_data/   *.parquet           # 10 benchmark parquets (HF dataset repo)
    teacher_targets.pt(+.responses.pt)        # CoT teacher cache (HF dataset repo)
    code_teacher_targets.pt(+.responses.pt)   # code teacher cache (HF dataset repo)
  checkpoints/                       # training output (local; push best to HF)
```

---

## 3. Models & datasets — pull from HuggingFace

```bash
# base model (public)
hf download Qwen/Qwen2.5-Math-1.5B-Instruct --local-dir /workspace/models/Qwen2.5-Math-1.5B-Instruct

# teacher caches + reproduction parquets (one dataset repo)
hf download swapnil7777/framework-teacher-cache --repo-type dataset --local-dir /tmp/fwcache
#   -> copy *.pt to /workspace/jepa-grpo-cache/, data/ and eval_data/ into /workspace/jepa-grpo-cache/

# merged inference models (optional, for eval / as a training init)
hf download swapnil7777/framework      --local-dir /workspace/models/framework      # root=DAPO-JEPA step690 (ours); DR_GRPO_FW_QWEN_MATH_1.5B/=tcr-dual step330; DAPO_QWEN_MATH_1.5B/=baseline step170
hf download swapnil7777/dapo-baseline  --local-dir /workspace/models/dapo-baseline   # DAPO-only baseline merged
```

**Important:** only *merged inference models* exist on HF — there is **no resumable
verl checkpoint** (no optimizer/LoRA-adapter/sampler state) anywhere. A new server can
**train fresh** or **eval**, but cannot **resume** an interrupted run.

Training parquets are filtered from public sets (DAPO-Math-17k, DeepScaleR-Preview);
eval parquets are regenerable via `python3 -m verl.experimental.fepo.data` /
`...fepo.code_data`, but the exact files are on the dataset repo for byte-fidelity.

---

## 4. Training

Launch scripts in `examples/`:
- `examples/drgrpo_trainer/run_qwen_1_5b_dapo_fsdp.sh` — **pure DAPO** baseline (main_ppo + `dapo` reward manager).
- `examples/jepa_grpo_trainer/run_qwen_1_5b_dapo_ray.sh` — **DAPO + JEPA/TCR-dual** (Ray + FSDP + hybrid-engine vLLM). Set `JEPA_ENABLE=False` to run pure DAPO through this script.
- `examples/jepa_grpo_trainer/run_qwen25_1_5b_dual_ray.sh` — the lora=128 dual reference.

All scripts are env-var driven (defaults in their `### user-adjustable ###` block). Common knobs:

| Var | Meaning |
|---|---|
| `MODEL_PATH` | init checkpoint (base model, or a merged model to continue from) |
| `DAPO_USE_LORA`/`USE_LORA`, `LORA_RANK`, `LORA_ALPHA` | LoRA (rank 512 / alpha 1024 used here) |
| `MAX_OPTIMIZER_STEPS` | run length (800 = full schedule) |
| `BEST_CKPT_METRIC`/`BEST_CKPT_METRICS` | best-ckpt selection: `avg@8`, `pass@8`, `combined`, … over `BEST_CKPT_SOURCES` |
| `JEPA_ENABLE`, `JEPA_LOSS_TYPE`, `ALPHA`, `TEACHER_CACHE`, `CODE_TEACHER_CACHE` | JEPA-only (tcr-dual needs both caches) |
| `KL_COEF`/`KL_LOSS_COEF` | KL fully off requires **0.0** (use_kl_loss=false alone is not enough) |

Example — pure DAPO from a merged checkpoint, best-ckpt on avg@8:
```bash
cd /workspace/exploration
unset RAY_ADDRESS
MODEL_PATH=/workspace/models/framework EXPERIMENT_NAME=DAPO-RUN \
  DAPO_USE_LORA=true LORA_RANK=512 MAX_OPTIMIZER_STEPS=800 BEST_CKPT_METRIC=avg@8 \
  bash examples/drgrpo_trainer/run_qwen_1_5b_dapo_fsdp.sh 2>&1 | tee dapo_run.log
```

Run long jobs in **tmux** (detach/reattach), never blocking foreground. After a run:
push artifacts to HuggingFace, then clean local disk.

---

## 5. Evaluation

Scripts (repo root): `eval_math_benchmarks.py`, `eval_code_benchmarks.py`, fast code
judge `code_judge_worker.py` (one subprocess per program, SIGALRM timeout — do not
regress to verl's per-test subprocess scorer). `eval_code_fast.py` is a faster
single-generate n-split variant (NOT the canonical protocol — std understates seed
variance).

**Conventions (match exactly):**
- **3 fixed seeds**, default `1234,2025,7777`. Report **mean ± std** across seeds.
- Default rollout budget **n=8** (sometimes 16). At n=8, `pass@16` is meaningless — report only `pass@8` and `avg@1`.
- Math: amc23, aime24/25/26 (+ math500, minervamath, olympiadbench extended). Code: humanevalplus, mbppplus, livecodebench.
- Model context **4096** → `max_tokens 3072`.

```bash
python eval_code_benchmarks.py --model /workspace/models/framework \
  --benchmarks humanevalplus --seeds 1234,2025,7777 --n 8 --out result.json
```

---

## 6. Workflow conventions

How work is run and reported on this project (carried over from the original machine's
local instructions; follow these unless the user says otherwise):

- **tmux, never foreground.** Launch long jobs (training, eval sweeps) in a tmux session
  and detach — `tmux send-keys -t <session> "<cmd>" Enter`. Status checks should be quick:
  capture the pane + `nvidia-smi` and report progress/ETA tersely. On this server the
  convention was: **training → `jepa` session, eval/benchmarks → `dapo-jepa` session** (a
  fresh server can use any names — the point is detached + reattachable, not blocking).
- **Single GPU, sequential.** Schedule jobs one at a time; each python frees the GPU on
  exit. Poll with a `wait_gpu()` loop (`nvidia-smi memory.used < 5000 MiB`) between steps.
  `send-keys` sent while a prior job's vLLM/Ray engine is still shutting down gets silently
  eaten — wait until the GPU is truly free AND the prompt is idle.
- **After a run: push artifacts to HuggingFace, then clean local disk.** Models →
  `swapnil7777/framework` (ours) / `swapnil7777/dapo-baseline`; datasets/caches →
  `swapnil7777/framework-teacher-cache`. Push uploads in the background (bandwidth-limited
  ~3.7 MB/s, not hung).
- **Code pushes → the `jepa-in-rl` git remote**, not `origin`.
- **Be direct and concise.** Short answers, tables for results, no preamble/filler.
- **Honesty + root-cause over acceptance.** When a result looks wrong, investigate the
  mechanism (e.g. truncation of verbose CoT before the code fence closes) — don't just
  accept or bury it. Report failures and caveats plainly. Put results in the paper with
  exact numbers and the right seeds; don't average old+new seed sets.

---

## 7. Output: ICLR paper

Results feed `/workspace/paper/iclr2025_conference.tex` (+ `iclr2026_*`). LaTeX format
for a result: `$VAL${\scriptsize$\,\pm X.X$}`. Put exact numbers, right seeds, honest
framing; do not average old+new seed sets.

---

## 8. Dual-GPU RTX PRO 6000 Blackwell + FlashInfer stack (this server)

Everything in this section is hardware-specific to the 2× RTX PRO 6000 Blackwell Max-Q
box. It is what it took to (a) run one training job across **both** GPUs and (b) run
**FlashInfer** as the vLLM rollout backend. None of it is needed for single-GPU runs on
NVLink hardware. Symptoms → fixes, in the order you hit them.

### 8.1 NCCL: the cards' P2P and SHM transports are broken

The two cards have **no NVLink** (`nvidia-smi topo -m` shows `NODE`, i.e. PCIe via the
host bridge). NCCL's default P2P and shared-memory transports **fault on the very first
collective** with `CUDA error: an illegal memory access was encountered` (the process
group watchdog then aborts). This is a platform/driver issue (PCIe ACS on the host), not
a verl bug — reproducible with a bare 2-proc `torchrun` all-reduce.

Fix: force NCCL onto the socket transport. Neither flag alone is enough — you need
**both**:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
```
Verified: a 2-GPU all-reduce/all-gather stress test passes with both set; crashes with
either unset. These are baked into the Ray runtime env for Blackwell in
`verl/trainer/constants_ppo.py` (alongside `NCCL_NVLS_ENABLE=0`/`NCCL_MNNVL_ENABLE=0`),
but also pass them in the launch env so the driver process is covered.

### 8.2 Use both GPUs for one job: DDP-style replication, NOT full FSDP shard

With `NDEVICES_PER_NODE=2`, verl's default FSDP `FULL_SHARD` **wedges** partway into the
first training step — specifically inside the JEPA embed pass
(`score_cot_embeddings`/`embed_targets`): rank 0 spins at 100% GPU while rank 1 sits idle,
no progress, no crash. `FULL_SHARD` needs a tightly-synchronized per-forward all-gather
that this workload's irregular chunked-forward loop does not survive on the socket transport.

Fix: run **DDP-style** — each GPU holds a full model replica, only gradients all-reduce.
Set the FSDP sub-group size to 1:
```
actor_rollout_ref.actor.fsdp_config.fsdp_size=1
```
This builds a `(ddp=2, fsdp=1)` device mesh. You'll see a harmless
`FSDP is switching to use NO_SHARD ... world size is 1` warning — that's the fsdp
sub-dimension; the ddp dimension still does the cross-GPU gradient sync. Both GPUs then run
generation + loss + backward, and steps complete.

### 8.3 FlashInfer in vLLM AND flash_attention_2 on the actor: pin torch 2.9

The stock env (torch 2.7.1+cu128, vLLM 0.10.0) **cannot** run FlashInfer: flashinfer isn't
installed, and even installed, vLLM 0.10 calls `trtllm_batch_decode_with_kv_cache(...,
num_heads=...)` which flashinfer ≥0.5 rejects (`unexpected keyword argument 'num_heads'`).
You must move to a vLLM new enough to match a modern flashinfer.

**Pick the torch version carefully — this is the whole trick.** Each vLLM release pins a
torch and a flashinfer (checked via each version's `requires_dist`):

| vLLM | torch | flashinfer |
|---|---|---|
| 0.16.0 | **2.9.1** | 0.6.3 |
| 0.17–0.19 | 2.10.0 | 0.6.4–0.6.6 |
| 0.20–0.21 | 2.11.0 | 0.6.8.post1 |

**Do NOT go to torch 2.11 (vLLM 0.20/0.21).** flash_attn has no torch-2.11 build, and the
torch-2.9 wheel forced onto torch 2.11 fails: an ABI symbol change
(`c10_cuda_check_implementation`'s line-no param `int`→`unsigned int`) breaks *loading*, and
even with an LD_PRELOAD shim to fix the symbol, flash_attn's **varlen** forward then dies at
runtime with `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`
(its torch-custom-op layer is incompatible with torch 2.11). torch 2.11 is a dead end for
standalone flash_attn — the reference env that runs vLLM 0.21+FlashInfer has *no* flash_attn
at all. (A historical `abi_shim.cpp`/`libabishim.so` for that path is kept in the repo but is
**not used** on the torch-2.9 stack below.)

**Use vLLM 0.16.0 → torch 2.9.1.** flash_attn 2.8.3 has a real, supported torch-2.9 build,
so `flash_attention_2` works natively (varlen included), and FlashInfer works via
flashinfer 0.6.3. No shim, no `LD_PRELOAD`.

```bash
uv pip install --python .venv/bin/python vllm==0.16.0          # -> torch 2.9.1+cu128, transformers 4.57, numpy 2.2.6/numba 0.61 (compatible)
uv pip install --python .venv/bin/python flashinfer-python==0.6.3 flashinfer-cubin==0.6.3   # both MUST match or import raises a version-check error
# flash_attn: the torch2.9 prebuilt wheel is cu13, but loads fine on torch 2.9.1+cu128; no cu12 torch2.9 wheel exists
uv pip install --python .venv/bin/python \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```
Sanity-check before launching:
```bash
.venv/bin/python -c "import torch,flash_attn; from flash_attn import flash_attn_varlen_func; \
  q=torch.randn(8,4,64,device='cuda',dtype=torch.bfloat16); cu=torch.tensor([0,4,8],dtype=torch.int32,device='cuda'); \
  print('varlen OK', tuple(flash_attn_varlen_func(q,q,q,cu,cu,4,4,causal=True).shape))"
.venv/bin/python -c "import flashinfer,vllm; print('flashinfer',flashinfer.__version__,'vllm',vllm.__version__)"
```

To run FlashInfer, simply **omit** `VLLM_ATTENTION_BACKEND` (it defaults to FlashInfer on
this vLLM; you'll see `flashinfer.jit [Autotuner]` in the log confirming it's live). Keep
`flash_attention_2` on the actor by leaving `ACTOR_ATTENTION_IMPL` at its default (do NOT set
`sdpa`). If you ever need to skip FlashInfer, `VLLM_ATTENTION_BACKEND=FLASH_ATTN` still works.

### 8.4 verl code fixes applied for this stack (already committed)

- **PEFT checkpoint deadlock** — `verl/utils/checkpoint/fsdp_checkpoint_manager.py`: the LoRA
  `peft_adapter` save called `save_pretrained()` on **rank 0 only**, but gathering the
  adapter under FSDP is a collective → hangs at world_size>1 (GPU0 100%, GPU1 idle, right
  after "Saved model config"). Fixed to gather the full state dict on **all** ranks
  (`get_fsdp_full_state_dict(..., rank0_only=True)`), then rank 0 writes it. Never manifested
  single-GPU.
- **Worker PYTHONPATH** — `constants_ppo.py` hardcoded `/venv/main/...site-packages` into the
  Ray workers' `PYTHONPATH`, shadowing the active `.venv` and injecting an incompatible numpy
  (broke numba/scipy in vLLM workers). Now derived from the running interpreter
  (`sysconfig.get_paths()["purelib"]`).
- **Reward int overflow** — degenerate rollouts emit huge integers; the math reward hit
  CPython's 4300-digit int↔str limit. `PYTHONINTMAXSTRDIGITS=0` is set in the worker runtime
  env.

### 8.5 Memory tuning (per-phase) and the canonical 2-GPU launch

A JEPA-GRPO step has distinct memory phases; tune each with its own knob. **Peaks are
stack-dependent** — values tuned on torch 2.7 OOM on torch 2.11 (higher baseline), so
re-tune after any stack change and watch `nvidia-smi` across a full step.

| Knob | Phase it drives | Notes |
|---|---|---|
| `ROLLOUT_GPU_MEM_UTIL` | vLLM generation + KV cache | 0.6 here. **0.75 OOMs** at the vLLM sleep/wake weight-sync (cumem allocator) because the actor holds a *full* replica under `fsdp_size=1`. |
| `EMBED_MICRO_BATCH_SIZE` | JEPA cot/code embedding | biggest single-phase lever; 112 was fine on torch 2.7 (~63 GB), OOMs on torch 2.11 — start ~40 and climb. |
| `log_prob_max_token_len_per_gpu` | old/ref log-prob | rollout+ref; 131072 ≈ 74–79 GB on torch 2.7. |
| `ppo_max_token_len_per_gpu` (`PPO_MAX_TOKEN_LEN_PER_GPU`) | actor fwd/backward | activation memory scales with this; kept at 24576. `expandable_segments:True` is **incompatible** with vLLM's cumem allocator (hard assert) — do not use it to paper over fragmentation; lower the knobs instead. |

Canonical working 2-GPU launch on this box (torch-2.9 / vLLM-0.16 stack from §8.3, model
here was Qwen2.5-3B-Instruct). FlashInfer + `flash_attention_2` both active, no shim:
```bash
cd /workspace/exploration
NDEVICES_PER_NODE=2 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
VAL_BEFORE_TRAIN=false VALIDATION_SEEDS="[31415]" \
ROLLOUT_GPU_MEM_UTIL=0.6 EMBED_MICRO_BATCH_SIZE=40 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
MODEL_PATH=/workspace/models/Qwen2.5-3B-Instruct \
  bash examples/jepa_grpo_trainer/run_qwen_3b_dapo_ray.sh \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=98304 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=98304 \
  2>&1 | tee dapo_jepa_run.log
```
(Omit `VLLM_ATTENTION_BACKEND` → FlashInfer. `flash_attention_2` is the actor default.)
`VAL_BEFORE_TRAIN=false` skips the pre-train validation (faster to reach the first step
while tuning); `VALIDATION_SEEDS="[31415]"` runs a single val seed when validation is on.
