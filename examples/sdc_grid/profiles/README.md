# Blackwell 2-GPU run profile (`examples/sdc_grid/profiles/`)

Run one `examples/sdc_grid` arm on this box: 4x **NVIDIA RTX PRO 6000 Blackwell
Server Edition** (`sm_120`, 97887 MiB each, driver 580.178.04, PCIe host-bridge
only, no NVLink). We use **GPU 0 and 1**. GPU 2 and 3 usually carry another job.

The launcher defaults in `examples/sdc_grid/common.sh` target 4x H200 and do not
work here. This profile overrides only hardware, path, and runtime settings. It
changes **no learning-relevant hyperparameter**, so the arms stay comparable.

## Files

| file | purpose |
| --- | --- |
| `blackwell_2gpu.env` | sourceable env profile (interpreter, GPUs, attention, paths, vLLM) |
| `run_blackwell_2gpu.sh` | wrapper that sources the profile and execs one arm |
| `cudart13/libcudart.so.13` | symlink shim; see "Why the shim" below |

## Launch

```bash
cd /office/dev_workspace/swapnil/Joint-Embedding-Guided-Policy-Optimization

# 1. always dry-run first: prints the fully resolved config, launches nothing
bash examples/sdc_grid/profiles/run_blackwell_2gpu.sh sdc_tr/drgrpo_a0.5.sh --dry-run

# 2. real launch inside tmux (a full arm is many hours)
tmux new -s sdc-drgrpo-a05
bash examples/sdc_grid/profiles/run_blackwell_2gpu.sh sdc_tr/drgrpo_a0.5.sh
# detach: Ctrl-b d      reattach: tmux attach -t sdc-drgrpo-a05
```

The arm path may be `sdc_tr/drgrpo_a0.5.sh`, `sdc_tr/drgrpo_a0.5`, or a path from
the repo root. Extra arguments go straight to hydra, for example the new
zero-advantage repair knob:

```bash
SDC_TR_DEGENERATE_COEF=0.05 \
  bash examples/sdc_grid/profiles/run_blackwell_2gpu.sh sdc_tr/drgrpo_a0.5.sh
```

Any variable exported before the wrapper wins, because the profile uses
`${VAR:-default}`. Useful short-run overrides for debugging **only** (they break
comparability, so never use them for a grid cell):

```bash
TOTAL_TRAINING_STEPS=3 TEST_FREQ=0 SAVE_FREQ=0 VAL_BEFORE_TRAIN=False \
WANDB_MODE=offline \
  bash examples/sdc_grid/profiles/run_blackwell_2gpu.sh sdc_tr/drgrpo_a0.5.sh
```

The wrapper refuses a real launch (exit code 3) if a requested GPU already holds
more than 4 GiB. `--dry-run` is always allowed.

## Maximum-throughput launcher

`run_blackwell_2gpu_max_throughput.sh` wraps the validated runner with an aggressive
runtime profile for vLLM and actor updates. It keeps `enforce_eager=False`, enables
CUDA-graph execution, raises vLLM to `gpu_memory_utilization=0.70`,
`max_num_seqs=2048`, and `max_num_batched_tokens=262144`, and raises the actor
dynamic token budget to `PPO_MAX_TOKEN_LEN_PER_GPU=65536`. These settings reduce
scheduler stalls and actor micro-batch count without changing the learning recipe.

```bash
TOTAL_TRAINING_STEPS=5 TEST_FREQ=0 VAL_BEFORE_TRAIN=False SAVE_FREQ=0 \
  bash examples/sdc_grid/profiles/run_blackwell_2gpu_max_throughput.sh \
  sdc_tr_zeroadv/drgrpo_a0.5_dg0.5.sh --dry-run
```

For a real run, remove `--dry-run` and run inside tmux. If either GPU OOMs, use the
conservative `run_blackwell_2gpu.sh` or override `ROLLOUT_GPU_MEM_UTIL=0.62`,
`ROLLOUT_MAX_NUM_SEQS=1024`, `ROLLOUT_MAX_NUM_BATCHED_TOKENS=131072`, and
`PPO_MAX_TOKEN_LEN_PER_GPU=49152`. If TorchInductor compilation fails on this
machine, retry with `USE_TORCH_COMPILE=false`.

## What this profile changes, and why

| setting | H200 default | here | reason |
| --- | --- | --- | --- |
| `PYTHON_BIN` | `.venv-h200/bin/python` | GXPO speed-audit `.venv` | `.venv-h200` does not exist; this repo's own `.venv` has a broken torch (`undefined symbol: ncclCommResume`) |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | `0,1` | GPU 2 and 3 are in use |
| `NDEVICES_PER_NODE` / `FSDP_SIZE` | 4 / 4 | 2 / 2 | two usable GPUs |
| `ACTOR_ATTENTION_IMPL` | `flash_attention_3` | `flash_attention_2` | FA3 is not installed and has no `sm_120` kernels; `flash_attn` 2.8.3 is installed and verified |
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` | `FLASH_ATTN` | vLLM 0.16.0 picked `FLASH_ATTN` on `sm_120` even when asked for FlashInfer; we pin what actually ran |
| `MODEL_PATH` | `/workspace/models/...` | GXPO `models/Qwen2.5-Math-1.5B-Instruct` | `/workspace` does not exist |
| `GXPO_DATA_ROOT` | `/workspace/data` | GXPO `Code/SFPO/data` | same |
| `SDC_VAL_CACHE_ROOT` | `$DATA_ROOT/sdc_validation_normalized` | `<repo>/data/sdc_validation_normalized` | the normalized val parquets live in this repo |
| `LOCAL_CUDA_LIB_ROOT`, `GXPO_CUDA_HOME` | set | disabled | `common.sh` would put CUDA 13 libraries and `nvcc` 13 in front of a cu12.8 torch |
| `LD_LIBRARY_PATH` | - | `+ profiles/cudart13` | see below |
| `ROLLOUT_GPU_MEM_UTIL` | 0.80 | 0.62 | 96 GiB cards, not 141 GiB; see "Memory" |
| `ROLLOUT_MAX_NUM_SEQS` | 2048 | 1024 | half the GPUs, half the concurrency |
| `ROLLOUT_MAX_NUM_BATCHED_TOKENS` | 262144 | 131072 | same |

### Why the shim (`cudart13/`)

`flash_attn_2_cuda*.so` in that venv was built against CUDA 13, but torch is
cu12.8. So `import flash_attn` fails with:

```
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

vLLM also imports `flash_attn` (rotary embedding), so without the fix the **vLLM
engine core does not start at all**. `cudart13/` holds one symlink to the CUDA 13
runtime already shipped inside that venv. We expose only `libcudart.so.13`.
Putting the whole `nvidia/cu13/lib` directory on `LD_LIBRARY_PATH` would also
shadow `libcufft.so.12`, `libcurand.so.10`, `libcusparse.so.12` and NCCL for the
cu12.8 torch build - that is exactly how you get `undefined symbol: ncclCommResume`.

## Validation performed (2 GPU box, GPU 0 only, `CUDA_VISIBLE_DEVICES=0`)

All checks passed:

* `torch` 2.9.1+cu128, `get_arch_list()` contains `sm_120`, device capability `(12, 0)`.
* bf16 4096x4096 matmul on `cuda:0`.
* `flash_attn` 2.8.3 `flash_attn_func` causal forward, shape `(2, 512, 8, 64)` - only after the `libcudart.so.13` shim.
* HuggingFace `Qwen2.5-Math-1.5B-Instruct` forward with `attn_implementation="flash_attention_2"`.
* vLLM 0.16.0 engine built on one GPU (`gpu_memory_utilization=0.30`, `max_num_seqs=8`, `enforce_eager`), generated correct text, then `sleep(level=1)` freed 24.06 GiB and `wake_up()` returned in 0.14 s.
* 2-rank NCCL all-reduce over GPU 0 and 1 (NCCL 2.27.5) - no P2P hang, so `NCCL_P2P_DISABLE` is not needed.
* Dry run of `sdc_tr/drgrpo_a0.5.sh` through the wrapper: preflight OK, `import verl` resolves inside this repo.

## Memory: 2 GPUs and 2 SDC teacher sidecars

Per GPU: 95.0 GiB usable of 97887 MiB. Measured: weights 2.88 GiB, and
`gpu_memory_utilization=0.30` gave 23.97 GiB of KV cache.

Rough per-GPU budget with `FSDP_SIZE=2` (1.54 B parameters, sharded over 2 ranks):

| phase | component | GiB |
| --- | --- | --- |
| rollout | vLLM pool at 0.62 | ~59 |
| rollout | actor params + AdamW state (resident) | ~10 |
| training | vLLM after `sleep(1)` (KV released) | ~5 |
| training | actor params, grads, fp32 AdamW state | ~10 |
| training | 2 SDC teacher sidecars (`full_fsdp`, own AdamW, `auto_offload`) | ~16 |
| training | activations, 49152 tokens/GPU, grad checkpointing + Liger fused CE | ~10 |
| training | reference model (`param_offload=True`) | 0 (CPU) |

Peak is the rollout phase: ~69 GiB of 95 GiB, about 26 GiB of headroom. Keeping
`ROLLOUT_GPU_MEM_UTIL=0.80` would put the rollout peak near 86 GiB, which does
not survive allocator fragmentation across the alternating actor / vLLM /
sidecar phases. That is why this profile uses 0.62.

Sleep mode is **mandatory** here: `free_cache_engine=True` and
`enable_sleep_mode=True` are already the grid defaults. Do not disable them, or
the sidecars have nowhere to live.

## Known risks

1. **Wall clock.** Two `sm_120` cards instead of four H200s. Expect roughly 3-4x
   the reference time per arm. A 400-step arm is a multi-day run. Use the queue
   (`examples/sdc_grid/queue.sh`) or launch one arm at a time.
2. **GPU 2 and 3.** Another job holds ~80 GiB on each. `common.sh` only prints a
   warning; the wrapper hard-fails on a real launch. Check `nvidia-smi` first.
3. **`use_torch_compile=true`.** Torch Inductor JIT on `sm_120` is not exercised
   by the short smoke test. The profile sets `PYTHON_DEV_INCLUDE_ROOT` so Triton
   finds `Python.h`. If you see a compile failure in the actor, set
   `USE_TORCH_COMPILE=false` and report it.
4. **CUDA 13 vs cu12.8 mix.** The stack works only because exactly one CUDA 13
   library is exposed. Do not widen `LD_LIBRARY_PATH`, and do not put the CUDA 13
   `nvcc` on `PATH`.
5. **vLLM attention backend.** vLLM ignored `VLLM_ATTENTION_BACKEND=FLASHINFER`
   in the engine-core process and used `FLASH_ATTN`. That differs from the H200
   reference, so rollout numerics are not bit-identical to the GXPO run. It is
   identical across all arms here, so the grid comparison stays valid.
6. **No NVLink.** All-reduce goes over PCIe (`PHB`). FSDP-2 gather/scatter is
   slower than on the H200 reference. If you see a NCCL hang, try
   `NCCL_P2P_DISABLE=1` (commented in the profile).
7. **W&B.** `WANDB_API_KEY` is not set on this box, so preflight warns and uploads
   fail. Use `WANDB_MODE=offline` for local runs, or put the key in a `.env` at
   the repo root.

## What to watch after launch

```bash
watch -n 5 nvidia-smi                       # GPU 0,1 memory must stay below ~90 GiB
tail -f results/sdc_grid/<experiment>/train.log
```

* First 10 minutes: vLLM engine build, then `val_before_train` on the six
  benchmarks. `PREFLIGHT FAIL` lines mean nothing launched.
* `sdc_tr/` metrics: `tilt_ratio`, `reliability_gate`, and, when
  `SDC_TR_DEGENERATE_COEF>0`, `sdc_tr/degenerate_coef`, `sdc_tr/mu_eff`,
  `sdc_tr/degenerate_token_fraction`, `sdc_tr/pseudo_advantage_abs_mean`.
* Memory spikes happen at the rollout -> training transition. If it OOMs there,
  lower `ROLLOUT_GPU_MEM_UTIL` before touching anything else, then
  `PPO_MAX_TOKEN_LEN_PER_GPU`.
