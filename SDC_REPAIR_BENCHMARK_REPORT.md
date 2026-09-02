# SDC repair and benchmark report

Date: 2026-09-02
Branch: sdc
Environment: single node, 4x NVIDIA H200 143 GB, CUDA 13.0, PyTorch 2.13.0+cu130, Python 3.12, Ray 2.57.0.

## Commands

Baseline was launched in tmux session `sdc-baseline-20260902` with two policy steps,
SFT/checkpoint intervals of one, console logging, and the existing launcher. The log is
`/workspace/sdc-baseline-20260902.log`. It reached model/vLLM startup and the first
training step but did not complete a policy step before the run terminated. The log
shows the old path used replicated sidecars and a separate old-policy forward because
the 512-row smoke batch did not satisfy old-log-probability fusion.

The repaired launcher dry run was executed with:

~~~text
DRY_RUN=true WANDB_MODE=offline bash examples/sdc_grpo/run_sdc_grpo_fsdp.sh
~~~

It resolved actor/reference strategy `fsdp2`, `fsdp_size=4`, four visible H200 GPUs,
FlashAttention 3, Liger, `teacher_mode=full_fsdp`, and HF scoring without starting
Ray.

## Validation

Passed:

- SDC CPU and SDC-GRPO regression tests: 74 passed.
- Global contrast moment and microbatch partition-invariance tests.
- Python compile checks for SDC config, algebra, sharded teacher coordinator, actor
  engine, and trainer.
- Launcher shell syntax and dry-run preflight.
- PEFT named-adapter construction smoke check.

Not yet measured:

- paired-SFT refresh speedup versus the old rank-zero implementation;
- end-to-end repaired 4x H200 training, checkpoint/resume, and NCCL behavior;
- two-GPU FSDP1/FSDP2 numerical parity;
- legacy optimizer migration against a real v1 checkpoint;
- shared-LoRA vLLM round trip.

## Known limitations

The current work establishes the native sharded full_fsdp path and v2 checkpoint format.
A full GPU smoke run is currently blocked by an unrelated long-running GXPO process
holding all four H200s. The local-capacity scheduler for very long SFT batches and the
multi-node checkpoint harness still need dedicated distributed runs. These are validation
gaps, not silently treated as passed acceptance criteria.
