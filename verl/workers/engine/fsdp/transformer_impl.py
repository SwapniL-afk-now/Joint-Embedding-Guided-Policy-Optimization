# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import hashlib
import logging
import os
import time
import warnings
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.experimental.sdc.sft_trainer import (
    configure_sft_trainable_parameters,
    encode_outcome_example,
    outcome_sft_loss,
    pair_outcome_records_prompt_matched,
)
from verl.experimental.sdc.data_collector import OutcomeExample
from verl.experimental.sdc.sdc_state import build_sdc_checkpoint_state, clone_frozen_state, mix_module_with_base
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.utils.padding import build_attention_mask_from_nested

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)


def _sdc_autocast_context(device):
    """Use BF16 for direct SDC clone forwards required by FlashAttention2."""

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def sdc_cast_state_to_dest(state: dict, dest_state: dict, param_dtype, device):
    """Cast a state dict to the destination module's per-tensor dtypes.

    Preserve each destination tensor's dtype so both SDC clones match the actor
    for parameters and buffers.
    """
    dest_dtypes = {k: v.dtype for k, v in dest_state.items() if torch.is_tensor(v)}
    return {
        k: v.to(device=device, dtype=dest_dtypes.get(k, param_dtype))
        if torch.is_floating_point(v)
        else v.to(device)
        for k, v in state.items()
    }
def sdc_layout_digest(entries: list[tuple[str, torch.Tensor]], include_optimizer: bool) -> int:
    """Order-sensitive sha256 digest over entry name/dtype/shape strings.

    Used to validate SDC synchronization layouts without shipping the whole
    layout list every refresh. The first 8 bytes are interpreted as a signed
    int64 so two ranks can compare digests with one tiny MIN/MAX all-reduce.
    """

    hasher = hashlib.sha256()
    hasher.update(f"sdc-layout-v1|{int(bool(include_optimizer))}|".encode())
    for name, tensor in entries:
        hasher.update(name.encode())
        hasher.update(f"|{tensor.dtype}|{tuple(tensor.shape)}|{tensor.device.type}".encode())
    return int.from_bytes(hasher.digest()[:8], "big", signed=True)


logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class FSDPEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()

        # Apply NPU patches for FSDP backend
        from .utils import apply_npu_fsdp_patches

        apply_npu_fsdp_patches()

        # build device mesh for Ulysses Sequence Parallel

        self.use_remove_padding = self.model_config.use_remove_padding

        self._init_device_mesh()

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        # Defaults for mixed-precision state. _build_fsdp_module overrides these when it
        # runs; subclasses that bypass _build_fsdp_module (e.g. VeOmniEngine) keep the
        # defaults so forward_step / optimizer_step can still read them safely.
        self._autocast_dtype = torch.bfloat16
        self.scaler = None

        # QAT (Quantization-Aware Training)
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            logger.info(f"QAT enabled: mode={self._qat_config.mode}, group_size={self._qat_config.group_size}")

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        # This is used to import external_lib into the huggingface systems
        self._build_model_optimizer()

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_parallel_group = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size
        dp_size = self.get_data_parallel_size()
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
            self.ulysses_parallel_group = self.ulysses_device_mesh["sp"].get_group()

        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _build_module(self):
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            # if it is training, we force torch_dtype to fp32
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_config.model_type == "language_model":
                auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

                module = auto_class.from_pretrained(
                    pretrained_model_name_or_path=self.model_config.local_path,
                    torch_dtype=torch_dtype,
                    config=self.model_config.hf_config,
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            else:
                from verl.utils.model import load_valuehead_model

                assert self.model_config.model_type == "value_model", (
                    f"Unsupported model type: {self.model_config.model_type}"
                )
                self.model_config.hf_config.num_labels = 1
                self.model_config.hf_config.classifier_dropout = 0.0
                self.model_config.hf_config.hidden_dropout = "0"
                self.model_config.hf_config.summary_dropout_prob = 0.0
                module = load_valuehead_model(
                    local_path=self.model_config.local_path,
                    torch_dtype=torch_dtype,
                    model_config=self.model_config.hf_config,
                    trust_remote_code=self.model_config.trust_remote_code,
                )

            use_liger = self.model_config.use_liger
            # Apply Liger kernel; disable fused_linear_cross_entropy (conflicts with verl's forward patching)
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(
                    model=module,
                    fused_linear_cross_entropy=False,
                    swiglu=True,
                )

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            use_fused_kernels = self.model_config.use_fused_kernels
            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype
            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()

        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            from peft import PeftModel

            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            # Copy adapter to local if needed
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module = PeftModel.from_pretrained(module, local_adapter_path, is_trainable=True)
            peft_config = module.peft_config["default"]
            # Ensure task_type is TaskType enum, not string
            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM
        else:
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module = get_peft_model(module, LoraConfig(**lora_config))

        return module

    def _build_fsdp_module(self, module):
        # TODO(ziheng): need to improve
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        self._autocast_dtype = param_dtype
        # fp16 training requires loss scaling to avoid gradient underflow. Mirror the pattern
        # landed in #4036 for the legacy dp_actor path. bf16 / fp32 do not need a scaler.
        if param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == "fsdp":
            # cpu_offload:
            # - actor: None
            # - critic: None
            # - ref: CPUOffload(offload_params=True)

            # We force reference policy to use CPUOffload to save memory.
            # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
            # - actor: offload_policy
            # - critic: offload_policy
            # - ref: CPUOffloadPolicy(pin_memory=True)
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        optimizer = build_optimizer(module.parameters(), self.optimizer_config)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config

        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        lr_scheduler_type = optim_config.lr_scheduler_type
        min_lr_ratio = optim_config.min_lr_ratio
        num_cycles = optim_config.num_cycles
        zero_indexed_step = optim_config.zero_indexed_step
        if num_warmup_steps <= 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
                zero_indexed_step=zero_indexed_step,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return lr_scheduler

    def _apply_qat(self, module):
        """Apply QAT transformations to the model before FSDP wrapping."""
        from verl.utils.qat.core import apply_qat, enable_qat_fuse

        module = apply_qat(
            module,
            {
                "enable": self._qat_config.enable,
                "mode": self._qat_config.mode,
                "group_size": self._qat_config.group_size,
                "ignore_patterns": list(self._qat_config.ignore_patterns),
                "activation_observer": self._qat_config.activation_observer,
            },
        )
        enable_qat_fuse(module)

        if self._qat_config.mode == "w4a4":
            self._restore_w4a4_input_scales(module, self.model_config.local_path)

        return module

    def _restore_w4a4_input_scales(self, model, model_path):
        """Restore input_global_scale and input_amax from checkpoint for W4A4 mode."""
        import glob

        from safetensors import safe_open

        safetensor_files = glob.glob(f"{model_path}/model*.safetensors")
        loaded_count = 0

        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    if "input_global_scale" in key:
                        module_path = key.replace(".input_global_scale", "")
                        amax_key = f"{module_path}.input_amax"

                        module = model
                        for part in module_path.split("."):
                            module = module[int(part)] if part.isdigit() else getattr(module, part)

                        scale_val = f.get_tensor(key)
                        val = scale_val.item() if scale_val.numel() == 1 else scale_val.max().item()
                        module.input_global_scale.fill_(val)

                        amax_val = f.get_tensor(amax_key)
                        amax = amax_val.item() if amax_val.numel() == 1 else amax_val.max().item()
                        module.input_amax.fill_(amax)
                        loaded_count += 1

        logger.info(f"[QAT W4A4] Restored {loaded_count} input_global_scale/input_amax from {model_path}")

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        # Load base model with specified configuration and dtype
        module = self._build_module()
        # Apply LoRA adapters if low-rank adaptation is enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Apply QAT before FSDP wrapping (training only)
        if self._qat_enabled and not self.engine_config.forward_only:
            module = self._apply_qat(module)

        # Synchronize all distributed processes before proceeding
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
        log_gpu_memory_usage("Before FSDP", logger=None)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After FSDP", logger=None)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """Clear FSDP gradients and reset per-logical-batch step counters."""
        if self.optimizer is None:
            return
        self.optimizer.zero_grad(set_to_none=True)
        self._optimizer_step_calls = 0
        self._optimizer_steps_skipped_nonfinite = 0

    def optimizer_step(self, clip_grad_override: Optional[float] = None):
        """Clip the global FSDP gradient norm and apply one optimizer step."""
        assert self.optimizer is not None, "optimizer_step() is invalid for a forward-only FSDP engine"

        max_norm = self.optimizer_config.clip_grad if clip_grad_override is None else clip_grad_override
        max_norm = float("inf") if max_norm is None else float(max_norm)
        scaler = getattr(self, "scaler", None)
        if scaler is not None:
            scaler.unscale_(self.optimizer)

        version = fsdp_version(self.module)
        if version == 1:
            grad_norm = self.module.clip_grad_norm_(max_norm=max_norm)
        elif version == 2:
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=max_norm)
        else:
            raise RuntimeError(f"Unsupported FSDP version for optimizer step: {version}")

        grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        finite = bool(torch.isfinite(torch.as_tensor(grad_norm_value)).item())
        self._optimizer_step_calls = getattr(self, "_optimizer_step_calls", 0) + 1

        if finite:
            if scaler is not None:
                scaler.step(self.optimizer)
            else:
                self.optimizer.step()
        else:
            self._optimizer_steps_skipped_nonfinite = (
                getattr(self, "_optimizer_steps_skipped_nonfinite", 0) + 1
            )
            self.optimizer.zero_grad(set_to_none=True)
            logger.warning("Skipping FSDP optimizer step because grad_norm is not finite: %s", grad_norm_value)

        if scaler is not None:
            scaler.update()
        return grad_norm_value

    def lr_scheduler_step(self):
        """Advance the FSDP learning-rate scheduler and return its current LR."""
        if self.lr_scheduler is None:
            return None
        self.lr_scheduler.step()
        last_lr = self.lr_scheduler.get_last_lr()
        return last_lr[0] if len(last_lr) == 1 else last_lr

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def get_data_parallel_group(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        raise NotImplementedError

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        # note that the global_batch_size should include data on all the dp
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        # SDC's actor denominator is global over failed response tokens. Do
        # this once for the full forward/backward batch, before splitting into
        # micro-batches, instead of launching an all-reduce per micro-batch.
        if "sdc_failure_mask" in data and "response_mask" in data:
            response_mask = data["response_mask"].to(dtype=torch.bool)
            failure_mask = data["sdc_failure_mask"].to(dtype=torch.bool)
            if failure_mask.ndim < response_mask.ndim:
                failure_mask = failure_mask.reshape(-1, 1)
            active_token_count = (response_mask & failure_mask).sum().to(get_device_id())
            torch.distributed.all_reduce(
                active_token_count,
                op=torch.distributed.ReduceOp.SUM,
                group=self.get_data_parallel_group(),
            )
            tu.assign_non_tensor(data, sdc_active_token_count=active_token_count.item())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__
        # and _build_fsdp_module, so self.scaler may not be set.
        scaler = getattr(self, "scaler", None)

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                if not forward_only:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            output_lst.append(meta_info)

        # postprocess and return
        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def _sdc_is_rank_zero(self) -> bool:
        return not (torch.distributed.is_available() and torch.distributed.is_initialized()) or (
            torch.distributed.get_rank() == 0
        )

    def _sdc_record_timing(self, key: str, start: float | None, accumulate: bool = False) -> None:
        """Record one wall-clock phase duration in ms on rank zero only.

        Nonzero ranks do nothing beyond their existing collectives.
        """

        if start is None or not self._sdc_is_rank_zero():
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings = getattr(self, "_sdc_phase_timings", None)
        if timings is None:
            timings = self._sdc_phase_timings = {}
        timings[key] = timings.get(key, 0.0) + elapsed_ms if accumulate else elapsed_ms

    def _sdc_timing_metrics(self, *keys: str) -> dict[str, float]:
        """Expose recorded phase timings as trainer-metric float keys."""

        if not self._sdc_is_rank_zero():
            return {}
        timings = getattr(self, "_sdc_phase_timings", None) or {}
        return {f"sdc/timing/{key}": float(timings.get(key, 0.0)) for key in keys}

    def _sdc_actor_state_cpu(self) -> dict[str, torch.Tensor]:
        def to_cpu(value):
            full = value.full_tensor() if hasattr(value, "full_tensor") else value
            return full.detach().to(device="cpu", copy=True)
        if fsdp_version(self.module) == 1 and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            wrapped = getattr(self.module, "_fsdp_wrapped_module", self.module)
            with FSDP.summon_full_params(self.module, writeback=False):
                return {k: to_cpu(v) for k, v in wrapped.state_dict().items() if torch.is_tensor(v)}
        return {k: to_cpu(v) for k, v in self.module.state_dict().items() if torch.is_tensor(v)}

    def _sdc_apply_sidecar_gradient_checkpointing(self, module) -> None:
        """Apply the optional sidecar-only gradient-checkpointing override.

        ``None`` (default) inherits whatever checkpointing configuration the
        actor module was built with. An explicit True/False touches ONLY this
        freshly built sidecar module; the actor module is never modified.
        Recompute-or-not does not change numerics for scoring or SFT.
        """
        setting = getattr(self, "_sdc_sidecar_gradient_checkpointing", None)
        if setting is None:
            return
        module_config = getattr(module, "config", None)
        if module_config is not None and hasattr(module_config, "use_cache"):
            module_config.use_cache = False
        if bool(setting):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
        elif hasattr(module, "gradient_checkpointing_disable"):
            module.gradient_checkpointing_disable()

    def _sdc_clone_from_state(self, state: dict[str, torch.Tensor]):
        module = self._build_module()
        if self._is_lora:
            module = self._build_lora_module(module)
        self._sdc_apply_sidecar_gradient_checkpointing(module)
        device = next(self.module.parameters()).device
        if any(param.is_meta for param in module.parameters()):
            module.to_empty(device=device)
        else:
            module.to(device=device)
        dtype = next(self.module.parameters()).dtype
        for param in module.parameters():
            param.data = param.data.to(dtype)
        loaded = sdc_cast_state_to_dest(state, module.state_dict(), next(module.parameters()).dtype, device)
        # Full fine-tuning must be an exact actor clone.  A permissive load can
        # silently leave randomly initialized parameters when an FSDP1 state
        # key changes.  LoRA keeps the permissive path because PEFT adds
        # adapter keys that are not present in the actor snapshot.
        module.load_state_dict(loaded, strict=not self._is_lora)
        module.train(True)
        configure_sft_trainable_parameters(module, lora_enabled=self._is_lora)
        return module

    def _sdc_adapter_base_state(self, model) -> dict[str, torch.Tensor]:
        """Capture only trainable LoRA tensors and release the full base copy."""

        existing = getattr(self, "_sdc_base_adapter_state_cpu", None)
        if existing is not None:
            return existing
        base_state = getattr(self, "_sdc_base_state_cpu", None)
        if base_state is None:
            raise RuntimeError("SDC adapter base state is unavailable.")
        adapter_state = {
            name: base_state[name].detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name in base_state
        }
        if not adapter_state:
            raise RuntimeError("Could not match trainable SDC adapter parameters to the actor state.")
        self._sdc_base_adapter_state_cpu = adapter_state
        # The frozen backbone is already owned by the actor. Keeping a second
        # full CPU snapshot is needlessly expensive for LoRA SDC.
        self._sdc_base_state_cpu = None
        return adapter_state

    def _sdc_mix_adapter_with_base(self, model) -> None:
        base_state = self._sdc_adapter_base_state(model)
        gamma = float(self._sdc_base_mix_gamma)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                base = base_state.get(name)
                if base is None:
                    continue
                mixed = (
                    parameter.detach().float() * gamma
                    + base.to(device=parameter.device, dtype=torch.float32) * (1.0 - gamma)
                )
                parameter.copy_(mixed.to(dtype=parameter.dtype))

    @staticmethod
    def _sdc_move_optimizer_state(optimizer, device):
        for group in optimizer.param_groups:
            step_on_parameter_device = bool(group.get("capturable", False) or group.get("fused", False))
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                for key, value in state.items():
                    if not torch.is_tensor(value):
                        continue
                    target_device = device
                    if key == "step" and not step_on_parameter_device:
                        target_device = torch.device("cpu")
                    value.data = value.to(device=target_device, non_blocking=True)

    def _sdc_prepare_models(self, include_optimizer=False):
        """Place SDC clones on the actor device for scoring or SFT."""
        device = next(self.module.parameters()).device
        for model, optimizer in ((self._sdc_success, self._sdc_success_optimizer), (self._sdc_failure, self._sdc_failure_optimizer)):
            if next(model.parameters()).device != device:
                model.to(device)
            if include_optimizer:
                self._sdc_move_optimizer_state(optimizer, device)

    @staticmethod
    def _sdc_has_cuda_state(model, optimizer) -> bool:
        """Whether any sidecar parameter or optimizer tensor lives on CUDA."""
        if any(parameter.device.type == "cuda" for parameter in model.parameters()):
            return True
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                for value in optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.device.type == "cuda":
                        return True
        return False

    def _sdc_offload_models(self) -> bool:
        """Release SDC clone and optimizer GPU memory before vLLM wake-up.

        Returns True when at least one GPU-resident sidecar tensor was moved
        to host memory. ``torch.cuda.empty_cache()`` runs only in that case;
        a call where everything is already on CPU never touches the allocator.
        """
        cpu = torch.device("cpu")
        pairs = (
            (self._sdc_success, self._sdc_success_optimizer),
            (self._sdc_failure, self._sdc_failure_optimizer),
        )
        moved = any(self._sdc_has_cuda_state(model, optimizer) for model, optimizer in pairs)
        for model, optimizer in pairs:
            self._sdc_move_optimizer_state(optimizer, cpu)
            model.to(cpu)
        if moved and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return moved

    def _sdc_release_models(self) -> None:
        """Offload sidecar clones unless residency mode keeps them on GPU."""
        if getattr(self, "_sdc_sidecar_residency", "auto_offload") == "resident":
            return
        self._sdc_offload_models()

    def sdc_release_residency(self) -> None:
        """Explicit release point for ``sidecar_residency='resident'``.

        Offloads both SDC clones plus their optimizer state and frees cached
        CUDA blocks so vLLM can reclaim GPU memory at wake-up. Trainers using
        resident mode MUST call this before waking vLLM / syncing rollout
        weights. The default 'auto_offload' mode does not need it because
        every scoring/SFT pass already releases on its own.
        """
        self._sdc_offload_models()


    @staticmethod
    def _sdc_ensure_optimizer_state(optimizer) -> None:
        """Materialize AdamW state on ranks that skipped rank-zero SFT work."""

        for group in optimizer.param_groups:
            amsgrad = bool(group.get("amsgrad", optimizer.defaults.get("amsgrad", False)))
            step_on_parameter_device = bool(group.get("capturable", False) or group.get("fused", False))
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                if state:
                    continue
                step_device = parameter.device if step_on_parameter_device else torch.device("cpu")
                state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(parameter)

    def _sdc_model_log_probs(self, module, input_ids, attention_mask, response_mask):
        was_training = module.training
        module.eval()
        try:
            with torch.no_grad(), _sdc_autocast_context(input_ids.device):
                outputs = module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                labels = input_ids[:, 1:].contiguous()
                if outputs.logits is not None:
                    logits = outputs.logits[:, :-1].contiguous()
                    token_lp = logprobs_from_logits(
                        logits.view(-1, logits.size(-1)), labels.view(-1), inplace_backward=False
                    ).view(labels.shape)
                elif getattr(outputs, "log_probs", None) is not None:
                    # The Triton fused actor path returns per-token log-probs rather
                    # than logits. Drop its final rolled-label position to match the
                    # ordinary shifted-label path above.
                    token_lp = outputs.log_probs.reshape(input_ids.shape)[:, :-1].contiguous()
                else:
                    raise RuntimeError("SDC model output contains neither logits nor fused log_probs.")
                width = response_mask.shape[-1]
                prompt_width = input_ids.shape[-1] - width
                result = torch.zeros(input_ids.shape[0], width, device=input_ids.device, dtype=torch.float32)
                available = min(width, token_lp.shape[-1] - prompt_width + 1)
                if available > 0:
                    result[:, :available] = token_lp[:, prompt_width - 1 : prompt_width - 1 + available].float()
        finally:
            module.train(was_training)
        return result.detach()

    @staticmethod
    def _sdc_parameter_counts(model, optimizer) -> dict[str, int]:
        parameters = list(model.parameters())
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        return {
            "total": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "optimizer": sum(parameter.numel() for parameter in parameters if id(parameter) in optimizer_parameters),
        }

    def sdc_init(self, sdc_config: dict):
        if getattr(self, "_sdc_initialized", False):
            return {"sdc_initialized": True}
        if self.engine_config.strategy not in ("fsdp", "fsdp2"):
            raise NotImplementedError("SDC currently supports only FSDP/FSDP2 HF actors.")
        state = self._sdc_actor_state_cpu()
        self._sdc_base_state_cpu = clone_frozen_state(state)
        self._sdc_base_adapter_state_cpu = None
        # Fresh base snapshot: the next checkpoint save embeds it again and
        # re-anchors the slim-checkpoint marker chain.
        self._sdc_saved_base_global_step = None
        self._sdc_base_mix_gamma = float(sdc_config.get("base_mix_gamma", 0.9))
        if not 0.0 <= self._sdc_base_mix_gamma <= 1.0:
            raise ValueError(
                f"custom_sdc.base_mix_gamma must be between 0.0 and 1.0, got {self._sdc_base_mix_gamma}."
            )
        sidecar_residency = str(sdc_config.get("sidecar_residency") or "auto_offload")
        if sidecar_residency not in ("auto_offload", "resident"):
            raise ValueError(
                f"custom_sdc.sidecar_residency must be 'auto_offload' or 'resident', got {sidecar_residency!r}."
            )
        self._sdc_sidecar_residency = sidecar_residency
        sidecar_gradient_checkpointing = sdc_config.get("sidecar_gradient_checkpointing", None)
        self._sdc_sidecar_gradient_checkpointing = (
            None if sidecar_gradient_checkpointing is None else bool(sidecar_gradient_checkpointing)
        )
        self._sdc_success = self._sdc_clone_from_state(self._sdc_base_state_cpu)
        self._sdc_failure = self._sdc_clone_from_state(self._sdc_base_state_cpu)
        lr = float(sdc_config.get("sft_lr", 1e-6))
        fused_adamw = bool(sdc_config.get("fused_adamw", False))
        self._sdc_success_optimizer = torch.optim.AdamW(
            (parameter for parameter in self._sdc_success.parameters() if parameter.requires_grad),
            lr=lr,
            fused=fused_adamw,
        )
        self._sdc_failure_optimizer = torch.optim.AdamW(
            (parameter for parameter in self._sdc_failure.parameters() if parameter.requires_grad),
            lr=lr,
            fused=fused_adamw,
        )
        self._sdc_success_model_update_count = int(sdc_config.get("sdc_success_model_update_count", 0))
        self._sdc_failure_model_update_count = int(sdc_config.get("sdc_failure_model_update_count", 0))
        self._sdc_models_ready = bool(sdc_config.get("sdc_models_ready", False))
        self._sdc_tokenizer = None
        self._sdc_initialized = True
        actor_counts = {
            "total": sum(parameter.numel() for parameter in self.module.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad),
        }
        success_counts = self._sdc_parameter_counts(self._sdc_success, self._sdc_success_optimizer)
        failure_counts = self._sdc_parameter_counts(self._sdc_failure, self._sdc_failure_optimizer)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed and torch.distributed.get_rank() != 0 and not self._is_lora:
            # Only rank zero interpolates SDC weights with the frozen actor.
            # Scoring ranks retain the two synchronized models but not this
            # extra full-model CPU snapshot.
            self._sdc_base_state_cpu = None
        # In resident mode keep the freshly built clones on the actor GPU; the
        # explicit sdc_release_residency() hook does the release later.
        self._sdc_release_models()
        return {
            "sdc_initialized": True,
            "sdc/base_mix_gamma": self._sdc_base_mix_gamma,
            "sdc/actor_total_parameters": actor_counts["total"],
            "sdc/actor_trainable_parameters": actor_counts["trainable"],
            "sdc/success_total_parameters": success_counts["total"],
            "sdc/success_trainable_parameters": success_counts["trainable"],
            "sdc/success_optimizer_parameters": success_counts["optimizer"],
            "sdc/failure_total_parameters": failure_counts["total"],
            "sdc/failure_trainable_parameters": failure_counts["trainable"],
            "sdc/failure_optimizer_parameters": failure_counts["optimizer"],
        }

    def sdc_reinitialize_from_actor(self, sdc_config: dict):
        self._sdc_initialized = False
        reset_config = dict(sdc_config or {})
        reset_config.pop("sdc_success_model_update_count", None)
        reset_config.pop("sdc_failure_model_update_count", None)
        reset_config.pop("sdc_models_ready", None)
        return self.sdc_init(reset_config)

    def _sdc_mix_with_base(self, model) -> None:
        """Restore the common frozen-base component after a paired SFT refresh."""

        if self._is_lora:
            self._sdc_mix_adapter_with_base(model)
        else:
            mix_module_with_base(model, self._sdc_base_state_cpu, self._sdc_base_mix_gamma)

    def sdc_compute_success_failure_log_probs(self, data: TensorDict) -> TensorDict:
        config = tu.get_non_tensor_data(data=data, key="custom_sdc", default={}) or {}
        self.sdc_init(config)
        scoring_start = time.perf_counter() if self._sdc_is_rank_zero() else None
        padded = data.to_padded_tensor()
        device = next(self.module.parameters()).device
        ids = padded["input_ids"].to(device)
        attention = padded["attention_mask"].to(device)
        response = padded["response_mask"].to(device)
        active = padded.get("sdc_failure_mask", torch.ones(ids.shape[0], dtype=torch.bool, device=device)).to(device).bool()
        success = torch.zeros_like(response, dtype=torch.float32)
        failure = torch.zeros_like(response, dtype=torch.float32)
        if bool(active.any()):
            # Prepare once per scoring pass instead of once per clone; the
            # move is idempotent, so results are bitwise identical.
            self._sdc_prepare_models()
            try:
                active_ids, active_attention, active_response = ids[active], attention[active], response[active]
                active_index = active.nonzero(as_tuple=True)[0]
                # Score active rows in bounded batch chunks instead of one dense
                # forward over every row padded to the global max length. Rows are
                # independent under no_grad and the padded width is unchanged, so
                # per-row log-probs are bitwise identical to the single-pass result.
                micro_batch_size = int(config.get("vllm_score_micro_batch_size", 0) or 0)
                if micro_batch_size <= 0:
                    micro_batch_size = 8
                total_active = active_ids.shape[0]
                for start in range(0, total_active, micro_batch_size):
                    end = min(start + micro_batch_size, total_active)
                    rows = slice(start, end)
                    success[active_index[rows]] = self._sdc_model_log_probs(
                        self._sdc_success, active_ids[rows], active_attention[rows], active_response[rows]
                    )
                    failure[active_index[rows]] = self._sdc_model_log_probs(
                        self._sdc_failure, active_ids[rows], active_attention[rows], active_response[rows]
                    )
            finally:
                self._sdc_release_models()
        self._sdc_record_timing("scoring_ms", scoring_start)
        return TensorDict(
            {"sdc_success_log_probs": success, "sdc_failure_log_probs": failure}, batch_size=[ids.shape[0]]
        )

    def _sdc_get_tokenizer(self):
        if self._sdc_tokenizer is None:
            from transformers import AutoTokenizer
            self._sdc_tokenizer = AutoTokenizer.from_pretrained(self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code)
            if self._sdc_tokenizer.pad_token_id is None:
                self._sdc_tokenizer.pad_token = self._sdc_tokenizer.eos_token
        return self._sdc_tokenizer

    def _sdc_match_records(self, success_records: list[dict], failure_records: list[dict], config: dict):
        max_prompt_length = int(config.get("max_prompt_length", 1024))
        max_response_length = int(config.get("max_response_length", 2048))
        token_budget = int(config.get("sft_max_token_len_per_gpu", 0) or 0)
        max_length = max_prompt_length + max_response_length
        batch_size = int(config.get("sft_batch_size", 8))
        if token_budget > 0:
            batch_size = min(batch_size, max(1, token_budget // max_length))
        max_examples = batch_size * int(config.get("sft_max_updates_per_interval", 1))
        # Canonical behavior: prompt-matched teacher pairing. Falls back to the
        # legacy count-prefix pairing internally when no uids overlap.
        return pair_outcome_records_prompt_matched(
            success_records, failure_records, max_examples=max_examples
        )

    def _sdc_response_token_count(self, records: list[dict], config: dict) -> int:
        tokenizer = self._sdc_get_tokenizer()
        max_prompt_length = int(config.get("max_prompt_length", 1024))
        max_response_length = int(config.get("max_response_length", 2048))
        return sum(
            int(
                encode_outcome_example(
                    OutcomeExample(**record),
                    tokenizer,
                    max_prompt_length=max_prompt_length,
                    max_response_length=max_response_length,
                )[1].sum()
            )
            for record in records
        )

    def _sdc_sync_model_and_optimizer(
        self,
        model,
        optimizer,
        *,
        include_optimizer: bool = True,
    ) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        entries: list[tuple[str, torch.Tensor]] = []
        entries.extend(
            (f"parameter:{name}", parameter.data)
            for name, parameter in model.named_parameters()
            if not self._is_lora or parameter.requires_grad
        )
        entries.extend((f"buffer:{name}", buffer.data) for name, buffer in model.named_buffers())

        if include_optimizer:
            # Iterate optimizer state by parameter-group order and fixed state-key
            # order. Relying on optimizer.state insertion order is unsafe when only
            # rank zero performs the first AdamW step.
            preferred_state_keys = ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq")
            for group_index, group in enumerate(optimizer.param_groups):
                for parameter_index, parameter in enumerate(group["params"]):
                    state = optimizer.state.get(parameter, {})
                    ordered_keys = [key for key in preferred_state_keys if key in state]
                    ordered_keys.extend(sorted(key for key in state if key not in preferred_state_keys))
                    for key in ordered_keys:
                        value = state[key]
                        if torch.is_tensor(value):
                            entries.append((f"optimizer:{group_index}:{parameter_index}:{key}", value.data))

        # Fail before launching tensor broadcasts if ranks disagree on model
        # or optimizer state. Otherwise a first-refresh mismatch can hang NCCL.
        # The full all_gather_object comparison runs ONLY on the first refresh;
        # afterwards an order-sensitive sha256 digest compared through one tiny
        # int64 MIN/MAX all-reduce validates the layout at negligible cost.
        cache = getattr(self, "_sdc_layout_digest_cache", None)
        if cache is None:
            cache = self._sdc_layout_digest_cache = {}
        cached_digest = cache.get(bool(include_optimizer))
        local_digest = sdc_layout_digest(entries, include_optimizer)
        backend = str(torch.distributed.get_backend()).lower()
        token_device = get_device_id() if "nccl" in backend else "cpu"
        # Pack [digest, -digest]; one MIN all-reduce yields both the global min
        # and (negated) the global max. Agreement means every rank matched.
        token = torch.tensor([local_digest, -local_digest], dtype=torch.int64, device=token_device)
        torch.distributed.all_reduce(token, op=torch.distributed.ReduceOp.MIN)
        digests_agree = int(token[0].item()) == -int(token[1].item())
        if not digests_agree or cached_digest is None:
            layout = [(name, tensor.device.type, str(tensor.dtype), tensor.numel()) for name, tensor in entries]
            if cached_digest is None:
                # First refresh with this configuration: validate fully so any
                # mismatch is reported against the exact offending rank.
                layouts: list[list[tuple[str, str, str, int]] | None] = [
                    None
                ] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(layouts, layout)
                if any(candidate != layout for candidate in layouts):
                    mismatch_rank = next(index for index, candidate in enumerate(layouts) if candidate != layout)
                    raise RuntimeError(
                        "SDC synchronization layout differs across ranks; "
                        f"local rank {torch.distributed.get_rank()} does not match rank {mismatch_rank}."
                    )
                if not digests_agree:
                    raise RuntimeError(
                        "SDC synchronization layout digests disagree across ranks even though "
                        f"rank {torch.distributed.get_rank()} gathered identical layouts."
                    )
                cache[bool(include_optimizer)] = local_digest
            else:
                raise RuntimeError(
                    "SDC synchronization layout hash differs across ranks; "
                    f"local rank {torch.distributed.get_rank()} digest {local_digest:#018x} "
                    f"disagrees with the group (min {int(token[0].item()):#018x})."
                )

        grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for _, tensor in entries:
            if tensor.numel() == 0:
                continue
            grouped.setdefault((tensor.device, tensor.dtype), []).append(tensor)

        rank = torch.distributed.get_rank()
        max_chunk_bytes = int(getattr(self, "_sdc_sync_chunk_bytes", 64 * 1024 * 1024))
        if max_chunk_bytes <= 0:
            raise ValueError("SDC synchronization chunk size must be positive.")
        staging_buffers: dict[tuple, torch.Tensor] = getattr(self, "_sdc_staging_buffers", None) or {}
        self._sdc_staging_buffers = staging_buffers

        with torch.no_grad():
            for (device, dtype), grouped_tensors in sorted(
                grouped.items(), key=lambda item: (item[0][0].type, item[0][0].index or -1, str(item[0][1]))
            ):
                max_chunk_elements = max(1, max_chunk_bytes // torch.empty((), dtype=dtype).element_size())
                # One persistent pinned staging buffer per (device, dtype)
                # group, reused across flushes instead of fresh torch.cat
                # allocations. Chunk ordering and broadcast semantics are
                # unchanged; only the temporary allocation is removed.
                group_key = ("staging", str(device), str(dtype))
                staged = staging_buffers.get(group_key)
                if staged is None or staged.numel() != max_chunk_elements:
                    staged = torch.empty(max_chunk_elements, dtype=dtype, device=device)
                    if device.type == "cpu" and torch.cuda.is_available():
                        staged = staged.pin_memory()
                    staging_buffers[group_key] = staged
                relayed = None
                if device.type == "cpu" and "nccl" in backend:
                    relay_key = ("relay", str(device), str(dtype))
                    relayed = staging_buffers.get(relay_key)
                    if relayed is None or relayed.numel() != max_chunk_elements:
                        relayed = torch.empty(max_chunk_elements, dtype=dtype, device=get_device_id())
                        staging_buffers[relay_key] = relayed

                pending: list[torch.Tensor] = []
                pending_elements = 0

                def flush_pending(_staged=staged, _relayed=relayed) -> None:
                    nonlocal pending, pending_elements
                    if not pending:
                        return
                    offset = 0
                    for piece in pending:
                        size = piece.numel()
                        _staged[offset : offset + size].copy_(piece)
                        offset += size
                    communication = _staged
                    if _relayed is not None:
                        _relayed[:pending_elements].copy_(_staged[:pending_elements], non_blocking=True)
                        communication = _relayed
                    torch.distributed.broadcast(communication[:pending_elements], src=0)
                    if rank != 0:
                        offset = 0
                        for piece in pending:
                            size = piece.numel()
                            piece.copy_(communication[offset : offset + size].view_as(piece))
                            offset += size
                    pending = []
                    pending_elements = 0

                for tensor in grouped_tensors:
                    tensor_flat = tensor.reshape(-1)
                    tensor_offset = 0
                    while tensor_offset < tensor_flat.numel():
                        available = max_chunk_elements - pending_elements
                        take = min(available, tensor_flat.numel() - tensor_offset)
                        pending.append(tensor_flat[tensor_offset : tensor_offset + take])
                        pending_elements += take
                        tensor_offset += take
                        if pending_elements == max_chunk_elements:
                            flush_pending()
                flush_pending()

    def _sdc_write_checkpoint(self, local_path: str, global_step: int, config: dict):
        """Write replicated SFT state once, then release every worker together."""

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        is_rank_zero = not distributed or torch.distributed.get_rank() == 0
        status = {"ok": True, "error": None}
        save_start = time.perf_counter() if is_rank_zero else None
        if is_rank_zero:
            try:
                os.makedirs(local_path, exist_ok=True)
                for name, model, optimizer in (
                    ("success", self._sdc_success, self._sdc_success_optimizer),
                    ("failure", self._sdc_failure, self._sdc_failure_optimizer),
                ):
                    directory = os.path.join(local_path, f"{name}_sft")
                    os.makedirs(directory, exist_ok=True)
                    model_state = self._sdc_peft_state(model) if self._is_lora else model.state_dict()
                    # Atomic per-file writes: a crash mid-save cannot leave a
                    # truncated payload that later loads would silently accept.
                    torch.save(model_state, os.path.join(directory, "pytorch_model.bin.tmp"))
                    os.replace(os.path.join(directory, "pytorch_model.bin.tmp"), os.path.join(directory, "pytorch_model.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(directory, "optimizer.pt.tmp"))
                    os.replace(os.path.join(directory, "optimizer.pt.tmp"), os.path.join(directory, "optimizer.pt"))
                base_adapter_state = (
                    self._sdc_adapter_base_state(self._sdc_success) if self._is_lora else None
                )
                # Full-parameter slimming: embed the frozen-base CPU snapshot
                # only on the FIRST save of this engine process. Later saves
                # write a light 'base_saved' marker referencing the earlier
                # step; sdc_load then relies on the init-time clone produced by
                # sdc_reinitialize_from_actor. LoRA checkpoints keep embedding
                # the (small) adapter snapshot every interval.
                embed_base = False
                base_saved_marker = False
                saved_base_step = getattr(self, "_sdc_saved_base_global_step", None)
                if not self._is_lora and self._sdc_base_state_cpu is not None:
                    if saved_base_step is None:
                        embed_base = True
                        self._sdc_saved_base_global_step = int(global_step)
                        saved_base_step = self._sdc_saved_base_global_step
                    else:
                        base_saved_marker = True
                torch.save(
                    build_sdc_checkpoint_state(
                        global_step=global_step,
                        success_model_update_count=self._sdc_success_model_update_count,
                        failure_model_update_count=self._sdc_failure_model_update_count,
                        models_ready=self._sdc_models_ready,
                        base_state=self._sdc_base_state_cpu if embed_base else None,
                        base_adapter_state=base_adapter_state,
                        adapter_only=bool(self._is_lora),
                        base_mix_gamma=self._sdc_base_mix_gamma,
                        success_buffer_state=config.get("success_buffer_state"),
                        failure_buffer_state=config.get("failure_buffer_state"),
                        base_saved=base_saved_marker,
                        base_global_step=saved_base_step,
                    ),
                    os.path.join(local_path, "sdc_state.pt.tmp"),
                )
                os.replace(os.path.join(local_path, "sdc_state.pt.tmp"), os.path.join(local_path, "sdc_state.pt"))
            except Exception as exc:
                status = {"ok": False, "error": repr(exc)}
        if distributed:
            # Broadcast the save status BEFORE the barrier so that a rank-0
            # failure surfaces on every rank instead of leaving ranks 1..N
            # blocked forever at the barrier below.
            status_list = [status]
            torch.distributed.broadcast_object_list(status_list, src=0)
            status = status_list[0]
            torch.distributed.barrier()
        if not status["ok"]:
            raise RuntimeError(f"SDC checkpoint save failed on rank 0: {status['error']}")
        result = {
            "sdc_saved": True,
            "sdc_success_model_update_count": self._sdc_success_model_update_count,
            "sdc_failure_model_update_count": self._sdc_failure_model_update_count,
            "sdc_models_ready": self._sdc_models_ready,
        }
        if save_start is not None:
            result["sdc/timing/checkpoint_save_ms"] = (time.perf_counter() - save_start) * 1000.0
        return result

    def _sdc_sft_model(
        self,
        model,
        optimizer,
        records: list[dict],
        prefix: str,
        config: dict,
        response_token_budget: int | None = None,
    ):
        empty = {
            f"sdc/{prefix}_sft_updates": 0.0,
            f"sdc/{prefix}_sft_loss": 0.0,
            f"sdc/{prefix}_sft_response_tokens": 0.0,
            f"sdc/{prefix}_sft_examples": 0.0,
        }
        if not records:
            return empty
        tokenizer = self._sdc_get_tokenizer()
        device = next(self.module.parameters()).device
        max_prompt_length = int(config.get("max_prompt_length", 1024))
        max_response_length = int(config.get("max_response_length", 2048))
        max_total_length = max_prompt_length + max_response_length
        batch_size = int(config.get("sft_batch_size", 8))
        max_updates = int(config.get("sft_max_updates_per_interval", 1))
        token_budget = int(config.get("sft_max_token_len_per_gpu", 0) or 0)
        if token_budget > 0:
            batch_size = min(batch_size, max(1, token_budget // max_total_length))
        total_loss = 0.0
        total_response_tokens = 0
        updates = 0
        for start in range(0, min(len(records), batch_size * max_updates), batch_size):
            chunks = records[start : start + batch_size]
            rows, masks = [], []
            for record in chunks:
                row, response_mask = encode_outcome_example(
                    OutcomeExample(**record),
                    tokenizer,
                    max_prompt_length=max_prompt_length,
                    max_response_length=max_response_length,
                )
                rows.append(row)
                masks.append(response_mask)
            if response_token_budget is not None:
                remaining = max(0, response_token_budget - total_response_tokens)
                for index, mask in enumerate(masks):
                    keep = min(int(mask.sum()), remaining)
                    if keep < int(mask.sum()):
                        clipped = torch.zeros_like(mask)
                        positions = mask.nonzero(as_tuple=False).flatten()[:keep]
                        clipped[positions] = 1
                        masks[index] = clipped
                    remaining -= keep
            width = max(row.numel() for row in rows)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            # Assemble on CPU and transfer each SFT batch once. The old loop
            # performed a separate host-to-device copy for every example.
            input_ids_cpu = torch.full((len(rows), width), pad_id, dtype=torch.long)
            attention_cpu = torch.zeros_like(input_ids_cpu)
            response_cpu = torch.zeros_like(input_ids_cpu)
            for index, row in enumerate(rows):
                input_ids_cpu[index, : row.numel()] = row
                attention_cpu[index, : row.numel()] = 1
                response_cpu[index, : row.numel()] = masks[index]
            if device.type == "cuda":
                input_ids_cpu = input_ids_cpu.pin_memory()
                attention_cpu = attention_cpu.pin_memory()
                response_cpu = response_cpu.pin_memory()
            input_ids = input_ids_cpu.to(device, non_blocking=True)
            attention = attention_cpu.to(device, non_blocking=True)
            response = response_cpu.to(device, non_blocking=True)
            with _sdc_autocast_context(input_ids.device):
                output = model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True)
                labels = input_ids[:, 1:]
                if output.logits is not None:
                    logits = output.logits[:, :-1]
                    token_log_probs = logprobs_from_logits(
                        logits.view(-1, logits.size(-1)), labels.view(-1), inplace_backward=False
                    ).view(labels.shape)
                elif getattr(output, "log_probs", None) is not None:
                    token_log_probs = output.log_probs.reshape(input_ids.shape)[:, :-1].contiguous()
                else:
                    raise RuntimeError("SDC SFT output contains neither logits nor fused log_probs.")
                loss = outcome_sft_loss(token_log_probs, response[:, 1:].bool())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_response_tokens += int(response.sum().detach().cpu())
            updates += 1
        return {
            f"sdc/{prefix}_sft_updates": float(updates),
            f"sdc/{prefix}_sft_loss": total_loss / max(updates, 1),
            f"sdc/{prefix}_sft_response_tokens": float(total_response_tokens),
            f"sdc/{prefix}_sft_examples": float(min(len(records), batch_size * max_updates)),
        }

    def _sdc_sft_update_local(self, success_records: list[dict], failure_records: list[dict], config: dict):
        self.sdc_init(config)
        # Phase timers cover exactly one refresh; zero them so accumulate-mode
        # recordings below never carry values from previous intervals.
        timings = getattr(self, "_sdc_phase_timings", None)
        if timings is None:
            timings = self._sdc_phase_timings = {}
        for phase_key in ("success_sft_ms", "failure_sft_ms", "base_mixing_ms", "model_sync_ms"):
            timings[phase_key] = 0.0
        if not success_records or not failure_records:
            return {
                **self._sdc_timing_metrics("success_sft_ms", "failure_sft_ms", "base_mixing_ms", "model_sync_ms"),
                "sdc/success_sft_updates": 0.0,
                "sdc/failure_sft_updates": 0.0,
                "sdc/success_sft_response_tokens": 0.0,
                "sdc/failure_sft_response_tokens": 0.0,
                "sdc/success_sft_examples": 0.0,
                "sdc/failure_sft_examples": 0.0,
                "sdc/sft_pair_skipped_unmatched": 1.0,
                "sdc/success_model_update_count": float(self._sdc_success_model_update_count),
                "sdc/failure_model_update_count": float(self._sdc_failure_model_update_count),
                "sdc/models_ready": float(self._sdc_models_ready),
            }
        success_records, failure_records = self._sdc_match_records(success_records, failure_records, config)
        if not success_records or not failure_records:
            return {
                **self._sdc_timing_metrics("success_sft_ms", "failure_sft_ms", "base_mixing_ms", "model_sync_ms"),
                "sdc/success_sft_updates": 0.0,
                "sdc/failure_sft_updates": 0.0,
                "sdc/sft_pair_skipped_unmatched": 1.0,
                "sdc/success_model_update_count": float(self._sdc_success_model_update_count),
                "sdc/failure_model_update_count": float(self._sdc_failure_model_update_count),
                "sdc/models_ready": float(self._sdc_models_ready),
            }
        common_response_tokens = min(
            self._sdc_response_token_count(success_records, config),
            self._sdc_response_token_count(failure_records, config),
        )
        if common_response_tokens <= 0:
            return {
                **self._sdc_timing_metrics("success_sft_ms", "failure_sft_ms", "base_mixing_ms", "model_sync_ms"),
                "sdc/success_sft_updates": 0.0,
                "sdc/failure_sft_updates": 0.0,
                "sdc/sft_pair_skipped_unmatched": 1.0,
                "sdc/success_model_update_count": float(self._sdc_success_model_update_count),
                "sdc/failure_model_update_count": float(self._sdc_failure_model_update_count),
                "sdc/models_ready": float(self._sdc_models_ready),
            }
        success_sft_start = time.perf_counter() if self._sdc_is_rank_zero() else None
        success = self._sdc_sft_model(
            self._sdc_success,
            self._sdc_success_optimizer,
            success_records,
            "success",
            config,
            response_token_budget=common_response_tokens,
        )
        self._sdc_record_timing("success_sft_ms", success_sft_start)
        failure_sft_start = time.perf_counter() if self._sdc_is_rank_zero() else None
        failure = self._sdc_sft_model(
            self._sdc_failure,
            self._sdc_failure_optimizer,
            failure_records,
            "failure",
            config,
            response_token_budget=common_response_tokens,
        )
        self._sdc_record_timing("failure_sft_ms", failure_sft_start)
        success_updates = int(success["sdc/success_sft_updates"])
        failure_updates = int(failure["sdc/failure_sft_updates"])
        if success_updates != failure_updates:
            raise RuntimeError("SDC success/failure refresh produced different optimizer-step counts.")
        success_tokens = int(success["sdc/success_sft_response_tokens"])
        failure_tokens = int(failure["sdc/failure_sft_response_tokens"])
        if success_tokens != failure_tokens:
            raise RuntimeError("SDC success/failure refresh consumed different response-token budgets.")
        if success_updates:
            # Base mixing changes model parameters only.  The two AdamW
            # optimizers retain their independent moments and step counters.
            base_mix_start = time.perf_counter() if self._sdc_is_rank_zero() else None
            self._sdc_mix_with_base(self._sdc_success)
            self._sdc_mix_with_base(self._sdc_failure)
            self._sdc_record_timing("base_mixing_ms", base_mix_start)
            sync_start = time.perf_counter() if self._sdc_is_rank_zero() else None
            if not getattr(self, "_sdc_defer_sync", False):
                self._sdc_sync_model_and_optimizer(self._sdc_success, self._sdc_success_optimizer)
                self._sdc_sync_model_and_optimizer(self._sdc_failure, self._sdc_failure_optimizer)
            self._sdc_record_timing("model_sync_ms", sync_start)
        self._sdc_success_model_update_count += success_updates
        self._sdc_failure_model_update_count += failure_updates
        minimum = int(config.get("min_sft_updates_before_use", 1))
        self._sdc_models_ready = (
            self._sdc_success_model_update_count >= minimum
            and self._sdc_failure_model_update_count >= minimum
        )
        return {
            **success,
            **failure,
            **self._sdc_timing_metrics("success_sft_ms", "failure_sft_ms", "base_mixing_ms", "model_sync_ms"),
            "sdc/sft_pair_skipped_unmatched": 0.0,
            "sdc/sft_pair_response_tokens": min(
                success["sdc/success_sft_response_tokens"], failure["sdc/failure_sft_response_tokens"]
            ),
            "sdc/success_model_update_count": float(self._sdc_success_model_update_count),
            "sdc/failure_model_update_count": float(self._sdc_failure_model_update_count),
            "sdc/models_ready": float(self._sdc_models_ready),
        }

    def sdc_sft_update(self, success_records: list[dict], failure_records: list[dict], config: dict):
        """Run the replicated SFT update once, then synchronize its result.

        ``ONE_TO_ALL`` dispatch gives every engine rank the same records. Only
        rank zero needs to perform the optimizer work; all ranks still enter
        the small result broadcast and the existing model/optimizer sync.
        """

        self.sdc_init(config)
        self._sdc_prepare_models(include_optimizer=True)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed:
            try:
                return self._sdc_sft_update_local(success_records, failure_records, config)
            finally:
                self._sdc_release_models()

        is_rank_zero = torch.distributed.get_rank() == 0
        try:
            self._sdc_defer_sync = True
            try:
                if is_rank_zero:
                    try:
                        result = self._sdc_sft_update_local(success_records, failure_records, config)
                    except Exception as exc:
                        # Ensure other ranks leave the collective instead of
                        # hanging if rank zero encounters a data/model error.
                        result = {"_sdc_failed": True, "_sdc_error": repr(exc)}
                else:
                    result = None
            finally:
                self._sdc_defer_sync = False

            payload = [result]
            torch.distributed.broadcast_object_list(payload, src=0)
            result = payload[0] or {}
            if result.get("_sdc_failed", False):
                raise RuntimeError(f"Rank-zero SDC SFT update failed: {result.get('_sdc_error', 'unknown error')}")
            success_updates = int(result.get("sdc/success_sft_updates", 0.0))
            failure_updates = int(result.get("sdc/failure_sft_updates", 0.0))
            if success_updates or failure_updates:
                if success_updates != failure_updates:
                    raise RuntimeError("SDC success/failure refresh produced different optimizer-step counts.")
                # Rank zero is the only SDC optimizer owner. Every rank needs the
                # refreshed model weights for scoring, but duplicating two complete
                # AdamW states on nonzero ranks wastes tens of gigabytes of host RAM.
                sync_start = time.perf_counter() if is_rank_zero else None
                self._sdc_sync_model_and_optimizer(
                    self._sdc_success,
                    self._sdc_success_optimizer,
                    include_optimizer=False,
                )
                self._sdc_sync_model_and_optimizer(
                    self._sdc_failure,
                    self._sdc_failure_optimizer,
                    include_optimizer=False,
                )
                if sync_start is not None:
                    self._sdc_record_timing("model_sync_ms", sync_start)
                    timings = getattr(self, "_sdc_phase_timings", None) or {}
                    result["sdc/timing/model_sync_ms"] = float(timings.get("model_sync_ms", 0.0))
                if self._is_lora:
                    # Ranks that skipped rank-zero SFT can now release their full
                    # CPU base snapshot too; rank zero captured it during mixing.
                    self._sdc_adapter_base_state(self._sdc_success)
            self._sdc_success_model_update_count = int(
                result.get("sdc/success_model_update_count", self._sdc_success_model_update_count)
            )
            self._sdc_failure_model_update_count = int(
                result.get("sdc/failure_model_update_count", self._sdc_failure_model_update_count)
            )
            self._sdc_models_ready = bool(result.get("sdc/models_ready", self._sdc_models_ready))
        finally:
            self._sdc_release_models()
        if not is_rank_zero:
            # Timings are rank-zero-only measurements; stripping them on the
            # other ranks keeps cross-rank averaging from diluting them.
            result = {key: value for key, value in result.items() if not key.startswith("sdc/timing/")}
        return result

    def _sdc_peft_state(self, model):
        from peft.utils.save_and_load import get_peft_model_state_dict
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items() if torch.is_tensor(v)}

    def _sdc_load_model_state(self, model, state: dict) -> None:
        if self._is_lora:
            try:
                from peft import set_peft_model_state_dict

                set_peft_model_state_dict(model, state)
                return
            except (ImportError, KeyError, RuntimeError, ValueError):
                # Older PEFT versions may not expose the setter. Loading with
                # strict=False preserves compatibility with their key format.
                pass
        model.load_state_dict(state, strict=not self._is_lora)

    def sdc_export_vllm_adapters(self, config: dict):
        self.sdc_init(config)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return {"enabled": False}
        if config.get("scoring_backend") != "vllm" or not self._is_lora:
            return {"enabled": False}
        wrapped = getattr(self.module, "_fsdp_wrapped_module", self.module); peft_config = getattr(wrapped, "peft_config", {}).get("default", None)
        if hasattr(peft_config, "to_dict"): peft_config = peft_config.to_dict()
        return {"enabled": True, "peft_config": peft_config, "success": self._sdc_peft_state(self._sdc_success), "failure": self._sdc_peft_state(self._sdc_failure)}

    def sdc_save_checkpoint(self, local_path: str, global_step: int, config: dict):
        self.sdc_init(config)
        return self._sdc_write_checkpoint(local_path, global_step, config)

    def sdc_load(self, local_path: str, config: dict | None = None):
        config = config or {}
        # Rebuild both clones from the CURRENT actor first. This fully
        # reconstructs ``self._sdc_base_state_cpu`` (rank zero keeps it) BEFORE
        # any checkpoint payload below can overwrite or preserve it, which is
        # what the slim-checkpoint fallback relies on.
        self.sdc_reinitialize_from_actor(config)
        state_path = os.path.join(local_path, "sdc_state.pt")
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        is_sdc_optimizer_owner = not distributed or torch.distributed.get_rank() == 0
        heavy_keys = ("sdc_base_state_cpu", "sdc_base_adapter_state_cpu")
        state: dict = {}
        light: dict | None = None
        load_error: str | None = None
        if os.path.exists(state_path):
            if is_sdc_optimizer_owner:
                try:
                    state = torch.load(state_path, map_location="cpu")
                    # Rank zero alone touches the heavy payload; every other
                    # rank previously torch.loaded the whole file (including a
                    # third full model) only to discard the base tensors.
                    light = {key: value for key, value in state.items() if key not in heavy_keys}
                except Exception as exc:  # pragma: no cover - surfaced below
                    load_error = repr(exc)
            if distributed:
                # Broadcast the failure flag together with the light scalars so
                # a corrupt sidecar fails everywhere instead of hanging NCCL.
                holder = [{"error": load_error, "light": light}]
                torch.distributed.broadcast_object_list(holder, src=0)
                payload = holder[0] or {}
                if payload.get("error"):
                    raise RuntimeError(f"SDC sidecar load failed on rank 0: {payload['error']}")
                light = payload.get("light")
                if not is_sdc_optimizer_owner:
                    state = light or {}
        elif distributed:
            # Keep collective participation symmetric even without a sidecar.
            holder = [None]
            torch.distributed.broadcast_object_list(holder, src=0)
        if load_error:
            # Single-process path: no broadcast surfaced it yet.
            raise RuntimeError(f"SDC sidecar load failed: {load_error}")
        if is_sdc_optimizer_owner and state.get("sdc_base_state_cpu") is not None:
            # Old checkpoints embed the full frozen base; overwrite exactly as
            # before so legacy payloads load bit-for-bit identically.
            self._sdc_base_state_cpu = clone_frozen_state(state["sdc_base_state_cpu"])
        elif is_sdc_optimizer_owner and state.get("base_saved", False):
            # Slim checkpoint ('base_saved' marker, no embedded base): rely on
            # the init-time clone reconstructed above; nothing overwrites it.
            recorded_step = state.get("base_global_step")
            if recorded_step is not None:
                # Continue the slim chain across the resume: the first save
                # after loading emits another marker referencing this base step
                # instead of re-embedding the frozen base snapshot.
                self._sdc_saved_base_global_step = int(recorded_step)
            restored_step = int(config.get("global_policy_step", -1))
            if recorded_step is not None and restored_step >= 0 and int(recorded_step) != restored_step:
                logger.warning(
                    "SDC slim checkpoint references a base captured around policy step %s while "
                    "the restored actor is at step %s; keeping the base reconstructed from the "
                    "restored actor by sdc_reinitialize_from_actor.",
                    recorded_step,
                    restored_step,
                )
        if is_sdc_optimizer_owner and state.get("sdc_base_adapter_state_cpu") is not None:
            self._sdc_base_adapter_state_cpu = clone_frozen_state(state["sdc_base_adapter_state_cpu"])
            if state.get("sdc_adapter_only", False):
                self._sdc_base_state_cpu = None
        self._sdc_base_mix_gamma = float(state.get("base_mix_gamma", self._sdc_base_mix_gamma))
        if not 0.0 <= self._sdc_base_mix_gamma <= 1.0:
            raise ValueError(f"Invalid checkpoint base_mix_gamma={self._sdc_base_mix_gamma}.")
        self._sdc_success_model_update_count = int(state.get("sdc_success_model_update_count", 0))
        self._sdc_failure_model_update_count = int(state.get("sdc_failure_model_update_count", 0))
        self._sdc_models_ready = bool(state.get("sdc_models_ready", False))
        for name, model, optimizer in (
            ("success", self._sdc_success, self._sdc_success_optimizer),
            ("failure", self._sdc_failure, self._sdc_failure_optimizer),
        ):
            directory = os.path.join(local_path, f"{name}_sft")
            model_path = os.path.join(directory, "pytorch_model.bin")
            optimizer_path = os.path.join(directory, "optimizer.pt")
            if os.path.exists(model_path):
                self._sdc_load_model_state(model, torch.load(model_path, map_location="cpu"))
            if is_sdc_optimizer_owner and os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
                self._sdc_move_optimizer_state(optimizer, next(model.parameters()).device)
        return {
            "sdc_loaded": os.path.exists(state_path),
            "sdc_success_model_update_count": self._sdc_success_model_update_count,
            "sdc_failure_model_update_count": self._sdc_failure_model_update_count,
            "sdc_models_ready": self._sdc_models_ready,
            "success_buffer_state": state.get("success_buffer_state"),
            "failure_buffer_state": state.get("failure_buffer_state"),
        }
    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        tag: str = "default",
        save_contents: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Save FSDP checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            tag=tag,
            save_contents=save_contents,
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        # The state-dict gathers above leave large freed blocks cached in this
        torch.cuda.empty_cache()

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load FSDP checkpoint, restoring parameters and optimizer state.
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get("default", None)
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                )
                if not base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:  # merge lora
                with merged_lora_context(self.module, backup_adapters=True):
                    params = self.module.state_dict()
                    params = normalize_peft_param_name(params)
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items()
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer
            from verl.utils.torch_dtypes import PrecisionType

            mixed_precision_config = self.engine_config.mixed_precision
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            else:
                param_dtype = torch.bfloat16

            quantizer = QATQuantizer(
                mode=self._qat_config.mode,
                group_size=self._qat_config.group_size,
                ignore_patterns=list(self._qat_config.ignore_patterns),
                device=torch.device(get_device_id()),
                param_dtype=param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return per_tensor_param, peft_config_dict

    def disable_adapter(self) -> ContextManager:
        return self.module.disable_adapter()


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)
        # keep_grad: the caller deferred its optimizer step and needs the accumulated
        # gradient to survive this context so it can fuse a second backward into ONE step.
        # Zeroing here is otherwise belt-and-braces -- train_batch already zeroes on entry.
        if not self.keep_grad:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithLMHead(FSDPEngine):
    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        temperature_item = temperature
        if use_fused_kernels:
            assert not isinstance(temperature, torch.Tensor), (
                "use_fused_kernels does not support per sample temperature yet"
            )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        # args used to get outputs
        output_args = {}

        if use_remove_padding:
            # support per sample temperature
            # temperature (bsz,)
            # input_ids (bsz, j1)
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
            temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.use_ulysses_sp:
                is_vlm_model = hasattr(getattr(self.module, "module", self.module).config, "vision_config")
                if is_vlm_model:
                    # vlm model's inputs will be sliced after embedding
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                else:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                        skip_position_ids_rmpad=True if self.__class__.__name__ == "VeOmniEngineWithLMHead" else False,
                    )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    position_ids_rmpad=None,
                    sp_size=self.ulysses_sequence_parallel_size,
                )

                temperature_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                    temperature_rmpad, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size, pad_value=1
                )

                output_args["pad_size"] = pad_size

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad

            # only pass input_ids and position_ids to enable flash_attn_varlen

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = int(seq_len_effective.max().item())

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
                # we store the per sample temperature
                output_args["temperature"] = temperature

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)  # (4, batch_size, max_seq_len)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask = build_attention_mask_from_nested(
                    input_ids=micro_batch["input_ids"], max_seq_len=max_seq_len
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature_item
            extra_args["return_dict"] = True
            if use_remove_padding:
                # We have already computed `input_ids_rmpad_rolled` from the *full*
                # global sequence and (when SP>1) SP-sliced it. Pass it into the model
                # so the fused forward uses these labels verbatim instead of redoing
                # `torch.roll` on the local SP shard, which would wrap around the
                # shard boundary rather than the global sequence (issue #6068). This
                # mirrors what the veomni engine already does for fused kernels.
                extra_args["shift_labels"] = output_args["input_ids_rmpad_rolled"].unsqueeze(0)

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(
            data=micro_batch, key="calculate_sum_pi_squared", default=False
        )
        distillation_use_topk = tu.get_non_tensor_data(data=micro_batch, key="distillation_use_topk", default=False)

        if calculate_sum_pi_squared and use_fused_kernels:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True: "
                "fused kernels do not materialize the full logits tensor needed for Σπ²."
            )

        model_output = {}

        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            if use_fused_kernels:
                # temperature is singleton
                log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
            else:
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                # compute entropy
                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

                # compute sum_pi_squared (Σπ²) for optimal-baseline advantage estimators
                if calculate_sum_pi_squared:
                    sum_pi_squared_rmpad = verl_F.calculate_sum_pi_squared_from_logits(logits_rmpad)

                # logits_processor_func return tensors with shape (1, total_nnz/sp_size)
                if distillation_use_topk:
                    outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
                    cu_seqlens = input_ids.offsets()
                    for k, v in outputs.items():
                        v = v.squeeze(0)
                        assert v.shape == log_probs.shape, f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
                        if self.use_ulysses_sp:
                            pad_size = output_args["pad_size"]
                            v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)

            # gather log_prob if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]

                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(
                    log_probs,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                if calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                if calculate_sum_pi_squared:
                    sum_pi_squared_rmpad = gather_outputs_and_unpad(
                        sum_pi_squared_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                if calculate_sum_pi_squared:
                    sum_pi_squared = torch.nested.nested_tensor_from_jagged(sum_pi_squared_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:  # not using rmpad and no ulysses sp
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:
                logits = output.logits  # (bsz, response_length, vocab_size)
                temperature = output_args["temperature"]  # (bsz,)
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if calculate_sum_pi_squared:
                    sum_pi_squared = verl_F.calculate_sum_pi_squared_from_logits(logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                    # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = torch.nested.narrow(
                            sum_pi_squared, 1, starts, seq_lengths, layout=torch.jagged
                        )
                        sum_pi_squared_rmpad = torch.cat([t for t in sum_pi_squared.unbind()])
                        sum_pi_squared = torch.nested.nested_tensor_from_jagged(sum_pi_squared_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy
        if calculate_sum_pi_squared:
            model_output["sum_pi_squared"] = sum_pi_squared


        return model_output


    @staticmethod
    # Diverges from FSDPEngine on purpose: FSDP2 moves every optimizer state
    # tensor (including AdamW "step") unconditionally, while FSDPEngine keeps
    # "step" on CPU unless the param group sets capturable/fused. Do not merge
    # with the parent implementation.
    def _sdc_move_optimizer_state(optimizer, device):
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.data = value.to(device=device, non_blocking=True)


    def sdc_init(self, sdc_config: dict):
        if getattr(self, "_sdc_initialized", False):
            return {"sdc_initialized": True}
        if self.engine_config.strategy not in ("fsdp", "fsdp2"):
            raise NotImplementedError("SDC currently supports only FSDP/FSDP2 HF actors.")
        state = self._sdc_actor_state_cpu()
        self._sdc_base_state_cpu = clone_frozen_state(state)
        # Fresh base snapshot: the next checkpoint save embeds it again and
        # re-anchors the slim-checkpoint marker chain.
        self._sdc_saved_base_global_step = None
        # Invalidate any cached adapter snapshot so a reinit (e.g. from
        # sdc_reinitialize_from_actor after a resume) cannot mix checkpoints
        # or base-interpolate against a PREVIOUS actor's adapter values.
        self._sdc_base_adapter_state_cpu = None
        self._sdc_base_mix_gamma = float(sdc_config.get("base_mix_gamma", 0.9))
        if not 0.0 <= self._sdc_base_mix_gamma <= 1.0:
            raise ValueError(
                f"custom_sdc.base_mix_gamma must be between 0.0 and 1.0, got {self._sdc_base_mix_gamma}."
            )
        sidecar_residency = str(sdc_config.get("sidecar_residency") or "auto_offload")
        if sidecar_residency not in ("auto_offload", "resident"):
            raise ValueError(
                f"custom_sdc.sidecar_residency must be 'auto_offload' or 'resident', got {sidecar_residency!r}."
            )
        self._sdc_sidecar_residency = sidecar_residency
        sidecar_gradient_checkpointing = sdc_config.get("sidecar_gradient_checkpointing", None)
        self._sdc_sidecar_gradient_checkpointing = (
            None if sidecar_gradient_checkpointing is None else bool(sidecar_gradient_checkpointing)
        )
        self._sdc_success = self._sdc_clone_from_state(self._sdc_base_state_cpu)
        self._sdc_failure = self._sdc_clone_from_state(self._sdc_base_state_cpu)
        lr = float(sdc_config.get("sft_lr", 1e-6))
        fused_adamw = bool(sdc_config.get("fused_adamw", False))
        self._sdc_success_optimizer = torch.optim.AdamW(
            (parameter for parameter in self._sdc_success.parameters() if parameter.requires_grad),
            lr=lr,
            fused=fused_adamw,
        )
        self._sdc_failure_optimizer = torch.optim.AdamW(
            (parameter for parameter in self._sdc_failure.parameters() if parameter.requires_grad),
            lr=lr,
            fused=fused_adamw,
        )
        self._sdc_success_model_update_count = int(sdc_config.get("sdc_success_model_update_count", 0))
        self._sdc_failure_model_update_count = int(sdc_config.get("sdc_failure_model_update_count", 0))
        self._sdc_models_ready = bool(sdc_config.get("sdc_models_ready", False))
        self._sdc_tokenizer = None
        self._sdc_initialized = True
        actor_counts = {
            "total": sum(parameter.numel() for parameter in self.module.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad),
        }
        success_counts = self._sdc_parameter_counts(self._sdc_success, self._sdc_success_optimizer)
        failure_counts = self._sdc_parameter_counts(self._sdc_failure, self._sdc_failure_optimizer)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed and torch.distributed.get_rank() != 0 and not self._is_lora:
            # Only rank zero interpolates SDC weights with the frozen actor.
            # Scoring ranks retain the two synchronized models but not this
            # extra full-model CPU snapshot.
            self._sdc_base_state_cpu = None
        # In resident mode keep the freshly built clones on the actor GPU; the
        # explicit sdc_release_residency() hook does the release later.
        self._sdc_release_models()
        return {
            "sdc_initialized": True,
            "sdc/base_mix_gamma": self._sdc_base_mix_gamma,
            "sdc/actor_total_parameters": actor_counts["total"],
            "sdc/actor_trainable_parameters": actor_counts["trainable"],
            "sdc/success_total_parameters": success_counts["total"],
            "sdc/success_trainable_parameters": success_counts["trainable"],
            "sdc/success_optimizer_parameters": success_counts["optimizer"],
            "sdc/failure_total_parameters": failure_counts["total"],
            "sdc/failure_trainable_parameters": failure_counts["trainable"],
            "sdc/failure_optimizer_parameters": failure_counts["optimizer"],
        }










    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        exploration = getattr(self, "exploration", None)
        if not forward_only and exploration is not None and exploration.enabled:
            tu.assign_non_tensor_data(micro_batch, "calculate_entropy", True)
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        # Honor mixed_precision.param_dtype resolved during FSDP setup. When dtype is fp32,
        # autocast is a no-op at best and a footgun at worst, so skip it entirely.
        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__
        # and _build_fsdp_module, so self._autocast_dtype may not be set.
        autocast_dtype = getattr(self, "_autocast_dtype", torch.bfloat16)
        autocast_ctx: ContextManager = (
            nullcontext()
            if autocast_dtype == torch.float32
            else torch.autocast(device_type=device_name, dtype=autocast_dtype)
        )
        with autocast_ctx:
            ema_log_probs = None
            if not forward_only and exploration is not None and exploration.should_run_ema_forward():
                saved = exploration.ema_tracker.load_into_model(self.module)
                old_calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
                tu.assign_non_tensor_data(micro_batch, "calculate_entropy", False)
                try:
                    with torch.no_grad():
                        raw_ema_output = self.module(
                            **model_inputs,
                            use_cache=False,
                        )
                        ema_model_output = self.prepare_model_outputs(
                            output=raw_ema_output,
                            output_args=output_args,
                            micro_batch=micro_batch,
                            logits_processor_func=loss_function,
                        )
                        ema_log_probs = ema_model_output["log_probs"].detach()
                finally:
                    tu.assign_non_tensor_data(micro_batch, "calculate_entropy", old_calculate_entropy)
                    exploration.ema_tracker.restore_model(self.module, saved)

            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch, logits_processor_func=loss_function
            )
            if ema_log_probs is not None:
                model_output["ema_log_probs"] = ema_log_probs

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithValueHead(FSDPEngineWithLMHead):
    """
    The only difference between critic and actor is how the raw model output is processed
    """

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values_rmpad = output[2].squeeze(0)
            else:
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz, 1)
                # critic model arch is like Qwen3ForTokenClassfication and num_labels=1
                # so we squeeze the last dimension here to get the value for each token
                values_rmpad = values_rmpad.squeeze(-1)

            # gather output if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values = output[2]
            else:
                values = output.logits.squeeze(-1)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                values = torch.nested.narrow(values, 1, starts, seq_lengths, layout=torch.jagged)
                values_rmpad = torch.cat([t for t in values.unbind()])
                # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        return {"values": values}