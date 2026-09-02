from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from verl.experimental.sdc.data_collector import OutcomeExample
from verl.experimental.sdc.sft_trainer import encode_outcome_example, pair_outcome_records_prompt_matched
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    fsdp2_load_full_state_dict,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import logprobs_from_logits


class SDCShardedTeacherPair:
    """Own two native FSDP teacher modules and their independent optimizers.

    The actor engine supplies the model/FSDP construction primitives. This keeps
    the teacher topology identical to the actor while ensuring every teacher
    parameter and optimizer tensor is sharded on every rank.
    """

    FORMAT_VERSION = 2

    def __init__(self, owner, config: dict[str, Any]):
        self.owner = owner
        self.config = dict(config or {})
        self.strategy = str(owner.engine_config.strategy)
        self.teacher_mode = str(self.config.get("teacher_mode", "full_fsdp"))
        self.shared_lora = self.teacher_mode == "shared_lora"
        self.group = owner.get_data_parallel_group()
        self.dp_size = owner.get_data_parallel_size()
        self.rank = owner.get_data_parallel_rank()
        self.device = get_device_id()
        self.gamma = float(self.config.get("base_mix_gamma", 0.9))
        self.lr = float(self.config.get("sft_lr", 1e-6))
        self.fused_adamw = bool(self.config.get("fused_adamw", False))
        self.teachers: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.checkpoint_managers: dict[str, Any] = {}
        self.base_state: dict[str, torch.Tensor] = {}
        self.base_metadata: dict[str, Any] = {}
        self.update_counts = {"success": 0, "failure": 0}
        self.models_ready = False
        self.initialized = False
        self._resident = False
        self._active_outcome = "success"

    @staticmethod
    def _distributed() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _barrier(self) -> None:
        if self._distributed():
            torch.distributed.barrier(group=self.group)

    def _all_reduce(self, value: torch.Tensor, op=torch.distributed.ReduceOp.SUM) -> torch.Tensor:
        if self._distributed():
            torch.distributed.all_reduce(value, op=op, group=self.group)
        return value

    @staticmethod
    def _local_parameter(parameter: torch.Tensor) -> torch.Tensor:
        if hasattr(parameter, "to_local"):
            return parameter.to_local()
        if hasattr(parameter, "_local_tensor"):
            return parameter._local_tensor
        return parameter.data

    def _adapter_parameters(self, module: torch.nn.Module, adapter: str) -> list[torch.nn.Parameter]:
        marker = f"lora_A.{adapter}"
        marker_b = f"lora_B.{adapter}"
        parameters = [parameter for name, parameter in module.named_parameters() if marker in name or marker_b in name]
        if not parameters:
            raise RuntimeError(f"Unable to find trainable parameters for SDC adapter {adapter!r}.")
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters

    def _activate_adapter(self, outcome: str) -> None:
        if not self.shared_lora:
            return
        self._active_outcome = outcome
        self.teachers["success"].set_adapter("default" if outcome == "success" else "failure")

    def _new_teacher(self) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        # Full-parameter SDC is intentionally independent of actor LoRA state.
        # The shared-backbone LoRA implementation is selected separately and is
        # not allowed to silently change this exact mode.
        if self.shared_lora:
            if int(getattr(self.owner.model_config, "lora_rank", 0) or 0) <= 0:
                raise ValueError("shared_lora SDC requires actor model.lora_rank > 0")
            module = self.owner._build_lora_module(self.owner._build_module())
            module.add_adapter("failure", module.peft_config["default"])
            module = self.owner._build_fsdp_module(module)
            module.train(True)
            optimizer = torch.optim.AdamW(
                self._adapter_parameters(module, "default"),
                lr=self.lr,
                fused=self.fused_adamw,
            )
            return module, optimizer
        if int(getattr(self.owner.model_config, "lora_rank", 0) or 0) > 0:
            raise ValueError("full_fsdp SDC teachers require actor model.lora_rank=0")
        module = self.owner._build_module()
        module = self.owner._build_fsdp_module(module)
        module.train(True)
        optimizer = torch.optim.AdamW(
            (p for p in module.parameters() if p.requires_grad),
            lr=self.lr,
            fused=self.fused_adamw,
        )
        return module, optimizer

    def _copy_actor_local_state(self, teacher: torch.nn.Module) -> None:
        actor_parameters = dict(self.owner.module.named_parameters())
        teacher_parameters = dict(teacher.named_parameters())
        with torch.no_grad():
            for name, target in teacher_parameters.items():
                source = actor_parameters.get(name)
                if source is None and self.shared_lora:
                    source = actor_parameters.get(name.replace(".failure.", ".default."))
                if source is None:
                    raise RuntimeError(f"SDC teacher parameter {name} does not match the actor layout")
                source_local = self._local_parameter(source)
                target_local = self._local_parameter(target)
                if source_local.shape != target_local.shape:
                    raise RuntimeError(f"SDC teacher local parameter shard shape differs for {name}")
                target_local.copy_(source_local.to(device=target_local.device, dtype=target_local.dtype))
            actor_buffers = dict(self.owner.module.named_buffers())
            for name, target in teacher.named_buffers():
                source = actor_buffers.get(name)
                if source is not None and source.shape == target.shape:
                    target.copy_(source.to(device=target.device, dtype=target.dtype))

    def _capture_base(self, teacher: torch.nn.Module) -> None:
        self.base_state = {}
        for name, parameter in teacher.named_parameters():
            local = self._local_parameter(parameter)
            self.base_state[name] = local.detach().to(device="cpu", copy=True)
        self.base_metadata = {
            "parameter_names": tuple(self.base_state),
            "parameter_shapes": {name: tuple(value.shape) for name, value in self.base_state.items()},
            "world_size": self.dp_size,
            "strategy": self.strategy,
        }

    def initialize(self) -> dict[str, Any]:
        if self.initialized:
            return self.metrics()
        if self.strategy not in ("fsdp", "fsdp2"):
            raise NotImplementedError("Sharded SDC teachers require the HF FSDP or FSDP2 engine")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"custom_sdc.base_mix_gamma must be between 0 and 1, got {self.gamma}")
        outcomes = ("success",) if self.shared_lora else ("success", "failure")
        for outcome in outcomes:
            teacher, optimizer = self._new_teacher()
            self._copy_actor_local_state(teacher)
            self.teachers[outcome] = teacher
            self.optimizers[outcome] = optimizer
        if self.shared_lora:
            self.teachers["failure"] = self.teachers["success"]
            self.optimizers["failure"] = torch.optim.AdamW(
                self._adapter_parameters(self.teachers["success"], "failure"),
                lr=self.lr,
                fused=self.fused_adamw,
            )
        self._capture_base(self.teachers["success"])
        self._activate_adapter("success")
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

        processing_class = self.owner.model_config.get_processor()
        self.checkpoint_managers = {
            outcome: FSDPCheckpointManager(
                model=self.teachers[outcome],
                optimizer=self.optimizers[outcome],
                processing_class=processing_class,
                checkpoint_config=self.owner.checkpoint_config,
                trust_remote_code=self.owner.model_config.trust_remote_code,
            )
            for outcome in ("success", "failure")
        }
        self.initialized = True
        self._resident = True
        self._release_if_needed()
        return self.metrics()

    def metrics(self) -> dict[str, Any]:
        return {
            "sdc_initialized": self.initialized,
            "sdc/base_mix_gamma": self.gamma,
            "sdc/teacher_strategy": self.strategy,
            "sdc/teacher_sharded": True,
            "sdc/teacher_world_size": self.dp_size,
            "sdc/success_total_parameters": sum(p.numel() for p in self.teachers.get("success", []).parameters()) if "success" in self.teachers else 0,
            "sdc/failure_total_parameters": sum(p.numel() for p in self.teachers.get("failure", []).parameters()) if "failure" in self.teachers else 0,
        }

    def _load_models(self) -> None:
        if self._resident:
            return
        for teacher in self.teachers.values():
            load_fsdp_model_to_gpu(teacher)
        self._resident = True

    def _offload_models(self) -> None:
        if not self._resident:
            return
        for optimizer in self.optimizers.values():
            offload_fsdp_optimizer(optimizer)
        for teacher in self.teachers.values():
            offload_fsdp_model_to_cpu(teacher, empty_cache=False)
        self._resident = False
        get_torch_device().empty_cache()

    def _release_if_needed(self) -> None:
        if str(self.config.get("sidecar_residency") or "resident") == "auto_offload":
            self._offload_models()

    def release_residency(self) -> None:
        self._offload_models()

    def reinitialize_from_actor(self, config: dict[str, Any]) -> dict[str, Any]:
        self._offload_models()
        self.config.update(config or {})
        self.gamma = float(self.config.get("base_mix_gamma", self.gamma))
        self.initialized = False
        self.teachers.clear()
        self.optimizers.clear()
        self.base_state.clear()
        self.update_counts = {"success": 0, "failure": 0}
        self.models_ready = False
        return self.initialize()

    @staticmethod
    def _dense_data(data: TensorDict) -> dict[str, torch.Tensor]:
        padded = data.to_padded_tensor()
        return {key: value for key, value in padded.items() if torch.is_tensor(value)}

    def _forward_log_probs(
        self,
        teacher: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        was_training = teacher.training
        teacher.eval()
        try:
            autocast = (
                torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16)
                if input_ids.device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
            )
            with torch.no_grad(), autocast:
                output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                labels = input_ids[:, 1:].contiguous()
                if getattr(output, "logits", None) is not None:
                    logits = output.logits[:, :-1].contiguous()
                    token_lp = logprobs_from_logits(
                        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), inplace_backward=False
                    ).reshape(labels.shape)
                elif getattr(output, "log_probs", None) is not None:
                    token_lp = output.log_probs.reshape(input_ids.shape)[:, :-1].contiguous()
                else:
                    raise RuntimeError("SDC teacher output contains neither logits nor fused log_probs.")
                width = response_mask.shape[-1]
                prompt_width = input_ids.shape[-1] - width
                result = torch.zeros(input_ids.shape[0], width, device=input_ids.device, dtype=torch.float32)
                start = max(prompt_width - 1, 0)
                available = min(width, max(token_lp.shape[-1] - start, 0))
                if available:
                    result[:, :available] = token_lp[:, start : start + available].float()
                return result
        finally:
            teacher.train(was_training)

    def compute_log_probs(self, data: TensorDict) -> TensorDict:
        self.initialize()
        values = self._dense_data(data)
        ids = values["input_ids"].to(self.device)
        attention = values["attention_mask"].to(self.device)
        response = values["response_mask"].to(self.device)
        active = values.get("sdc_failure_mask", torch.ones(ids.shape[0], dtype=torch.bool)).to(self.device).bool().reshape(-1)
        local_active = torch.tensor([int(active.any())], device=self.device, dtype=torch.int32)
        self._all_reduce(local_active, op=torch.distributed.ReduceOp.MAX)
        success = torch.zeros(ids.shape[0], response.shape[-1], device=self.device, dtype=torch.float32)
        failure = torch.zeros_like(success)
        if int(local_active.item()):
            self._load_models()
            indices = active.nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                indices = torch.zeros(1, dtype=torch.long, device=self.device)
                active_ids, active_attention, active_response = ids[:1], attention[:1], response[:1] * 0
            else:
                active_ids, active_attention, active_response = ids[indices], attention[indices], response[indices]
            micro = max(int(self.config.get("vllm_score_micro_batch_size", 16) or 16), 1)
            success_active, failure_active = [], []
            for start in range(0, active_ids.shape[0], micro):
                end = min(start + micro, active_ids.shape[0])
                self._activate_adapter("success")
                success_active.append(self._forward_log_probs(self.teachers["success"], active_ids[start:end], active_attention[start:end], active_response[start:end]))
                self._activate_adapter("failure")
                failure_active.append(self._forward_log_probs(self.teachers["failure"], active_ids[start:end], active_attention[start:end], active_response[start:end]))
            if active.any():
                success[indices] = torch.cat(success_active, dim=0)
                failure[indices] = torch.cat(failure_active, dim=0)
            self._release_if_needed()
        return TensorDict(
            {"sdc_success_log_probs": success.cpu(), "sdc_failure_log_probs": failure.cpu()},
            batch_size=[ids.shape[0]],
        )

    @staticmethod
    def _clip_paired_records(
        success_records: list[dict],
        failure_records: list[dict],
        max_response_length: int,
        budget: int,
    ) -> tuple[list[dict], list[dict]]:
        success_clipped, failure_clipped = [], []
        remaining = max(int(budget), 0)
        for success, failure in zip(success_records, failure_records, strict=True):
            success_item, failure_item = dict(success), dict(failure)
            success_ids = success_item.get("response_ids")
            failure_ids = failure_item.get("response_ids")
            success_length = min(len(success_ids), max_response_length) if success_ids is not None else max_response_length
            failure_length = min(len(failure_ids), max_response_length) if failure_ids is not None else max_response_length
            keep = min(success_length, failure_length, remaining)
            if success_ids is not None:
                success_item["response_ids"] = list(success_ids)[:keep]
            if failure_ids is not None:
                failure_item["response_ids"] = list(failure_ids)[:keep]
            success_clipped.append(success_item)
            failure_clipped.append(failure_item)
            remaining -= keep
            if remaining == 0:
                break
        return success_clipped, failure_clipped

    def _encode_batch(
        self,
        records: list[dict],
        tokenizer,
        max_prompt_length: int,
        max_response_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, masks = [], []
        for record in records:
            row, mask = encode_outcome_example(
                OutcomeExample(**record),
                tokenizer,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
            )
            rows.append(row)
            masks.append(mask)
        width = max((row.numel() for row in rows), default=1)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        response = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : row.numel()] = row
            attention[index, : row.numel()] = 1
            response[index, : row.numel()] = masks[index]
        return ids, attention, response

    def _sft_loss(self, teacher: torch.nn.Module, ids, attention, response, global_valid: int) -> torch.Tensor:
        output = teacher(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
        labels = ids[:, 1:].contiguous()
        if getattr(output, "logits", None) is not None:
            logits = output.logits[:, :-1].contiguous()
            token_lp = logprobs_from_logits(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), inplace_backward=True
            ).reshape(labels.shape)
        elif getattr(output, "log_probs", None) is not None:
            token_lp = output.log_probs.reshape(ids.shape)[:, :-1].contiguous()
        else:
            raise RuntimeError("SDC teacher output contains neither logits nor fused log_probs.")
        mask = response[:, 1:].bool()
        token_count = mask.sum(dim=-1)
        valid = token_count > 0
        sequence_loss = -(token_lp * mask).sum(dim=-1) / token_count.clamp_min(1)
        local_sum = sequence_loss[valid].sum()
        return local_sum / max(global_valid, 1) * self.dp_size

    def _sft_outcome(self, outcome: str, records: list[dict], config: dict[str, Any], tokenizer, common_budget: int) -> dict[str, float]:
        batch_size = int(config.get("sft_batch_size", 8))
        max_updates = int(config.get("sft_max_updates_per_interval", 1))
        max_prompt = int(config.get("max_prompt_length", 1024))
        max_response = int(config.get("max_response_length", 2048))
        records = list(records[: batch_size * max_updates])
        real_count = len(records)
        if not records:
            return {f"sdc/{outcome}_sft_updates": 0.0, f"sdc/{outcome}_sft_loss": 0.0, f"sdc/{outcome}_sft_response_tokens": 0.0, f"sdc/{outcome}_sft_examples": 0.0}
        self._load_models()
        self._activate_adapter(outcome)
        teacher = self.teachers[outcome]
        optimizer = self.optimizers[outcome]
        load_fsdp_optimizer(optimizer, self.device)
        total_loss = 0.0
        total_tokens = 0
        updates = 0
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            # Every rank receives the same number of examples and therefore the
            # same number of FSDP collectives. Masked dummies cover remainders.
            padded_count = ((len(batch_records) + self.dp_size - 1) // self.dp_size) * self.dp_size
            dummy = {"prompt": "", "response": "", "prompt_ids": [0], "response_ids": [], "reward": 0, "metadata": {}}
            batch_records.extend([dummy] * (padded_count - len(batch_records)))
            local_records = batch_records[self.rank : padded_count : self.dp_size]
            ids, attention, response = self._encode_batch(local_records, tokenizer, max_prompt, max_response)
            ids, attention, response = ids.to(self.device), attention.to(self.device), response.to(self.device)
            local_valid = torch.tensor([int((response.sum(-1) > 0).sum())], device=self.device, dtype=torch.int64)
            global_valid = int(self._all_reduce(local_valid).item())
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16) if ids.device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
            with autocast:
                loss = self._sft_loss(teacher, ids, attention, response, global_valid)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_tokens += int(self._all_reduce(torch.tensor([int(response.sum())], device=self.device, dtype=torch.int64)).item())
            updates += 1
        return {
            f"sdc/{outcome}_sft_updates": float(updates),
            f"sdc/{outcome}_sft_loss": total_loss / max(updates, 1),
            f"sdc/{outcome}_sft_response_tokens": float(total_tokens),
            f"sdc/{outcome}_sft_examples": float(real_count),
        }

    def _mix_base(self, teacher: torch.nn.Module, outcome: str = "success") -> None:
        adapter_name = "default" if outcome == "success" else outcome
        with torch.no_grad():
            for name, parameter in teacher.named_parameters():
                if self.shared_lora and not (
                    f"lora_A.{adapter_name}" in name or f"lora_B.{adapter_name}" in name
                ):
                    continue
                base = self.base_state.get(name)
                if base is None:
                    raise RuntimeError(f"Missing immutable SDC base shard for {name}")
                local = self._local_parameter(parameter)
                mixed = local.float() * self.gamma + base.to(device=local.device, dtype=torch.float32) * (1.0 - self.gamma)
                local.copy_(mixed.to(dtype=local.dtype))

    def sft_update(self, success_records: list[dict], failure_records: list[dict], config: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        max_examples = int(config.get("sft_batch_size", 8)) * int(config.get("sft_max_updates_per_interval", 1))
        success_records, failure_records = pair_outcome_records_prompt_matched(
            success_records,
            failure_records,
            max_examples=max_examples,
        )
        if not success_records or not failure_records:
            return {"sdc/success_sft_updates": 0.0, "sdc/failure_sft_updates": 0.0, "sdc/sft_pair_skipped_unmatched": 1.0}
        tokenizer = self.owner._sdc_get_tokenizer()
        max_response = int(config.get("max_response_length", 2048))
        def response_budget(records):
            total = 0
            for record in records:
                response_ids = record.get("response_ids")
                total += min(len(response_ids), max_response) if response_ids is not None else max_response
            return total

        common_budget = min(response_budget(success_records), response_budget(failure_records))
        success_records, failure_records = self._clip_paired_records(
            success_records, failure_records, max_response, common_budget
        )
        if common_budget <= 0:
            return {"sdc/success_sft_updates": 0.0, "sdc/failure_sft_updates": 0.0, "sdc/sft_pair_skipped_unmatched": 1.0}
        started = time.perf_counter()
        success = self._sft_outcome("success", success_records, config, tokenizer, common_budget)
        failure = self._sft_outcome("failure", failure_records, config, tokenizer, common_budget)
        self._mix_base(self.teachers["success"], "success")
        self._mix_base(self.teachers["failure"], "failure")
        success_updates = int(success["sdc/success_sft_updates"])
        failure_updates = int(failure["sdc/failure_sft_updates"])
        if success_updates != failure_updates:
            raise RuntimeError("SDC success/failure SFT must take the same number of optimizer steps")
        self.update_counts["success"] += success_updates
        self.update_counts["failure"] += failure_updates
        minimum = int(config.get("min_sft_updates_before_use", 1))
        self.models_ready = self.update_counts["success"] >= minimum and self.update_counts["failure"] >= minimum
        self._release_if_needed()
        return {
            **success,
            **failure,
            "sdc/sft_pair_skipped_unmatched": 0.0,
            "sdc/sft_pair_response_tokens": float(min(success["sdc/success_sft_response_tokens"], failure["sdc/failure_sft_response_tokens"])),
            "sdc/success_model_update_count": float(self.update_counts["success"]),
            "sdc/failure_model_update_count": float(self.update_counts["failure"]),
            "sdc/models_ready": float(self.models_ready),
            "sdc/timing/sharded_sft_ms": (time.perf_counter() - started) * 1000.0 if self.rank == 0 else 0.0,
        }

    def _save_rank_state(self, path: Path, outcome: str) -> None:
        teacher = self.teachers[outcome]
        params = {
            name: self._local_parameter(parameter).detach().cpu()
            for name, parameter in teacher.named_parameters()
        }
        torch.save(
            {"version": self.FORMAT_VERSION, "parameters": params, "optimizer": self.optimizers[outcome].state_dict()},
            path / f"{outcome}_rank_{self.rank}_world_{self.dp_size}.pt",
        )

    def _write_base_shard(self, path: Path, global_step: int) -> None:
        torch.save(
            {
                "version": self.FORMAT_VERSION,
                "base_state": self.base_state,
                "base_metadata": self.base_metadata,
                "global_step": int(global_step),
                "update_counts": dict(self.update_counts),
                "models_ready": bool(self.models_ready),
                "gamma": self.gamma,
            },
            path / f"base_rank_{self.rank}_world_{self.dp_size}.pt",
        )

    def save_checkpoint(self, local_path: str, global_step: int, config: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        self._load_models()
        # Native manager output is the authoritative v2 model/optimizer format.
        # Both managers are invoked by every rank in the same order.
        for outcome in ("success", "failure"):
            load_fsdp_optimizer(self.optimizers[outcome], self.device)
            try:
                self.checkpoint_managers[outcome].save_checkpoint(
                    local_path=str(path / f"{outcome}_sft"),
                    global_step=int(global_step),
                    max_ckpt_to_keep=None,
                    save_contents=["model", "optimizer", "extra"],
                )
            finally:
                offload_fsdp_optimizer(self.optimizers[outcome])
        self._write_base_shard(path, global_step)
        if self.rank == 0:
            runtime = {
                "version": self.FORMAT_VERSION,
                "global_step": int(global_step),
                "update_counts": dict(self.update_counts),
                "models_ready": bool(self.models_ready),
                "gamma": self.gamma,
                "config": dict(config or {}),
            }
            torch.save(runtime, path / "runtime.pt")
            manifest = {
                "format_version": self.FORMAT_VERSION,
                "world_size": self.dp_size,
                "strategy": self.strategy,
                "global_step": int(global_step),
                "update_counts": dict(self.update_counts),
                "models_ready": bool(self.models_ready),
                "base_metadata": self.base_metadata,
                "teacher_paths": {"success": "success_sft", "failure": "failure_sft"},
            }
            torch.save(manifest, path / "manifest.pt.tmp")
            os.replace(path / "manifest.pt.tmp", path / "manifest.pt")
            (path / "_SUCCESS").touch()
        self._barrier()
        self._release_if_needed()
        return {
            "sdc_saved": True,
            "sdc_checkpoint_format": self.FORMAT_VERSION,
            "sdc_success_model_update_count": self.update_counts["success"],
            "sdc_failure_model_update_count": self.update_counts["failure"],
            "sdc_models_ready": self.models_ready,
        }

    def _load_rank_state(self, path: Path, outcome: str) -> None:
        state_path = path / f"{outcome}_rank_{self.rank}_world_{self.dp_size}.pt"
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        teacher = self.teachers[outcome]
        with torch.no_grad():
            for name, parameter in teacher.named_parameters():
                if name not in payload["parameters"]:
                    raise RuntimeError(f"Missing sharded SDC parameter {name}")
                self._local_parameter(parameter).copy_(payload["parameters"][name].to(self._local_parameter(parameter).device))
        self.optimizers[outcome].load_state_dict(payload["optimizer"])

    def _load_full_state(self, teacher: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
        if fsdp_version(teacher) == 2:
            fsdp2_load_full_state_dict(teacher, state, self.owner.device_mesh, None)
            return
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            teacher,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
        ):
            teacher.load_state_dict(state, strict=True)

    @staticmethod
    def _legacy_named_optimizer_state(model: torch.nn.Module, payload: dict[str, Any]) -> dict[str, Any]:
        named_parameters = list(model.named_parameters())
        flat_ids = [parameter_id for group in payload.get("param_groups", []) for parameter_id in group.get("params", [])]
        if len(flat_ids) != len(named_parameters) or len(set(flat_ids)) != len(flat_ids):
            raise RuntimeError("Legacy SDC optimizer parameter IDs cannot be mapped bijectively to teacher FQNs.")
        id_to_name = {parameter_id: name for parameter_id, (name, _) in zip(flat_ids, named_parameters)}
        state = {}
        for parameter_id, values in payload.get("state", {}).items():
            if parameter_id not in id_to_name:
                raise RuntimeError(f"Legacy SDC optimizer references unknown parameter ID {parameter_id}.")
            state[id_to_name[parameter_id]] = values
        if len(state) not in (0, len(named_parameters)):
            raise RuntimeError("Legacy SDC optimizer state has incomplete parameter coverage.")
        groups = []
        for group in payload.get("param_groups", []):
            group_copy = dict(group)
            group_copy["params"] = [id_to_name[parameter_id] for parameter_id in group.get("params", [])]
            groups.append(group_copy)
        return {"state": state, "param_groups": groups}

    def _load_legacy_optimizer(self, teacher, optimizer, payload: dict[str, Any]) -> None:
        named_state = self._legacy_named_optimizer_state(teacher, payload)
        if not named_state["state"]:
            return
        if fsdp_version(teacher) == 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            sharded = FSDP.optim_state_dict_to_load(
                teacher,
                optimizer,
                named_state,
                is_named_optimizer=True,
                group=self.group,
            )
            optimizer.load_state_dict(sharded)
            return
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

        set_optimizer_state_dict(
            teacher,
            optimizer,
            named_state,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

    def _legacy_payload(self, path: Path) -> dict[str, Any]:
        payload = None
        error = None
        if self.rank == 0:
            try:
                state_path = path / "sdc_state.pt"
                if not state_path.exists():
                    raise FileNotFoundError(f"Neither SDC v2 manifest nor legacy state exists at {path}")
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                base = state.get("sdc_base_state_cpu")
                if base is None and state.get("base_saved", False):
                    base_step = state.get("base_global_step")
                    if base_step is None:
                        raise RuntimeError("Slim legacy SDC checkpoint has no base_global_step.")
                    base_path = path.parent.parent / f"global_step_{int(base_step)}" / "sdc" / "sdc_state.pt"
                    if not base_path.exists():
                        raise FileNotFoundError(
                            f"Legacy SDC checkpoint references missing immutable base checkpoint {base_path}"
                        )
                    base = torch.load(base_path, map_location="cpu", weights_only=False).get("sdc_base_state_cpu")
                if base is None:
                    raise RuntimeError("Legacy SDC checkpoint has no immutable base state.")
                models, optimizers = {}, {}
                for outcome in ("success", "failure"):
                    model_path = path / f"{outcome}_sft" / "pytorch_model.bin"
                    optimizer_path = path / f"{outcome}_sft" / "optimizer.pt"
                    if not model_path.exists() or not optimizer_path.exists():
                        raise FileNotFoundError(f"Legacy {outcome} SDC model/optimizer is incomplete at {path}")
                    models[outcome] = torch.load(model_path, map_location="cpu", weights_only=False)
                    optimizers[outcome] = torch.load(optimizer_path, map_location="cpu", weights_only=False)
                payload = {"state": state, "base": base, "models": models, "optimizers": optimizers}
            except Exception as exc:
                error = repr(exc)
        if self._distributed():
            holder = [{"error": error, "payload": payload}]
            torch.distributed.broadcast_object_list(holder, src=0, group=self.group)
            error, payload = holder[0].get("error"), holder[0].get("payload")
        if error:
            raise RuntimeError(f"SDC legacy migration failed: {error}")
        if payload is None:
            raise RuntimeError("Legacy SDC migration received no payload.")
        return payload

    def _migrate_legacy_checkpoint(self, path: Path, config: dict[str, Any]) -> dict[str, Any]:
        payload = self._legacy_payload(path)
        state = payload["state"]
        for outcome in ("success", "failure"):
            self._load_full_state(self.teachers[outcome], payload["base"])
            if outcome == "success":
                self._capture_base(self.teachers[outcome])
            self._load_full_state(self.teachers[outcome], payload["models"][outcome])
            self._load_legacy_optimizer(self.teachers[outcome], self.optimizers[outcome], payload["optimizers"][outcome])
        self.update_counts = {
            "success": int(state.get("sdc_success_model_update_count", 0)),
            "failure": int(state.get("sdc_failure_model_update_count", 0)),
        }
        self.models_ready = bool(state.get("sdc_models_ready", False))
        self.gamma = float(state.get("base_mix_gamma", self.gamma))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"Invalid legacy checkpoint base_mix_gamma={self.gamma}.")
        self._release_if_needed()
        result = self.save_checkpoint(str(path), int(config.get("global_policy_step", state.get("global_policy_step", 0))), config)
        result["sdc_migrated_from_v1"] = True
        result["success_buffer_state"] = state.get("success_buffer_state")
        result["failure_buffer_state"] = state.get("failure_buffer_state")
        return result

    def load_checkpoint(self, local_path: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.initialize()
        config = config or {}
        path = Path(local_path)
        manifest_path = path / "manifest.pt"
        marker = path / "_SUCCESS"
        is_v2 = manifest_path.exists() and marker.exists()
        if not is_v2:
            return self._migrate_legacy_checkpoint(path, config)
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        if int(manifest.get("format_version", -1)) != self.FORMAT_VERSION:
            raise RuntimeError(f"Unsupported SDC checkpoint format {manifest.get("format_version")}")
        if int(manifest.get("world_size", -1)) != self.dp_size:
            raise RuntimeError("SDC checkpoint world size differs from the active teacher mesh.")
        for outcome in ("success", "failure"):
            if not (path / f"{outcome}_sft").is_dir():
                raise RuntimeError(f"SDC v2 checkpoint is missing the {outcome} native shard directory.")
        base_path = path / f"base_rank_{self.rank}_world_{self.dp_size}.pt"
        runtime_path = path / "runtime.pt"
        if not runtime_path.exists():
            raise RuntimeError("SDC v2 checkpoint is incomplete: runtime state is missing.")
        local_error = None if base_path.exists() else f"missing immutable base shard: {base_path}"
        for outcome in ("success", "failure"):
            manager = self.checkpoint_managers[outcome]
            outcome_path = path / f"{outcome}_sft"
            required = (
                outcome_path / f"model_world_size_{manager.world_size}_rank_{manager.rank}.pt",
                outcome_path / f"optim_world_size_{manager.world_size}_rank_{manager.rank}.pt",
                outcome_path / f"extra_state_world_size_{manager.world_size}_rank_{manager.rank}.pt",
            )
            missing = [str(item) for item in required if not item.exists()]
            if missing:
                local_error = f"missing native SDC shard(s): {missing}"
                break
        if self._distributed():
            status = torch.tensor([int(local_error is not None)], device=self.device, dtype=torch.int32)
            self._all_reduce(status, op=torch.distributed.ReduceOp.MAX)
            if int(status.item()):
                holder = [local_error]
                torch.distributed.broadcast_object_list(holder, src=0, group=self.group)
                raise RuntimeError(holder[0] or "SDC v2 checkpoint has a missing rank shard.")
        elif local_error:
            raise RuntimeError(local_error)
        base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
        self.base_state = {name: value.clone() for name, value in base_payload["base_state"].items()}
        self.base_metadata = dict(base_payload.get("base_metadata", {}))
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        self._load_models()
        for outcome in ("success", "failure"):
            self.checkpoint_managers[outcome].load_checkpoint(
                local_path=str(path / f"{outcome}_sft"),
                del_local_after_load=False,
            )
        self.update_counts = {
            "success": int(runtime.get("update_counts", {}).get("success", 0)),
            "failure": int(runtime.get("update_counts", {}).get("failure", 0)),
        }
        self.models_ready = bool(runtime.get("models_ready", False))
        self.gamma = float(runtime.get("gamma", self.gamma))
        self._release_if_needed()
        return {
            "sdc_loaded": True,
            "sdc_checkpoint_format": self.FORMAT_VERSION,
            "sdc_success_model_update_count": self.update_counts["success"],
            "sdc_failure_model_update_count": self.update_counts["failure"],
            "sdc_models_ready": self.models_ready,
            "success_buffer_state": runtime.get("config", {}).get("success_buffer_state"),
            "failure_buffer_state": runtime.get("config", {}).get("failure_buffer_state"),
        }

    def export_vllm_adapters(self, config: dict[str, Any]) -> dict[str, Any]:
        if not self.shared_lora or self.rank != 0:
            return {"enabled": False}
        module = self.teachers["success"]
        peft_config = getattr(module, "peft_config", {}).get("default")
        if peft_config is None:
            wrapped = getattr(module, "_fsdp_wrapped_module", module)
            peft_config = getattr(wrapped, "peft_config", {}).get("default")
        if hasattr(peft_config, "to_dict"):
            peft_config = peft_config.to_dict()
        from verl.utils.fsdp_utils import collect_lora_params

        adapters = {}
        for outcome in ("success", "failure"):
            self._activate_adapter(outcome)
            state = collect_lora_params(module, layered_summon=True, base_sync_done=True)
            # PEFT intentionally strips the adapter name from exported
            # keys; the active adapter selected above determines the payload.
            adapters[outcome] = {
                name: value.detach().cpu().clone()
                for name, value in state.items()
                if torch.is_tensor(value)
            }
        self._activate_adapter("success")
        if not adapters["success"] or not adapters["failure"]:
            raise RuntimeError("shared_lora export produced an empty adapter state")
        return {"enabled": True, "peft_config": peft_config, **adapters}
