from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from verl.experimental.sdc_grpo.data_collector import OutcomeExample


@dataclass
class SFTStepResult:
    loss: float
    token_count: int
    batch_size: int


def outcome_sft_loss(log_prob: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    mask = response_mask.to(device=log_prob.device, dtype=log_prob.dtype)
    token_count = mask.sum(dim=-1).clamp_min(1.0)
    sequence_loss = -(log_prob * mask).sum(dim=-1) / token_count
    valid = (mask.sum(dim=-1) > 0).to(log_prob.dtype)
    return (sequence_loss * valid).sum() / valid.sum().clamp_min(1.0)


def encode_outcome_example(
    example: OutcomeExample,
    tokenizer,
    *,
    max_prompt_length: int,
    max_response_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode an outcome example without using the response limit as total length.

    The prompt is truncated independently from the left, while the assistant
    response keeps its own limit. This preserves response-token supervision for
    a long prompt plus a long response.
    """

    prompt = {"role": "user", "content": str(example.prompt)}
    assistant = {"role": "assistant", "content": str(example.response)}
    prompt_ids = list(tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True))
    full_ids = list(tokenizer.apply_chat_template([prompt, assistant], add_generation_prompt=False, tokenize=True))
    response_ids = full_ids[len(prompt_ids) : len(prompt_ids) + max_response_length]
    prompt_ids = prompt_ids[-max_prompt_length:]
    ids = torch.as_tensor(prompt_ids + response_ids, dtype=torch.long)
    response_mask = torch.zeros_like(ids)
    response_mask[len(prompt_ids) :] = 1
    return ids, response_mask


def configure_sft_trainable_parameters(module: nn.Module, *, lora_enabled: bool) -> list[nn.Parameter]:
    """Keep PEFT base weights frozen while retaining adapter trainability."""

    if lora_enabled:
        trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
    else:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        trainable = list(module.parameters())
    if not trainable:
        raise ValueError("SDC SFT model has no trainable parameters.")
    return trainable


def pair_outcome_records(
    success_records: list[dict],
    failure_records: list[dict],
    *,
    max_examples: int,
) -> tuple[list[dict], list[dict]]:
    """Select the same number of examples without consuming either buffer."""

    if not success_records or not failure_records or max_examples <= 0:
        return [], []
    count = min(len(success_records), len(failure_records), max_examples)
    return success_records[:count], failure_records[:count]


def mix_state_with_base(
    current_state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    *,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Apply the same common-base interpolation to one outcome model."""

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    mixed = {}
    for name, value in current_state.items():
        base = base_state.get(name)
        if base is None or not torch.is_floating_point(value):
            mixed[name] = value
        else:
            mixed[name] = gamma * value.detach() + (1.0 - gamma) * base.to(value.device, value.dtype)
    return mixed


class OutcomeSFTTrainer:
    """Response-only SFT trainer with an optimizer separate from the actor."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

    def assert_separate_from(self, other_optimizer: torch.optim.Optimizer) -> None:
        other_ids = {id(p) for group in other_optimizer.param_groups for p in group["params"]}
        if self.optimizer_param_ids & other_ids:
            raise AssertionError("Outcome-SFT optimizer shares parameters with the GRPO optimizer.")

    def step_tensor_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor) -> SFTStepResult:
        self.model.train()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1]
        labels = input_ids[:, 1:]
        shifted_mask = response_mask[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        loss = outcome_sft_loss(log_probs, shifted_mask)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return SFTStepResult(float(loss.detach().cpu()), int(shifted_mask.sum().cpu()), int(input_ids.shape[0]))

    @staticmethod
    def build_text_batch(
        examples: Iterable[OutcomeExample],
        tokenizer,
        max_length: int | None = None,
        *,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
    ):
        encoded_rows, response_masks = [], []
        for example in examples:
            if max_prompt_length is not None and max_response_length is not None:
                ids, mask = encode_outcome_example(
                    example,
                    tokenizer,
                    max_prompt_length=max_prompt_length,
                    max_response_length=max_response_length,
                )
            else:
                if max_length is None:
                    raise ValueError("Provide max_length or separate prompt/response limits.")
                prompt = {"role": "user", "content": example.prompt}
                assistant = {"role": "assistant", "content": example.response}
                prompt_ids = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
                full_ids = tokenizer.apply_chat_template([prompt, assistant], add_generation_prompt=False, tokenize=True)
                ids = torch.as_tensor(full_ids[:max_length], dtype=torch.long)
                mask = torch.zeros_like(ids)
                mask[min(len(prompt_ids), ids.numel()) :] = 1
            encoded_rows.append(ids)
            response_masks.append(mask)
        if not encoded_rows:
            raise ValueError("Cannot build an SFT batch from zero examples.")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        width = max(row.numel() for row in encoded_rows)
        input_ids = torch.full((len(encoded_rows), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        response_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(encoded_rows):
            input_ids[i, : ids.numel()] = ids
            attention_mask[i, : ids.numel()] = 1
            response_mask[i, : ids.numel()] = response_masks[i]
        return input_ids, attention_mask, response_mask
