from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from verl.experimental.sdc.data_collector import OutcomeExample


@dataclass
class SFTStepResult:
    loss: float
    token_count: int
    batch_size: int


def outcome_sft_loss(log_prob: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    mask = response_mask.to(device=log_prob.device, dtype=log_prob.dtype)
    token_count = mask.sum(dim=-1)
    valid = token_count > 0
    sequence_loss = -(log_prob * mask).sum(dim=-1) / token_count.clamp_min(1.0)
    return (sequence_loss * valid.to(log_prob.dtype)).sum() / valid.to(log_prob.dtype).sum().clamp_min(1.0)


def encode_outcome_example(
    example: OutcomeExample,
    tokenizer,
    *,
    max_prompt_length: int,
    max_response_length: int,
):
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
    if lora_enabled:
        trainable = [p for p in module.parameters() if p.requires_grad]
        for p in module.parameters():
            p.requires_grad_(False)
        for p in trainable:
            p.requires_grad_(True)
    else:
        for p in module.parameters():
            p.requires_grad_(True)
        trainable = list(module.parameters())
    if not trainable:
        raise ValueError("SDC SFT model has no trainable parameters.")
    return trainable


def pair_outcome_records(success_records: list[dict], failure_records: list[dict], *, max_examples: int):
    if not success_records or not failure_records or max_examples <= 0:
        return [], []
    count = min(len(success_records), len(failure_records), max_examples)
    return success_records[:count], failure_records[:count]


class OutcomeSFTTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model, self.optimizer = model, optimizer
        self.optimizer_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}

    def assert_separate_from(self, other_optimizer: torch.optim.Optimizer) -> None:
        other = {id(p) for g in other_optimizer.param_groups for p in g["params"]}
        if self.optimizer_param_ids & other:
            raise AssertionError("Outcome-SFT optimizer shares parameters with another optimizer.")

    @staticmethod
    def build_text_batch(
        examples: Iterable[OutcomeExample],
        tokenizer,
        max_length: int | None = None,
        *,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
    ):
        rows, masks = [], []
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
                full_ids = tokenizer.apply_chat_template(
                    [prompt, assistant], add_generation_prompt=False, tokenize=True
                )
                ids = torch.as_tensor(full_ids[:max_length], dtype=torch.long)
                mask = torch.zeros_like(ids)
                mask[min(len(prompt_ids), ids.numel()) :] = 1
            rows.append(ids)
            masks.append(mask)
        if not rows:
            raise ValueError("Cannot build an SFT batch from zero examples.")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        width = max(row.numel() for row in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        response = torch.zeros_like(input_ids)
        for i, row in enumerate(rows):
            input_ids[i, : row.numel()] = row
            attention[i, : row.numel()] = 1
            response[i, : row.numel()] = masks[i]
        return input_ids, attention, response
