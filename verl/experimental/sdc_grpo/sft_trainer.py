from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    def build_text_batch(examples: Iterable[OutcomeExample], tokenizer, max_length: int):
        encoded_rows, response_masks = [], []
        for example in examples:
            prompt = {"role": "user", "content": example.prompt}
            assistant = {"role": "assistant", "content": example.response}
            prompt_ids = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
            full_ids = tokenizer.apply_chat_template([prompt, assistant], add_generation_prompt=False, tokenize=True)
            ids = torch.as_tensor(full_ids[:max_length], dtype=torch.long)
            mask = torch.zeros_like(ids)
            mask[min(len(prompt_ids), ids.numel()):] = 1
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
