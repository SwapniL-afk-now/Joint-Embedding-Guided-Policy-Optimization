# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Assemble the mixed online/offline candidate group for GPI-CE.

vLLM generates only ``K_on`` responses per prompt. After the rollout replicas are
asleep, this builder appends ``K_off`` pre-tokenized offline responses per prompt,
producing ``B * K`` rows laid out group-contiguously:

    prompt 0: online 0..K_on-1, offline 0..K_off-1
    prompt 1: online 0..K_on-1, offline 0..K_off-1
    ...

Offline rows are built by cloning the prompt half of that prompt's first online
row and splicing the offline response into the response half. Cloning rather than
re-tokenizing guarantees the prompt encoding, padding and position ids are
bit-identical to the online candidates for the same prompt -- the group softmax is
only meaningful if every candidate is conditioned on exactly the same context.
"""

from __future__ import annotations

import numpy as np
import torch

from verl.protocol import DataProto


def _offline_token_lists(row_extra: dict, num_needed: int, allow_short: bool, prompt_id) -> list[list[int]]:
    """Pull the pre-tokenized offline responses stashed on the dataset row."""
    stored = row_extra.get("gpi_offline_input_ids", None)
    if stored is None:
        raise ValueError(
            f"prompt {prompt_id!r} has no 'gpi_offline_input_ids'; the dataset must be "
            "recipe GPICEDataset (data.custom_cls) and the parquet must carry offline_responses."
        )
    stored = [list(map(int, seq)) for seq in stored if len(seq) > 0]
    if len(stored) < num_needed:
        if not allow_short:
            raise ValueError(
                f"prompt {prompt_id!r} has {len(stored)} usable offline responses but "
                f"offline_per_prompt={num_needed}. Set custom_gpi_ce.allow_short_offline=true "
                "to cycle the available ones, or fix the dataset."
            )
        if not stored:
            raise ValueError(f"prompt {prompt_id!r} has zero usable offline responses.")
        stored = [stored[i % len(stored)] for i in range(num_needed)]
    return stored[:num_needed]


class MixedCandidateBuilder:
    """Builds the B x K mixed batch from a B x K_on online batch."""

    def __init__(self, pad_token_id: int, allow_short_offline: bool = False):
        self.pad_token_id = int(pad_token_id)
        self.allow_short_offline = bool(allow_short_offline)

    def build(self, online_batch: DataProto, online_per_prompt: int, offline_per_prompt: int) -> DataProto:
        k_on, k_off = int(online_per_prompt), int(offline_per_prompt)
        group_size = k_on + k_off
        num_rows = len(online_batch.batch["input_ids"])
        if num_rows % k_on != 0:
            raise ValueError(f"online batch has {num_rows} rows, not a multiple of online_per_prompt={k_on}")
        num_prompts = num_rows // k_on

        uids = np.asarray(online_batch.non_tensor_batch["uid"], dtype=object)
        for b in range(num_prompts):
            grp = uids[b * k_on : (b + 1) * k_on]
            if len(set(grp)) != 1:
                raise ValueError(
                    f"online rows for prompt {b} span multiple uids {set(grp)}; GPI-CE requires the "
                    "generated batch to stay group-contiguous (do not shuffle or balance before building)."
                )

        if k_off == 0:
            out = online_batch
        else:
            out = self._splice(online_batch, num_prompts, k_on, k_off)

        n = len(out.batch["input_ids"])
        candidate_id = np.tile(np.arange(group_size, dtype=np.int64), num_prompts)
        is_offline = np.tile(
            np.array([False] * k_on + [True] * k_off, dtype=bool),
            num_prompts,
        )
        out.non_tensor_batch["gpi_candidate_id"] = candidate_id
        out.non_tensor_batch["gpi_source"] = np.array(
            ["offline" if f else "online" for f in is_offline], dtype=object
        )
        out.batch["gpi_is_offline"] = torch.from_numpy(is_offline)
        out.batch["gpi_group_id"] = torch.arange(num_prompts, dtype=torch.long).repeat_interleave(group_size)

        assert n == num_prompts * group_size, f"mixed batch has {n} rows, expected {num_prompts * group_size}"
        final_uids = np.asarray(out.non_tensor_batch["uid"], dtype=object).reshape(num_prompts, group_size)
        assert all(len(set(row)) == 1 for row in final_uids), "a mixed group contains rows from different prompts"
        return out

    def _splice(self, online: DataProto, num_prompts: int, k_on: int, k_off: int) -> DataProto:
        responses = online.batch["responses"]
        response_len = responses.shape[1]
        prompt_len = online.batch["input_ids"].shape[1] - response_len
        position_ids = online.batch.get("position_ids", None)
        if position_ids is not None and position_ids.dim() != 2:
            raise NotImplementedError(
                "GPI-CE offline splicing supports 2D position_ids only; multi-modal (3D) rope is not supported."
            )

        # Row indices into a batch that already contains the offline rows as copies of
        # the group's first online row; overwrite the response half of the copies below.
        template_index = []
        for b in range(num_prompts):
            template_index.extend(range(b * k_on, (b + 1) * k_on))  # the real online rows
            template_index.extend([b * k_on] * k_off)  # placeholders to overwrite
        mixed = online.select_idxs(torch.as_tensor(template_index, dtype=torch.long))

        group_size = k_on + k_off
        extras = online.non_tensor_batch.get("extra_info", None)
        if extras is None:
            raise ValueError("GPI-CE needs non_tensor_batch['extra_info'] to carry the offline responses.")

        offline_rows = []  # (dest_row, token_ids)
        for b in range(num_prompts):
            token_lists = _offline_token_lists(
                dict(extras[b * k_on] or {}),
                k_off,
                self.allow_short_offline,
                online.non_tensor_batch["uid"][b * k_on],
            )
            for j, ids in enumerate(token_lists):
                offline_rows.append((b * group_size + k_on + j, ids[:response_len]))

        input_ids = mixed.batch["input_ids"]
        attention_mask = mixed.batch["attention_mask"]
        new_responses = mixed.batch["responses"]
        for dest, ids in offline_rows:
            length = len(ids)
            tokens = torch.full((response_len,), self.pad_token_id, dtype=new_responses.dtype)
            tokens[:length] = torch.as_tensor(ids, dtype=new_responses.dtype)
            new_responses[dest] = tokens
            input_ids[dest, prompt_len:] = tokens

            resp_mask = torch.zeros(response_len, dtype=attention_mask.dtype)
            resp_mask[:length] = 1
            attention_mask[dest, prompt_len:] = resp_mask
            if position_ids is not None:
                # Continue the prompt's own numbering; padded tail values are masked out.
                last = mixed.batch["position_ids"][dest, prompt_len - 1]
                mixed.batch["position_ids"][dest, prompt_len:] = last + torch.arange(
                    1, response_len + 1, dtype=position_ids.dtype
                )

        mixed.batch["response_mask"] = attention_mask[:, -response_len:].clone()
        # Rollout log-probs describe generated tokens only; an offline row has none, so
        # drop the field rather than let a stale copy be mistaken for one.
        for key in ("rollout_log_probs", "old_log_probs"):
            if key in mixed.batch.keys():
                mixed.batch.pop(key)
        return mixed
