# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dataset that carries verified offline responses alongside each prompt.

Wire it up with::

    data:
      custom_cls:
        path: verl/experimental/gpi_ce/dataset.py
        name: GPICEDataset

Each parquet row adds two columns to the usual RLHF schema::

    offline_responses: list[str]     # distinct verified solutions for THIS prompt
    offline_rewards:   list[float]   # optional; diagnostics only unless verify_offline=false

Offline responses are tokenized once at init (no BOS re-added, truncated to
``data.max_response_length``) and stashed under ``extra_info.gpi_offline_input_ids``.
They are deliberately NOT expanded into rows here -- expansion happens after vLLM
generation so online and offline candidates land in one group.
"""

from __future__ import annotations

import logging

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class GPICEDataset(RLHFDataset):
    """RLHFDataset plus pre-tokenized offline candidates per prompt."""

    OFFLINE_RESPONSE_KEY = "offline_responses"
    OFFLINE_REWARD_KEY = "offline_rewards"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_response_length = self.config.get("max_response_length", 2048)
        self._offline_cache: dict[int, dict] = {}
        self._warned_missing = False

    def _tokenize_offline(self, item: int, row: dict) -> dict:
        cached = self._offline_cache.get(item)
        if cached is not None:
            return cached

        texts = row.get(self.OFFLINE_RESPONSE_KEY, None)
        texts = [] if texts is None else [str(t) for t in texts]
        rewards = row.get(self.OFFLINE_REWARD_KEY, None)
        rewards = [] if rewards is None else [float(r) for r in rewards]

        token_ids, kept_texts, kept_rewards = [], [], []
        for idx, text in enumerate(texts):
            # add_special_tokens=False: the prompt half already carries BOS, and the
            # response half of an online rollout never contains one either.
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            truncated = len(ids) > self.max_response_length
            ids = ids[: self.max_response_length]
            if not truncated and self.tokenizer.eos_token_id is not None and ids[-1] != self.tokenizer.eos_token_id:
                # Online rollouts terminate with EOS; keep the offline ones comparable.
                ids = ids[: self.max_response_length - 1] + [self.tokenizer.eos_token_id]
            token_ids.append(ids)
            kept_texts.append(text)
            kept_rewards.append(rewards[idx] if idx < len(rewards) else 1.0)

        if not token_ids and not self._warned_missing:
            logger.warning(
                "GPICEDataset: row %s has no usable %s. Every prompt needs offline candidates "
                "unless custom_gpi_ce.offline_per_prompt=0.",
                item,
                self.OFFLINE_RESPONSE_KEY,
            )
            self._warned_missing = True

        cached = {
            "gpi_offline_input_ids": token_ids,
            "gpi_offline_text": kept_texts,
            "gpi_offline_reward": kept_rewards,
            "gpi_offline_length": [len(i) for i in token_ids],
        }
        self._offline_cache[item] = cached
        return cached

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        raw = self.dataframe[item]
        row_dict["extra_info"] = {**(row_dict.get("extra_info") or {}), **self._tokenize_offline(item, raw)}
        # Offline text is only needed by the reward path; drop the big column itself.
        row_dict.pop(self.OFFLINE_RESPONSE_KEY, None)
        row_dict.pop(self.OFFLINE_REWARD_KEY, None)
        return row_dict
