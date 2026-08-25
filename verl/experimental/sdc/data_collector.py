from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import torch


def _as_id_array(ids: Any) -> Optional[np.ndarray]:
    """Normalize token ids into one compact int64 numpy array.

    Python int lists cost roughly an order of magnitude more memory per token
    than int64 arrays and pickle far less compactly.  Lists are converted
    exactly once, here.
    """
    if ids is None:
        return None
    return np.asarray(ids, dtype=np.int64)


def validate_sdc_outcome(outcome: torch.Tensor | Any) -> torch.BoolTensor:
    """Validate one exact semantic 0/1 outcome per response row."""

    values = torch.as_tensor(outcome)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError(f"sdc_outcome must contain one scalar per row, got shape {tuple(values.shape)}")
    values = values.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sdc_outcome cannot contain NaN or Inf.")
    valid = (values == 0.0) | (values == 1.0)
    if not bool(valid.all()):
        invalid = values[~valid].tolist()
        raise ValueError(f"sdc_outcome must contain exact 0/1 values, received {invalid[:8]}")
    return values.to(dtype=torch.bool)


def extract_sdc_outcome(batch, reward_extra_infos: Optional[dict[str, Any]] = None) -> torch.BoolTensor:
    """Read the verifier outcome without looking at shaped token rewards."""

    if "sdc_outcome" in batch.batch:
        return validate_sdc_outcome(batch.batch["sdc_outcome"])
    if "acc" in batch.batch:
        return validate_sdc_outcome(batch.batch["acc"])
    extras = reward_extra_infos or {}
    for key in ("sdc_outcome", "acc", "correct", "correctness"):
        values = extras.get(key)
        if values is not None:
            return validate_sdc_outcome(values)
    raise ValueError(
        "SDC requires an explicit verifier correctness field (sdc_outcome/acc); "
        "it will not infer outcomes from shaped reward tensors."
    )


@dataclass
class OutcomeExample:
    prompt: Any
    response: Any
    reward: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # When available, keep the already-tokenized prompt/response. This avoids
    # decode -> chat-template -> tokenize churn before every SFT refresh.
    # Ids are stored as compact int64 numpy arrays (converted once on add);
    # plain int lists are accepted transparently at the constructor boundary.
    prompt_ids: Optional[np.ndarray] = None
    response_ids: Optional[np.ndarray] = None


class OutcomeDataCollector:
    """Bounded collector that accepts only one configured semantic outcome."""

    def __init__(
        self,
        target_reward: int,
        max_size: Optional[int] = None,
        sampling: str = "recent",
        seed: Optional[int] = None,
    ):
        if target_reward not in (0, 1):
            raise ValueError("target_reward must be 0 or 1")
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive or None")
        if sampling not in {"recent", "uniform"}:
            raise ValueError("sampling must be 'recent' or 'uniform'")
        self.target_reward = int(target_reward)
        self.max_size = max_size
        self.sampling = sampling
        self._rng = random.Random(seed)
        self._examples = deque(maxlen=max_size) if sampling == "recent" else []
        self.total_seen = 0
        # Incremental minimum explicit policy step among retained examples,
        # maintained on add/consume so buffer_age() is O(1) instead of a scan.
        # _min_policy_step_count tracks how many retained examples carry that
        # exact step so removals know when the cached minimum went stale and a
        # rescan is needed.
        self._min_policy_step: Optional[int] = None
        self._min_policy_step_count: int = 0

    def __len__(self) -> int:
        return len(self._examples)

    @staticmethod
    def _example_policy_step(metadata: dict[str, Any]) -> Optional[int]:
        value = metadata.get("global_policy_step", metadata.get("global_grpo_step"))
        return int(value) if value is not None else None

    def _track_step_on_add(self, example: OutcomeExample) -> None:
        step = self._example_policy_step(example.metadata)
        if step is None:
            return
        if self._min_policy_step is None or step < self._min_policy_step:
            self._min_policy_step = step
            self._min_policy_step_count = 1
        elif step == self._min_policy_step:
            self._min_policy_step_count += 1

    def _track_steps_on_remove(self, removed: list[OutcomeExample]) -> None:
        # Call only AFTER the examples are already out of self._examples: a
        # rescan triggered here observes the final retained set.
        for example in removed:
            if self._min_policy_step is None:
                return
            step = self._example_policy_step(example.metadata)
            if step is not None and step == self._min_policy_step:
                self._min_policy_step_count -= 1
                if self._min_policy_step_count <= 0:
                    self._recompute_min_policy_step()
                    return

    def _recompute_min_policy_step(self) -> None:
        steps = [s for s in map(self._example_policy_step, (e.metadata for e in self._examples)) if s is not None]
        if steps:
            self._min_policy_step = min(steps)
            self._min_policy_step_count = steps.count(self._min_policy_step)
        else:
            self._min_policy_step = None
            self._min_policy_step_count = 0

    def add(
        self,
        *,
        prompt: Any,
        response: Any,
        reward: float | int,
        metadata: Optional[dict[str, Any]] = None,
        prompt_ids: Optional[list[int]] = None,
        response_ids: Optional[list[int]] = None,
    ) -> bool:
        try:
            value = float(reward)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value) or value not in (0.0, 1.0) or int(value) != self.target_reward:
            return False
        example = OutcomeExample(
            prompt,
            response,
            self.target_reward,
            metadata or {},
            prompt_ids=_as_id_array(prompt_ids),
            response_ids=_as_id_array(response_ids),
        )
        self.total_seen += 1
        if self.sampling == "recent":
            self._track_step_on_add(example)
            self._examples.append(example)
        elif self.max_size is None or len(self._examples) < self.max_size:
            self._track_step_on_add(example)
            self._examples.append(example)
        else:
            index = self._rng.randrange(self.total_seen)
            if index < self.max_size:
                # Overwrite BEFORE removal tracking: _track_steps_on_remove()
                # must observe the post-eviction retained set when it rescans.
                evicted = self._examples[index]
                self._examples[index] = example
                self._track_steps_on_remove([evicted])
                self._track_step_on_add(example)
        return True

    def add_many(self, rows: Iterable[dict[str, Any]]) -> int:
        return sum(int(self.add(**row)) for row in rows)

    def sample(self, batch_size: int) -> list[OutcomeExample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        examples = list(self._examples)
        if len(examples) <= batch_size:
            return examples
        return examples[-batch_size:] if self.sampling == "recent" else self._rng.sample(examples, batch_size)

    def clear(self) -> None:
        self._examples.clear()
        self._min_policy_step = None
        self._min_policy_step_count = 0

    def consume(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if self.sampling == "recent":
            removed = []
            for _ in range(min(count, len(self._examples))):
                removed.append(self._examples.popleft())
            self._track_steps_on_remove(removed)
        else:
            bound = min(count, len(self._examples))
            removed = [self._examples[i] for i in range(bound)]
            del self._examples[:bound]
            self._track_steps_on_remove(removed)

    def to_records(self) -> list[dict[str, Any]]:
        # Boundary format choice: keep prompt_ids/response_ids as int64 numpy
        # arrays.  Every consumer (Ray pickling to SFT workers, torch.save of
        # state_dict, encode_outcome_example's list()/slicing/torch.as_tensor)
        # handles numpy arrays directly and they pickle far more compactly
        # than Python int lists, so lists are no longer materialized here.
        return [asdict(example) for example in self._examples]

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_reward": self.target_reward,
            "max_size": self.max_size,
            "sampling": self.sampling,
            "total_seen": self.total_seen,
            "examples": self.to_records(),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("target_reward", self.target_reward)) != self.target_reward:
            raise ValueError("Outcome collector checkpoint has the wrong target reward.")
        if state.get("sampling", self.sampling) != self.sampling:
            raise ValueError("Outcome collector checkpoint has the wrong sampling policy.")
        self.clear()
        for row in state.get("examples", []):
            self.add(
                prompt=row.get("prompt"),
                response=row.get("response"),
                reward=row.get("reward"),
                metadata=row.get("metadata"),
                prompt_ids=row.get("prompt_ids"),
                response_ids=row.get("response_ids"),
            )
        self.total_seen = int(state.get("total_seen", len(self._examples)))
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])

    def buffer_age(self, current_policy_step: int) -> float:
        # O(1) via the incrementally maintained minimum explicit policy step.
        # Equivalent to the old per-call scan: examples without an explicit
        # step key used the query-time step as a default, which never lowered
        # min() below any explicitly recorded step; when NO retained example
        # carries an explicit step, both versions return 0.0.
        if not self._examples or self._min_policy_step is None:
            return 0.0
        return float(max(0, int(current_policy_step) - self._min_policy_step))


class SuccessDataCollector(OutcomeDataCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, target_reward=1, **kwargs)


class FailureDataCollector(OutcomeDataCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, target_reward=0, **kwargs)


# Boundary compatibility for old callers.  New code must pass the verifier's
# semantic field rather than a shaped token reward tensor.
def binary_outcome_scores(outcome_tensor: torch.Tensor) -> torch.BoolTensor:
    return validate_sdc_outcome(outcome_tensor)
