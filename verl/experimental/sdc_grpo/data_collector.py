from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class OutcomeExample:
    prompt: Any
    response: Any
    reward: int
    metadata: dict[str, Any] = field(default_factory=dict)


class OutcomeDataCollector:
    """Bounded collector for exactly one binary outcome."""

    def __init__(self, target_reward: int, max_size: Optional[int] = None, sampling: str = "recent", seed: Optional[int] = None):
        if target_reward not in (0, 1):
            raise ValueError("target_reward must be 0 or 1")
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive or None")
        if sampling not in {"recent", "uniform"}:
            raise ValueError("sampling must be 'recent' or 'uniform'")
        self.target_reward = target_reward
        self.max_size = max_size
        self.sampling = sampling
        self._rng = random.Random(seed)
        self._examples = deque(maxlen=max_size) if sampling == "recent" else []
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self._examples)

    def add(self, *, prompt: Any, response: Any, reward: float | int, metadata: Optional[dict[str, Any]] = None) -> bool:
        if int(float(reward)) != self.target_reward or float(reward) not in (0.0, 1.0):
            return False
        example = OutcomeExample(prompt, response, self.target_reward, metadata or {})
        self.total_seen += 1
        if self.sampling == "recent":
            self._examples.append(example)
        elif self.max_size is None or len(self._examples) < self.max_size:
            self._examples.append(example)
        else:
            index = self._rng.randrange(self.total_seen)
            if index < self.max_size:
                self._examples[index] = example
        return True

    def add_many(self, rows: Iterable[dict[str, Any]]) -> int:
        return sum(int(self.add(prompt=row.get("prompt"), response=row.get("response"), reward=row.get("reward"), metadata=row.get("metadata"))) for row in rows)

    def sample(self, batch_size: int) -> list[OutcomeExample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        examples = list(self._examples)
        if len(examples) <= batch_size:
            return examples
        return examples[-batch_size:] if self.sampling == "recent" else self._rng.sample(examples, batch_size)

    def clear(self) -> None:
        self._examples.clear()

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(example) for example in self._examples]


class SuccessDataCollector(OutcomeDataCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(target_reward=1, *args, **kwargs)


class FailureDataCollector(OutcomeDataCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(target_reward=0, *args, **kwargs)
