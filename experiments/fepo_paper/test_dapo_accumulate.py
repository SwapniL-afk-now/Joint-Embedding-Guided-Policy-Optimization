#!/usr/bin/env python3
"""Self-check for RayPPOTrainer._dapo_accumulate (DAPO dynamic sampling).

Exercises the accumulator against a stub trainer -- no ray, no GPU, no model. What
it protects: the accumulator must never hand back a short batch while it is still
allowed to generate, must never exceed train_batch_size groups, and must keep whole
groups intact (a split group would corrupt every group-relative advantage).

    python experiments/fepo_paper/test_dapo_accumulate.py
"""

import itertools
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

N_ROLLOUTS = 4  # responses per prompt


_call_counter = itertools.count()


def make_batch(groups):
    """groups: list of per-group score lists. Builds a DataProto shaped like a rollout.

    Stamps global_token_num into meta_info, exactly as ray_trainer.py does before
    reward scoring -- this is what caught the real crash: DataProto.concat asserts
    non-metric meta_info is IDENTICAL across inputs, and this key is a per-batch-size
    list, so two differently-sized carried batches always conflicted here in the
    actual trainer despite the plain-batch unit tests below passing.
    """
    # A monotonic counter, not id(groups): CPython can reuse a freed list's address
    # for the very next literal, so id()-based uids collide across calls with no
    # warning -- this bit once already, as flakiness that looked like a real bug.
    call = next(_call_counter)
    scores, uids = [], []
    for gi, g in enumerate(groups):
        scores.extend(g)
        uids.extend([f"uid{call}-{gi}"] * len(g))
    n = len(scores)
    proto = DataProto.from_dict(
        tensors={
            "attention_mask": torch.ones(n, 8, dtype=torch.long),
            "responses": torch.zeros(n, 8, dtype=torch.long),
        },
        non_tensors={"uid": np.array(uids, dtype=object)},
    )
    proto.meta_info["global_token_num"] = torch.sum(proto.batch["attention_mask"], dim=-1).tolist()
    return (
        proto,
        torch.tensor(scores, dtype=torch.float32).unsqueeze(-1),
        {"acc": list(scores)},
    )


def make_trainer(train_batch_size, max_num_gen_batches=0):
    t = object.__new__(RayPPOTrainer)
    t.config = types.SimpleNamespace(
        algorithm={"filter_groups": {"enable": True, "metric": "acc", "max_num_gen_batches": max_num_gen_batches}},
        data=types.SimpleNamespace(train_batch_size=train_batch_size),
    )
    # config.algorithm.get(...) -- a plain dict already has .get
    t._dapo_carry = None
    t._dapo_gen_batches = 0
    return t


MIXED = [1.0, 0.0, 1.0, 0.0]
ALL_WRONG = [0.0] * N_ROLLOUTS
ALL_RIGHT = [1.0] * N_ROLLOUTS


def groups_of(result):
    uids = [str(u) for u in result[0].non_tensor_batch["uid"]]
    counts = defaultdict(int)
    for u in uids:
        counts[u] += 1
    return counts


def test_homogeneous_groups_are_dropped():
    t = make_trainer(train_batch_size=2)
    r = t._dapo_accumulate(*make_batch([MIXED, ALL_WRONG, ALL_RIGHT, MIXED]))
    assert r is not None, "two mixed groups is a full batch; should not ask for more"
    counts = groups_of(r)
    assert len(counts) == 2, f"expected 2 qualified groups, got {len(counts)}"
    assert all(c == N_ROLLOUTS for c in counts.values()), "groups must stay whole"


def test_short_batch_requests_more_rollouts():
    t = make_trainer(train_batch_size=4)
    assert t._dapo_accumulate(*make_batch([MIXED, ALL_WRONG])) is None, (
        "1 qualified group < 4 requested: must return None rather than train short"
    )


def test_accumulates_across_generation_batches():
    t = make_trainer(train_batch_size=4)
    assert t._dapo_accumulate(*make_batch([MIXED, MIXED, ALL_WRONG])) is None
    assert t._dapo_accumulate(*make_batch([MIXED, ALL_RIGHT])) is None
    r = t._dapo_accumulate(*make_batch([MIXED, MIXED]))
    assert r is not None, "5 qualified groups accumulated; should assemble"
    counts = groups_of(r)
    assert len(counts) == 4, f"must truncate to train_batch_size=4, got {len(counts)}"
    assert all(c == N_ROLLOUTS for c in counts.values()), "groups must stay whole"
    assert len(r[1]) == 4 * N_ROLLOUTS, "reward tensor must match the assembled batch"
    assert len(r[2]["acc"]) == 4 * N_ROLLOUTS, "extra infos must match the assembled batch"
    assert t._dapo_carry is None and t._dapo_gen_batches == 0, "state must reset after assembly"


def test_empty_batch_does_not_lose_the_carry():
    t = make_trainer(train_batch_size=3)
    assert t._dapo_accumulate(*make_batch([MIXED, MIXED])) is None
    assert t._dapo_accumulate(*make_batch([ALL_WRONG, ALL_RIGHT])) is None, "no new groups"
    r = t._dapo_accumulate(*make_batch([MIXED]))
    assert r is not None
    assert len(groups_of(r)) == 3, "the two carried groups must have survived the empty batch"


def test_generation_budget_is_respected():
    t = make_trainer(train_batch_size=8, max_num_gen_batches=2)
    assert t._dapo_accumulate(*make_batch([MIXED])) is None
    r = t._dapo_accumulate(*make_batch([MIXED]))
    assert r is not None, "budget of 2 exhausted: must return what it has, not loop forever"
    assert r[3]["dapo_filter/gen_budget_exhausted"] == 1.0
    assert len(groups_of(r)) == 2


def test_disabled_filter_is_passthrough():
    t = make_trainer(train_batch_size=2)
    t.config.algorithm["filter_groups"]["enable"] = False
    b, rt, ex = make_batch([MIXED, ALL_WRONG])
    r = t._dapo_accumulate(b, rt, ex)
    assert r is not None and len(r[0]) == len(b), "disabled filter must not drop rows"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
