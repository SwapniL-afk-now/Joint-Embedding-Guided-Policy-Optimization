#!/usr/bin/env python3
"""Recovery@delta and the bucket gains must be paired on ONE prompt set.

Both arms start from the same base model, but step 0 is sampled (n=8, temp 0.6),
so each arm's own step-0 pass disagrees about which prompts sit at p=0. If each arm
picks its own set, the Increment column subtracts rates measured on different
prompts. This asserts the base arm's reference drives both columns.

    python experiments/fepo_paper/test_paired_recovery.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import _bucket_gains, _recovery_at  # noqa: E402

# Base says A,B are p=0. FEPO's own step-0 draw says B,C are -- the disagreement a
# sampled reference produces in practice.
BASE = {0: {"A": 0.0, "B": 0.0, "C": 0.5}, 50: {"A": 0.0, "B": 0.0, "C": 0.5}}
FEPO = {0: {"A": 0.5, "B": 0.0, "C": 0.0}, 50: {"A": 0.5, "B": 1.0, "C": 0.0}}

failed = {p for p, r in BASE[0].items() if r == 0.0}
assert failed == {"A", "B"}

# Unpaired: FEPO uses its OWN p=0 set {B,C} -- B recovers, C does not -> 0.5.
assert _recovery_at(FEPO, 0, 50) == 0.5, "unpaired rate over FEPO's own {B,C}"
# Paired on the base set {A,B}: both are >0 at step 50 -> 1.0. Same run, same dumps,
# double the reported recovery purely from which prompts are counted.
assert _recovery_at(FEPO, 0, 50, failed) == 1.0, "paired rate over the base's {A,B}"

FEPO2 = {0: {"A": 0.0, "B": 0.0, "C": 0.0}, 50: {"A": 0.0, "B": 0.0, "C": 1.0}}
assert _recovery_at(FEPO2, 0, 50) == 1 / 3, "unpaired: C recovers out of {A,B,C}"
assert _recovery_at(FEPO2, 0, 50, failed) == 0.0, "paired: neither A nor B recovers"

# Bucket membership must come from the shared reference too.
gains_own = _bucket_gains(FEPO2)
gains_paired = _bucket_gains(FEPO2, ref_scores=BASE[0])
assert gains_own["p=0"] == 1 / 3, gains_own
assert gains_paired["p=0"] == 0.0, gains_paired
# C sits in 0.25<p<=0.50 by the base reference, and gained 1.0 - 0.0 in FEPO2.
assert gains_paired["0.25<p<=0.50"] == 1.0, gains_paired

print("ok: Recovery@delta and bucket gains are paired on the base arm's reference")
