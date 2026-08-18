# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build the (X, X') transfer-eval pairs from MATH-P-Simple.

MATH-P-Simple ships only the perturbed problems X' -- the originals X are not in
the repo, and ``problem_id`` is a random id, not a MATH index. So X is recovered
by fuzzy-matching each X' back to the MATH corpus within its own subject. The
perturbations are non-essential edits (numbers, names), so true matches sit near
1.0 and the threshold is not delicate: 254/279 match at >=0.9 and 270 at >=0.8.

    .venv/bin/python examples/abstract_rl/build_math_perturb_pairs.py \\
        --out /workspace/jepa-grpo-cache/eval_data/math_perturb_pairs.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

PERTURB_URL = "https://raw.githubusercontent.com/Kaffaljidhmah2/MATH-Perturb/main/math_perturb/math_perturb_simple.jsonl"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def best_match(query: str, candidates: list[tuple[str, str, str]]) -> tuple[float, tuple | None]:
    """Highest SequenceMatcher ratio, using the cheap bounds to skip hopeless candidates."""
    sm = SequenceMatcher()
    sm.set_seq2(query)
    best_score, best = 0.0, None
    for cand in candidates:
        sm.set_seq1(cand[0])
        if sm.real_quick_ratio() <= best_score or sm.quick_ratio() <= best_score:
            continue
        score = sm.ratio()
        if score > best_score:
            best_score, best = score, cand
    return best_score, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--math-repo", default="nlile/hendrycks-MATH-benchmark")
    ap.add_argument("--min-similarity", type=float, default=0.8)
    args = ap.parse_args()

    from datasets import load_dataset

    with urllib.request.urlopen(PERTURB_URL) as fh:
        perturbed = [json.loads(line) for line in fh.read().decode().splitlines() if line.strip()]

    math = load_dataset(args.math_repo)
    by_subject = defaultdict(list)
    for row in list(math["train"]) + list(math["test"]):
        by_subject[row["subject"]].append((norm(row["problem"]), row["answer"], row["solution"]))

    records, dropped = [], 0
    for row in perturbed:
        score, match = best_match(norm(row["problem"]), by_subject.get(row["type"], []))
        if match is None or score < args.min_similarity:
            dropped += 1
            continue
        original, original_answer, original_solution = match
        records.append(
            {
                "problem_id": str(row["problem_id"]),
                "original": original,
                "original_answer": str(original_answer),
                "original_solution": original_solution,
                "perturbed": row["problem"],
                # answers are mixed ints and latex; keep them all as str or arrow rejects the column
                "perturbed_answer": str(row["answer"]),
                "type": row["type"],
                "similarity": score,
            }
        )

    out = pd.DataFrame.from_records(records)
    out.to_parquet(args.out, index=False)
    print(f"wrote {len(out)}/{len(perturbed)} pairs to {args.out} (dropped {dropped} below {args.min_similarity})")
    print(f"similarity: mean {out.similarity.mean():.3f}, min {out.similarity.min():.3f}")


if __name__ == "__main__":
    main()
