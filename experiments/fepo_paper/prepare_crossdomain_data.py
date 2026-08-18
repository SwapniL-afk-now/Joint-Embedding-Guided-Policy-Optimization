#!/usr/bin/env python3
"""Build the IFEval / FollowBench / MMLU parquets for the Table 5 cross-domain arms.

Needs network (Hugging Face). Run once before x01/x02:

    python experiments/fepo_paper/prepare_crossdomain_data.py

Writes:
    /workspace/jepa-grpo-cache/data/ifeval/train.parquet   (training prompts)
    /workspace/jepa-grpo-cache/eval_data/ifeval.parquet
    /workspace/jepa-grpo-cache/eval_data/followbench.parquet
    /workspace/jepa-grpo-cache/eval_data/mmlu.parquet

Constraint filtering: IFEval ships ~25 instruction types; this repo's verifier
(verl/utils/reward_score/instruction_following.py) implements the subset that can be
checked exactly with no language-ID model or external dependency. Prompts carrying an
unsupported constraint are DROPPED rather than scored leniently -- a constraint that
silently passes would inflate Hard Success Rate and make the arm look better than it
is. Report the retained count in the paper as "IFEval (verifiable subset, n=...)".
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl.utils.reward_score.instruction_following import SUPPORTED_CONSTRAINTS  # noqa: E402

DATA = Path("/workspace/jepa-grpo-cache/data")
EVAL = Path("/workspace/jepa-grpo-cache/eval_data")

SYSTEM = "You are a helpful assistant. Follow every instruction in the request exactly."

# IFEval instruction_id -> this repo's constraint spec. Only these are kept.
IFEVAL_MAP = {
    "length_constraints:number_words": lambda kw: {
        "type": "word_count",
        "count": kw.get("num_words"),
        "relation": kw.get("relation", "at least"),
    },
    "length_constraints:number_sentences": lambda kw: {
        "type": "sentence_count",
        "count": kw.get("num_sentences"),
        "relation": kw.get("relation", "at least"),
    },
    "length_constraints:number_paragraphs": lambda kw: {"type": "paragraph_count", "count": kw.get("num_paragraphs")},
    "keywords:existence": lambda kw: {"type": "keyword_include", "keywords": kw.get("keywords") or []},
    "keywords:forbidden_words": lambda kw: {"type": "keyword_forbid", "keywords": kw.get("forbidden_words") or []},
    "keywords:frequency": lambda kw: {
        "type": "keyword_frequency",
        "keyword": kw.get("keyword"),
        "count": kw.get("frequency"),
        "relation": kw.get("relation", "at least"),
    },
    "startend:end_checker": lambda kw: {"type": "end_with", "suffix": kw.get("end_phrase")},
    "startend:quotation": lambda kw: {"type": "quoted"},
    "detectable_format:json_format": lambda kw: {"type": "json_format"},
    "change_case:english_lowercase": lambda kw: {"type": "lowercase"},
    "change_case:english_capital": lambda kw: {"type": "uppercase"},
    "punctuation:no_comma": lambda kw: {"type": "no_comma"},
    "detectable_format:number_bullet_lists": lambda kw: {"type": "bullet_count", "count": kw.get("num_bullets")},
    "detectable_format:title": lambda kw: {"type": "title_present"},
    "detectable_content:number_placeholders": lambda kw: {
        "type": "placeholder_count",
        "count": kw.get("num_placeholders"),
    },
    "detectable_format:multiple_sections": lambda kw: {
        "type": "section_count",
        "marker": kw.get("section_spliter", "SECTION"),
        "count": kw.get("num_sections"),
    },
    "detectable_content:postscript": lambda kw: {"type": "postscript", "marker": kw.get("postscript_marker", "P.S.")},
    "detectable_format:number_highlighted_sections": lambda kw: {
        "type": "highlight_count",
        "count": kw.get("num_highlights"),
    },
}


def _row(prompt: str, specs, source: str, idx: int, split: str):
    return {
        "data_source": source,
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "ability": "instruction_following",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(specs)},
        "extra_info": {"split": split, "index": idx, "num_constraints": len(specs) if isinstance(specs, list) else 1},
    }


def _convert_ifeval(rec):
    """Return specs list, or None if any constraint is unsupported."""
    ids = rec.get("instruction_id_list") or []
    kwargs_list = rec.get("kwargs") or [{}] * len(ids)
    specs = []
    for iid, kw in zip(ids, list(kwargs_list) + [{}] * len(ids), strict=False):
        fn = IFEVAL_MAP.get(iid)
        if fn is None:
            return None
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        spec = fn(kw)
        if spec.get("type") not in SUPPORTED_CONSTRAINTS:
            return None
        # A spec missing its required parameter cannot be checked exactly.
        if any(v is None for v in spec.values()):
            return None
        specs.append(spec)
    return specs or None


def build_ifeval():
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    rows, dropped = [], 0
    for i, rec in enumerate(ds):
        specs = _convert_ifeval(rec)
        if specs is None:
            dropped += 1
            continue
        rows.append(_row(rec["prompt"], specs, "ifeval", i, "train"))

    # IFEval has no official train/test split; hold out the last 20% for evaluation
    # so the cross-domain arms are not evaluated on prompts they trained on.
    cut = int(len(rows) * 0.8)
    train, test = rows[:cut], rows[cut:]
    for r in test:
        r["extra_info"]["split"] = "test"

    (DATA / "ifeval").mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(DATA / "ifeval" / "train.parquet", index=False)
    pd.DataFrame(test).to_parquet(EVAL / "ifeval.parquet", index=False)
    print(f"ifeval: {len(train)} train / {len(test)} eval  (dropped {dropped} unsupported-constraint prompts)")


def build_followbench():
    """FollowBench, restricted to its example-level verifiable constraints.

    FollowBench's own harness scores several categories with an LLM judge. Those are
    excluded on purpose: the entire point of this table is a rule-verifiable reward,
    so a judged category would reintroduce exactly the supervision the paper avoids.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("YuxinJiang/FollowBench", split="train")
    except Exception as exc:  # dataset id/config drift is common here
        print(f"followbench: SKIPPED ({exc}).", file=sys.stderr)
        print("  Table 5 can be reported on IFEval alone; note the omission.", file=sys.stderr)
        return

    rows = []
    for i, rec in enumerate(ds):
        prompt = rec.get("prompt") or rec.get("instruction")
        if not prompt:
            continue
        specs = []
        # Only the format category carries machine-checkable constraints in a stable
        # schema; the rest need the judge and are dropped.
        if str(rec.get("category", "")).lower() in {"format", "example"}:
            m = re.search(r"at least (\d+) words", prompt, flags=re.IGNORECASE)
            if m:
                specs.append({"type": "word_count", "count": int(m.group(1)), "relation": "at least"})
            m = re.search(r"exactly (\d+) bullet", prompt, flags=re.IGNORECASE)
            if m:
                specs.append({"type": "bullet_count", "count": int(m.group(1))})
        if specs:
            rows.append(_row(prompt, specs, "followbench", i, "test"))

    if not rows:
        print("followbench: no rule-verifiable prompts extracted; SKIPPED", file=sys.stderr)
        return
    pd.DataFrame(rows).to_parquet(EVAL / "followbench.parquet", index=False)
    print(f"followbench: {len(rows)} eval rows")


def build_mmlu(n=1000):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=3407).select(range(n))
    rows = []
    for i, rec in enumerate(ds):
        choices = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", rec["choices"], strict=False))
        q = f"{rec['question']}\n{choices}\n\nAnswer with the letter of the correct choice only."
        rows.append(
            {
                "data_source": "mmlu",
                "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                "ability": "knowledge",
                "reward_model": {"style": "rule", "ground_truth": "ABCD"[int(rec["answer"])]},
                "extra_info": {"split": "test", "index": i, "subject": rec.get("subject", "")},
            }
        )
    pd.DataFrame(rows).to_parquet(EVAL / "mmlu.parquet", index=False)
    print(f"mmlu: {len(rows)} eval rows (regression check only -- never in BEST_CKPT_SOURCES)")


if __name__ == "__main__":
    build_ifeval()
    build_followbench()
    build_mmlu()
