# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Rule-verifiable instruction-following rewards (IFEval / FollowBench) + MMLU.

Used by the FEPO paper's cross-domain check (Table 5), whose point is that the
reward stays *rule-verifiable* -- no model judge -- so the only thing changing from
the math experiments is the domain, not the supervision signal.

Scoring is Hard Success Rate: 1.0 only if EVERY constraint on the prompt holds,
0.0 otherwise. That all-or-nothing shape is deliberate -- it is what makes
multi-constraint prompts produce the homogeneous all-wrong groups the method
targets. A partial-credit reward would smooth exactly the failure mode under test.

Constraint specs travel in `ground_truth` as a JSON list of
`{"type": ..., <params>}`. Only the types in CHECKERS below are verifiable here;
`prepare_crossdomain_data.py` filters the datasets to prompts whose constraints are
all supported, so a silently-unscored constraint can never inflate the reward.
"""

from __future__ import annotations

import json
import re

# --- individual constraint checkers -------------------------------------------
# Each takes (response, spec) and returns a bool. Keep them total: a checker must
# never raise on adversarial model output, only return False.


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _c_word_count(r, s):
    n, rel = len(_words(r)), s.get("relation", "at least")
    target = int(s["count"])
    return n <= target if rel == "at most" else n >= target


def _c_sentence_count(r, s):
    n, rel = len(_sentences(r)), s.get("relation", "at least")
    target = int(s["count"])
    return n <= target if rel == "at most" else n >= target


def _c_paragraph_count(r, s):
    return len([p for p in re.split(r"\n\s*\n", r.strip()) if p.strip()]) == int(s["count"])


def _c_keyword_include(r, s):
    return all(k.lower() in r.lower() for k in s["keywords"])


def _c_keyword_forbid(r, s):
    return not any(k.lower() in r.lower() for k in s["keywords"])


def _c_keyword_frequency(r, s):
    n = len(re.findall(re.escape(s["keyword"]), r, flags=re.IGNORECASE))
    rel = s.get("relation", "at least")
    target = int(s["count"])
    return n <= target if rel == "at most" else n >= target


def _c_start_with(r, s):
    return r.strip().lower().startswith(s["prefix"].strip().lower())


def _c_end_with(r, s):
    return r.strip().lower().endswith(s["suffix"].strip().lower())


def _c_quoted(r, _s):
    t = r.strip()
    return len(t) >= 2 and t[0] == '"' and t[-1] == '"'


def _c_json_format(r, _s):
    t = r.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        json.loads(t)
        return True
    except (ValueError, TypeError):
        return False


def _c_lowercase(r, _s):
    return r == r.lower()


def _c_uppercase(r, _s):
    return r == r.upper()


def _c_no_comma(r, _s):
    return "," not in r


def _c_bullet_count(r, s):
    return len(re.findall(r"^\s*[-*•]\s+", r, flags=re.MULTILINE)) == int(s["count"])


def _c_title_present(r, _s):
    return bool(re.search(r"<<[^>]+>>", r))


def _c_placeholder_count(r, s):
    return len(re.findall(r"\[[^\]]+\]", r)) >= int(s["count"])


def _c_section_count(r, s):
    return len(re.findall(re.escape(s.get("marker", "SECTION")), r)) >= int(s["count"])


def _c_postscript(r, s):
    return s.get("marker", "P.S.").lower() in r.lower()


def _c_highlight_count(r, s):
    return len(re.findall(r"\*[^*\n]+\*", r)) >= int(s["count"])


CHECKERS = {
    "word_count": _c_word_count,
    "sentence_count": _c_sentence_count,
    "paragraph_count": _c_paragraph_count,
    "keyword_include": _c_keyword_include,
    "keyword_forbid": _c_keyword_forbid,
    "keyword_frequency": _c_keyword_frequency,
    "start_with": _c_start_with,
    "end_with": _c_end_with,
    "quoted": _c_quoted,
    "json_format": _c_json_format,
    "lowercase": _c_lowercase,
    "uppercase": _c_uppercase,
    "no_comma": _c_no_comma,
    "bullet_count": _c_bullet_count,
    "title_present": _c_title_present,
    "placeholder_count": _c_placeholder_count,
    "section_count": _c_section_count,
    "postscript": _c_postscript,
    "highlight_count": _c_highlight_count,
}

SUPPORTED_CONSTRAINTS = frozenset(CHECKERS)


def _strip_reasoning(solution_str: str) -> str:
    """Drop a chat prefix and any <think> block so constraints apply to the answer."""
    text = solution_str
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>", "", text)
    return text.strip()


def compute_score(solution_str: str, ground_truth, extra_info=None) -> dict:
    """Hard Success Rate over every constraint attached to the prompt."""
    response = _strip_reasoning(solution_str)

    specs = ground_truth
    if isinstance(specs, str):
        try:
            specs = json.loads(specs)
        except ValueError:
            return {"score": 0.0, "acc": 0.0, "pred": "<unparsable-constraints>"}
    if isinstance(specs, dict):
        specs = [specs]
    if not specs:
        return {"score": 0.0, "acc": 0.0, "pred": "<no-constraints>"}

    results = []
    for spec in specs:
        checker = CHECKERS.get(spec.get("type"))
        if checker is None:
            # Unreachable for prepared data (the prep script filters on
            # SUPPORTED_CONSTRAINTS). Fail closed rather than award free credit.
            results.append(False)
            continue
        try:
            results.append(bool(checker(response, spec)))
        except Exception:
            results.append(False)

    hard = 1.0 if all(results) else 0.0
    return {
        "score": hard,
        "acc": hard,
        # Soft rate is reported for analysis only; it never drives the reward.
        "soft_rate": sum(results) / len(results),
        "pred": f"{sum(results)}/{len(results)} constraints",
    }


_MMLU_RE = re.compile(r"\b([ABCD])\b")


def compute_score_mmlu(solution_str: str, ground_truth, extra_info=None) -> dict:
    """MMLU multiple choice, used purely as a regression check in Table 5.

    Never let this drive checkpoint selection -- the cross-domain arms train on
    instruction following, and MMLU exists only to show that capability did not
    regress while they did.
    """
    response = _strip_reasoning(solution_str)
    tail = response[-200:] if len(response) > 200 else response
    matches = _MMLU_RE.findall(tail.upper())
    pred = matches[-1] if matches else ""
    gt = str(ground_truth).strip().upper()
    if gt.isdigit():
        gt = "ABCD"[int(gt)] if int(gt) < 4 else gt
    correct = 1.0 if pred and pred == gt else 0.0
    return {"score": correct, "acc": correct, "pred": pred or "<none>"}
