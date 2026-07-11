# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Execute-and-score for JEPA-GRPO Code-view rollouts.

The Code view asks the policy for a Python program that PRINTS the answer.
Scoring the program SOURCE with the math string-matcher (what happened before
this module existed) cannot measure it — a program is only "correct" by
accident if the answer literal appears in the source. Instead we run the
program in a resource-limited subprocess and hand its stdout's final answer to
the same math verifier the CoT view uses, so both views are scored on the same
scale (±1) and the shared-uid GRPO groups stay comparable.

Sandboxing is best-effort (python -I, rlimits, tmp cwd, no stdin, minimal
env): good enough for self-generated math programs in a research trainer, NOT
a security boundary for untrusted code (no network/filesystem namespace
isolation).
"""

from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
import tempfile

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

DEFAULT_TIMEOUT_S = float(os.environ.get("JEPA_CODE_EXEC_TIMEOUT_S", "5"))


def extract_program(response: str) -> str:
    """Pull the Python program out of a rollout response.

    Prefers fenced ```python blocks (joined, in order) and falls back to the
    raw response — the code system prompt forbids prose outside comments, so
    an unfenced response is usually the bare program.
    """
    blocks = _FENCE_RE.findall(response)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return response.strip()


def extract_stdout_answer(stdout: str) -> str | None:
    """Final answer from program output: \\boxed{...} if printed, else the
    last non-empty line (the prompt instructs 'prints the answer')."""
    m = re.search(r"\\boxed\{([^}]+)\}", stdout)
    if m:
        return m.group(1).strip()
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def _limit_resources():  # pragma: no cover - runs in the child pre-exec
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass


def run_program(code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> tuple[bool, str]:
    """Run `code` in an isolated python subprocess. Returns (exec_ok, stdout).

    stdout is returned even on nonzero exit / timeout: a program can print the
    right answer and then crash, and it should still earn the reward.
    """
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="jepa_code_exec_") as tmp:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                preexec_fn=_limit_resources,
                env=env,
            )
        return proc.returncode == 0, proc.stdout or ""
    except subprocess.TimeoutExpired as e:
        out = e.stdout
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return False, out or ""
    except Exception:
        return False, ""


def score_code_response(
    response: str,
    ground_truth,
    data_source: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Execute a Code-view response and score its printed answer.

    Returns the same dict shape as the math branch of default_compute_score
    (score ±1, acc, pred, ...) plus `exec_ok`, so it drops into the DAPO
    reward manager / metrics pipeline unchanged.
    """
    from verl.experimental.fepo.math_parser import compute_math_reward

    program = extract_program(response)
    exec_ok, stdout = run_program(program, timeout_s=timeout_s)
    answer = extract_stdout_answer(stdout)

    parsed = None
    if answer is not None:
        parsed = compute_math_reward(f"\\boxed{{{answer}}}", ground_truth, dataset_kind=data_source)
    correct = bool(parsed.is_correct) if parsed is not None else False
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": parsed.prediction_normalized if parsed is not None else None,
        "has_parseable_answer": answer is not None,
        "ground_truth_normalized": parsed.ground_truth_normalized if parsed is not None else None,
        "exec_ok": exec_ok,
    }
