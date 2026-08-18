#!/usr/bin/env python3
"""Turn a run's validation dumps into Table 2 (mechanism) and Table 4 (systems).

Neither table needs a dedicated training run -- both are post-hoc reads of what the
Table 1 runs already emit, which is what cuts the program from ~42 runs to 22. This
script is the harvest step.

    # Table 2: Recovery@delta + initial-p difficulty buckets
    python experiments/fepo_paper/analyze.py recovery val_dumps/fepo-m06_dapo_fepo-20260807

    # sanity gate -- run this on the canary before launching the other 21 arms
    python experiments/fepo_paper/analyze.py check val_dumps/fepo-m06_dapo_fepo-20260807

    # Table 2 / Table 7: the paired base-vs-FEPO columns and the increment
    python experiments/fepo_paper/analyze.py compare val_dumps/<base> val_dumps/<fepo>

    # Figure 2: Recovery@delta trajectory
    python experiments/fepo_paper/analyze.py trajectory val_dumps/<exp>

    # Table 4: systems accounting
    python experiments/fepo_paper/analyze.py systems <entity/project/run_id> --target 0.60

Recovery@delta is defined against the step-0 reference checkpoint (which is why
_common.sh sets VAL_BEFORE_TRAIN=true): the fraction of probe prompts with success
rate exactly 0 at step 0 that have success rate > 0 by step delta.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BUCKETS = [
    ("p=0", lambda p: p == 0.0),
    ("0<p<=0.25", lambda p: 0.0 < p <= 0.25),
    ("0.25<p<=0.50", lambda p: 0.25 < p <= 0.50),
    ("0.50<p<=0.75", lambda p: 0.50 < p <= 0.75),
    ("0.75<p<=1", lambda p: 0.75 < p <= 1.0),
]


def load_probe_success(dump_dir: Path) -> dict[int, dict[str, float]]:
    """step -> {prompt: success_rate}, restricted to the probe set.

    _dump_generations writes one JSONL line per *response*, so the per-prompt success
    rate is the mean score over that prompt's n=8 rows. Prompts are grouped by their
    input text, which is unique for math datasets.
    """
    by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    files = sorted(dump_dir.glob("*.jsonl"), key=lambda f: int(f.stem))
    if not files:
        sys.exit(f"no *.jsonl in {dump_dir} -- was trainer.validation_data_dir set?")

    for f in files:
        step = int(f.stem)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # The probe rows are the only ones we can group by prompt across steps;
            # the five math benchmarks are scored too but are not the mechanism set.
            if row.get("data_source", "probe") != "probe":
                continue
            # `acc`, NOT `score`: this repo's math scorer returns +1.0/-1.0, so
            # averaging score puts an all-wrong prompt at p=-1 (matching no bucket,
            # silently dropped) and a 4-of-8 prompt at exactly p=0, where Recovery@delta
            # would count it as "never solved". acc is boolean, so p is the success
            # fraction in [0,1] that the buckets and Recovery@delta are defined on.
            by_step[step][row["input"]].append(1.0 if row["acc"] else 0.0)

    return {s: {p: sum(v) / len(v) for p, v in d.items()} for s, d in by_step.items()}


def cmd_check(dump_dir: Path) -> int:
    """Phase-0 gate. If this fails, sampled validation did not take effect."""
    hist = load_probe_success(dump_dir)
    if not hist:
        sys.exit("no probe rows found -- is probe256.parquet in data.val_files?")
    step = min(hist)
    ps = list(hist[step].values())
    counts = {name: sum(1 for p in ps if fn(p)) for name, fn in BUCKETS}

    print(f"steps dumped: {sorted(hist)}")
    print(f"probe prompts at step {step}: {len(ps)}")
    for name, n in counts.items():
        print(f"  {name:<14} {n:>4}  {'#' * (50 * n // max(len(ps), 1))}")

    degenerate = counts["p=0"] + counts["0.75<p<=1"] == len(ps)
    if degenerate:
        print()
        print("GATE FAILED: every prompt is at p=0 or p=1, so all five difficulty")
        print("  buckets collapse to two and Table 2 cannot be built. Validation is")
        print("  still greedy -- check VAL_ROLLOUT_N=8 / VAL_DO_SAMPLE=True actually")
        print("  reached the config. Do NOT launch the remaining arms.")
        return 1
    print("\nGATE PASSED: buckets are populated.")
    return 0


def cmd_recovery(dump_dir: Path, deltas: list[int]) -> int:
    hist = load_probe_success(dump_dir)
    ref = min(hist)
    base = hist[ref]
    failed = {p for p, r in base.items() if r == 0.0}

    print(f"reference step {ref}: {len(failed)}/{len(base)} probe prompts at p=0\n")
    if not failed:
        print("no p=0 prompts -- Recovery@delta is undefined for this run.")
        return 1

    print("Recovery@delta (of prompts at p=0 at the reference step):")
    for d in deltas:
        later = [s for s in hist if s >= ref + d]
        if not later:
            print(f"  @{d:<4} n/a (no dump at step >= {ref + d})")
            continue
        s = min(later)
        rec = sum(1 for p in failed if hist[s].get(p, 0.0) > 0.0) / len(failed)
        print(f"  @{d:<4} {rec:.3f}   (measured at step {s})")

    last = max(hist)
    print(f"\nAccuracy gain by initial success rate (step {ref} -> {last}):")
    for name, fn in BUCKETS:
        members = [p for p, r in base.items() if fn(r)]
        if not members:
            print(f"  {name:<14} n/a (empty bucket)")
            continue
        gain = sum(hist[last].get(p, 0.0) - base[p] for p in members) / len(members)
        print(f"  {name:<14} n={len(members):<4} delta_acc {gain:+.4f}")

    print("\nThe mechanism predicts the largest gain at p=0, shrinking as p rises.")
    print("A uniform gain across buckets weakens the failure-escape interpretation.")
    return 0


def _recovery_at(hist, ref, delta, failed=None):
    """Recovery@delta measured from the reference step, or None if not yet reached.

    `failed` overrides the prompt set. cmd_compare MUST pass one: both arms start
    from the same base model, but step 0 is sampled (n=8, temp 0.6), so each arm's
    own step-0 pass disagrees about which prompts sit at p=0. Letting each arm pick
    its own set makes the reported Increment a difference of two rates measured on
    different prompts -- not the paired quantity the table claims.
    """
    base = hist[ref]
    if failed is None:
        failed = {p for p, r in base.items() if r == 0.0}
    later = [s for s in hist if s >= ref + delta]
    if not failed or not later:
        return None
    s = min(later)
    return sum(1 for p in failed if hist[s].get(p, 0.0) > 0.0) / len(failed)


def _bucket_gains(hist, ref_scores=None):
    """Per-bucket mean accuracy gain from the reference step to the last step.

    `ref_scores` fixes bucket membership (and the baseline each gain is measured
    from) to one arm's reference, for the same pairing reason as _recovery_at.
    """
    ref, last = min(hist), max(hist)
    base = ref_scores if ref_scores is not None else hist[ref]
    out = {}
    for name, fn in BUCKETS:
        members = [p for p, r in base.items() if fn(r) and p in hist[ref]]
        out[name] = (
            sum(hist[last].get(p, 0.0) - hist[ref][p] for p in members) / len(members) if members else None
        )
    return out


def cmd_compare(base_dir: Path, fepo_dir: Path, deltas: list[int]) -> int:
    """Table 2 (mechanism) and Table 7 (easy-bucket delta): base vs +FEPO, paired.

    Pairing is by prompt: both runs evaluate the identical probe set, so the
    comparison is over the same prompts rather than two independent samples.
    """
    hb, hf = load_probe_success(base_dir), load_probe_success(fepo_dir)
    rb, rf = min(hb), min(hf)

    # One reference set for both arms, from the BASE arm's step-0 pass, restricted to
    # prompts both arms actually scored. Without this the two columns are computed on
    # different prompt subsets and the Increment is not a paired difference.
    shared = hb[rb].keys() & hf[rf].keys()
    ref_scores = {p: hb[rb][p] for p in shared}
    failed = {p for p, r in ref_scores.items() if r == 0.0}

    print(f"base: {base_dir.name}   +FEPO: {fepo_dir.name}")
    print(
        f"paired on {len(shared)} prompts scored by both arms; "
        f"{len(failed)} at p=0 in the base arm's step-{rb} reference "
        f"(this set is used for BOTH columns)\n"
    )
    print(f"{'':<10} {'Base':>10} {'+FEPO':>10} {'Increment':>12}")
    for d in deltas:
        a, b = _recovery_at(hb, rb, d, failed), _recovery_at(hf, rf, d, failed)
        if a is None or b is None:
            print(f"Rec@{d:<6} {'n/a':>10} {'n/a':>10} {'n/a':>12}")
        else:
            print(f"Rec@{d:<6} {a:>10.3f} {b:>10.3f} {b - a:>+12.3f}")

    print("\nAccuracy gain by initial success rate (Table 2 lower panel):")
    print(f"{'bucket':<14} {'Base dacc':>11} {'+FEPO dacc':>12} {'Increment':>12}")
    gb, gf = _bucket_gains(hb, ref_scores), _bucket_gains(hf, ref_scores)
    for name, _ in BUCKETS:
        a, b = gb[name], gf[name]
        if a is None or b is None:
            print(f"{name:<14} {'n/a':>11} {'n/a':>12} {'n/a':>12}")
        else:
            print(f"{name:<14} {a:>+11.4f} {b:>+12.4f} {b - a:>+12.4f}")

    easy = BUCKETS[-1][0]
    if gb[easy] is not None and gf[easy] is not None:
        print(f"\nTable 7 easy-bucket delta ({easy}): {gf[easy] - gb[easy]:+.4f}")
        print("Negative means FEPO costs accuracy on easy prompts relative to the base --")
        print("that is the retention side of the recovery/retention trade-off.")
    return 0


def cmd_trajectory(dump_dir: Path) -> int:
    """Figure 2: Recovery@delta as a function of delta, not just two checkpoints."""
    hist = load_probe_success(dump_dir)
    ref = min(hist)
    failed = {p for p, r in hist[ref].items() if r == 0.0}
    if not failed:
        print("no p=0 prompts at the reference step; trajectory undefined")
        return 1
    print(f"reference step {ref}, {len(failed)} prompts at p=0")
    print(f"{'delta':>8} {'step':>8} {'recovery':>10}")
    for s in sorted(hist):
        rec = sum(1 for p in failed if hist[s].get(p, 0.0) > 0.0) / len(failed)
        print(f"{s - ref:>8} {s:>8} {rec:>10.4f}")
    return 0


def cmd_systems(run_path: str, target: float) -> int:
    import wandb

    r = wandb.Api().run(run_path)
    keys = [
        "_step",
        "perf/time_per_step",
        "perf/peak_memory_gb",
        "perf/cumulative_tokens",
        "perf/cumulative_gpu_hours",
        "val/math500/pass_at_1",
    ]
    h = r.history(keys=keys, pandas=True)

    hit = h[h["val/math500/pass_at_1"] >= target]
    print(f"run: {r.name} ({r.state})")
    print(f"sec/update      {h['perf/time_per_step'].mean():.1f}")
    for k, label in [("perf/peak_memory_gb", "peak memory GB")]:
        print(f"{label:<15} {h[k].max():.1f}" if k in h else f"{label:<15} MISSING (C2 not landed)")

    if hit.empty:
        print(f"\ntarget MATH-500 pass@1 >= {target} never reached -- report as 'not reached',")
        print("do not lower the target after the fact.")
    else:
        row = hit.iloc[0]
        print(f"\nreached pass@1 {target} at step {int(row['_step'])}")
        print(f"  rollout tokens to target  {row.get('perf/cumulative_tokens', float('nan')):.3e}")
        print(f"  GPU-h to target           {row.get('perf/cumulative_gpu_hours', float('nan')):.2f}")
    print(f"\nfinal MATH-500 pass@1 {h['val/math500/pass_at_1'].iloc[-1]:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="Phase-0 gate: are the difficulty buckets populated?")
    c.add_argument("dump_dir", type=Path)

    r = sub.add_parser("recovery", help="Table 2: Recovery@delta + difficulty buckets")
    r.add_argument("dump_dir", type=Path)
    r.add_argument("--deltas", type=int, nargs="+", default=[50, 100])

    c2 = sub.add_parser("compare", help="Tables 2 and 7: base vs +FEPO, paired by prompt")
    c2.add_argument("base_dir", type=Path)
    c2.add_argument("fepo_dir", type=Path)
    c2.add_argument("--deltas", type=int, nargs="+", default=[50, 100])

    tr = sub.add_parser("trajectory", help="Figure 2: Recovery@delta over training")
    tr.add_argument("dump_dir", type=Path)

    s = sub.add_parser("systems", help="Table 4: systems accounting from wandb")
    s.add_argument("run_path", help="entity/project/run_id")
    s.add_argument("--target", type=float, default=0.60, help="pre-declared MATH-500 pass@1 target")

    a = ap.parse_args()
    if a.cmd == "check":
        return cmd_check(a.dump_dir)
    if a.cmd == "recovery":
        return cmd_recovery(a.dump_dir, a.deltas)
    if a.cmd == "compare":
        return cmd_compare(a.base_dir, a.fepo_dir, a.deltas)
    if a.cmd == "trajectory":
        return cmd_trajectory(a.dump_dir)
    return cmd_systems(a.run_path, a.target)


if __name__ == "__main__":
    raise SystemExit(main())
