"""Compare JEPA-GRPO arms from the file-logger jsonl dumps.

Usage: python3 ab_report.py [metrics_*.jsonl ...]   (default: all in cwd)

Reads VERL_FILE_LOGGER_PATH output ({"step":N,"data":{...}} per line).
Reports the collapse/reachability read-outs plus accuracy, and compares arms on
MATCHED step counts -- arms of different length are not comparable by raw mean.
"""
import glob
import json
import statistics as st
import sys

KEYS = ["jepa/effective_rank", "jepa/target_effective_rank", "jepa/retrieval_top1",
        "jepa/retrieval_chance", "jepa/cos_all_pairs", "jepa/cos_cot", "jepa/tcr_loss",
        "jepa/grad_cos_with_policy", "jepa/grad_ratio", "actor/grad_norm", "train/accuracy"]


def load(path):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = dict(r.get("data", r))
        d["step"] = r.get("step", d.get("step"))
        rows.append(d)
    return sorted(rows, key=lambda d: d["step"])


def main(paths):
    arms = {}
    for p in paths:
        rows = load(p)
        if rows:
            arms[p.split("/")[-1].replace("metrics_", "").replace(".jsonl", "")] = rows

    for name, rows in arms.items():
        print(f"\n=== {name}  ({len(rows)} steps)")
        present = [k for k in KEYS if any(k in r for r in rows)]
        print("step  " + "  ".join(f"{k.split('/')[-1][:13]:>13s}" for k in present))
        for d in rows:
            print(f"{d['step']:<5d} " + "  ".join(
                f"{d[k]:13.4f}" if isinstance(d.get(k), (int, float)) else f"{'-':>13s}"
                for k in present))

    acc = {n: [r["train/accuracy"] for r in rows if "train/accuracy" in r]
           for n, rows in arms.items()}
    acc = {n: a for n, a in acc.items() if a}
    if len(acc) > 1:
        k = min(len(a) for a in acc.values())
        print(f"\n=== matched on first {k} steps (means over different lengths are not comparable)")
        for n, a in sorted(acc.items(), key=lambda kv: -st.mean(kv[1][:k])):
            print(f"  {n:28s} {st.mean(a[:k]):.4f}   (sd {st.pstdev(a[:k]):.4f})")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("metrics_*.jsonl")))
