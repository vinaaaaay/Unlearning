"""Score the temperature sweep (gen_<tag>_T<t>.jsonl) next to the greedy run.

Each sample is scored with the same rules as score.py (lenient answer match).
Per question, two values:
  * per-sample accuracy: fraction of the N samples that are correct
  * any-of-N (P_leak):   1 if at least one of the N samples is correct
For T=0 the greedy output is used for both (N identical samples).

Primary comparison, fixed before the results were seen:
    any-of-N  Acc(unlearned) - Acc(clean), per temperature and hop.
Secondary: per-sample accuracy, and both metrics against the base model.
CIs: paired bootstrap over authors (see score.py).

Usage: python score_temperature.py
"""

import json
import random
from collections import defaultdict

from score import ORDER, PROBES, RESULTS, UNLEARNED, boot_ci, score_row

TEMPS = [0.0, 0.3, 0.7, 1.0, 1.3]
HOPS = [1, 2, 3]
METRICS = {"any": "any-of-N (P_leak)", "per_sample": "per-sample accuracy"}


def load_temp(m, t, probes):
    """Return {probe_id: {"hop", "author", "per_sample", "any", "trunc"}} or None."""
    path = RESULTS / (f"gen_{m}.jsonl" if t == 0 else f"gen_{m}_T{t}.jsonl")
    if not path.exists():
        return None
    samples = defaultdict(list)
    for r in map(json.loads, open(path)):
        samples[r["id"]].append(r)
    if len(samples) != len(probes):
        print(f"skipping {m} T={t}: {len(samples)}/{len(probes)} questions")
        return None
    n = {len(v) for v in samples.values()}
    if len(n) != 1:
        print(f"skipping {m} T={t}: uneven sample counts {sorted(n)}")
        return None
    out = {}
    for pid, rows in samples.items():
        p = probes[pid]
        ok = [score_row(r["generation"], p)["lenient"] for r in rows]
        out[pid] = {
            "hop": p["hop"], "author": p["author_idx"],
            "per_sample": sum(ok) / len(ok), "any": float(any(ok)),
            "trunc": sum(r["truncated"] for r in rows) / len(rows),
            "n_samples": len(rows),
        }
    return out


def per_author(rows, hop, key):
    acc = defaultdict(list)
    for r in rows.values():
        if r["hop"] == hop:
            acc[r["author"]].append(r[key])
    return {a: sum(v) / len(v) for a, v in acc.items()}


def paired(rows_m, rows_ref, hop, key, rng):
    u, c = per_author(rows_m, hop, key), per_author(rows_ref, hop, key)
    diff = {a: u[a] - c[a] for a in u}
    lo, hi = boot_ci(diff, rng)
    return {"diff": sum(diff.values()) / len(diff), "ci95": [lo, hi],
            "verdict": ">" if lo > 0 else "<" if hi < 0 else "~"}


def main():
    rng = random.Random(0)
    probes = {p["id"]: p for p in map(json.loads, open(PROBES))}
    data = {}
    for m in ORDER:
        for t in TEMPS:
            rows = load_temp(m, t, probes)
            if rows is not None:
                data[(m, t)] = rows
    models = [m for m in ORDER if any((m, t) in data for t in TEMPS)]
    out = {"accuracy": {}, "vs_clean": {}, "vs_base": {}, "truncation": {}}

    for key, label in METRICS.items():
        print(f"\n=== {label}: accuracy (95% author-bootstrap CI)")
        for h in HOPS:
            print(f"--- hop {h}")
            print(f"{'model':<10}" + "".join(f"{f'T={t}':>20}" for t in TEMPS))
            for m in models:
                line = f"{m:<10}"
                for t in TEMPS:
                    if (m, t) not in data:
                        line += f"{'-':>20}"
                        continue
                    pa = per_author(data[(m, t)], h, key)
                    mean = sum(pa.values()) / len(pa)
                    lo, hi = boot_ci(pa, rng)
                    out["accuracy"].setdefault(key, {}).setdefault(m, {}).setdefault(str(t), {})[h] = {
                        "mean": mean, "ci95": [lo, hi]}
                    line += f"{mean:>7.0%} [{lo:4.0%},{hi:4.0%}]"
                print(line)

    for ref, store in (("clean", "vs_clean"), ("base", "vs_base")):
        others = UNLEARNED if ref == "clean" else [m for m in models if m != "base"]
        for key, label in METRICS.items():
            tag = "PRIMARY" if (ref == "clean" and key == "any") else "secondary"
            print(f"\n=== [{tag}] {label}: Acc(model) - Acc({ref})   "
                  f"(> / < : 95% CI excludes 0, ~ : includes 0)")
            for h in HOPS:
                print(f"--- hop {h}")
                for m in [m for m in others if m in models]:
                    line = f"{m:<10}"
                    for t in TEMPS:
                        if (m, t) not in data or (ref, t) not in data:
                            line += f"{'-':>26}"
                            continue
                        r = paired(data[(m, t)], data[(ref, t)], h, key, rng)
                        out[store].setdefault(key, {}).setdefault(m, {}).setdefault(str(t), {})[h] = r
                        lo, hi = r["ci95"]
                        line += f"  T={t}: {r['diff']:+4.0%} {r['verdict']}".ljust(26)
                    print(line)

    print("\n=== Truncated samples (stopped at 384 tokens)")
    print(f"{'model':<10}" + "".join(f"{f'T={t}':>9}" for t in TEMPS))
    for m in models:
        line = f"{m:<10}"
        for t in TEMPS:
            if (m, t) in data:
                tr = sum(r["trunc"] for r in data[(m, t)].values()) / len(data[(m, t)])
                out["truncation"].setdefault(m, {})[str(t)] = tr
                line += f"{tr:>9.0%}"
            else:
                line += f"{'-':>9}"
        print(line)

    (RESULTS / "summary_temperature.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "scored_temperature.jsonl", "w") as f:
        for (m, t), rows in data.items():
            for pid, r in rows.items():
                f.write(json.dumps({"model": m, "temperature": t, "id": pid, **r}) + "\n")
    print(f"\nwrote {RESULTS / 'summary_temperature.json'} and scored_temperature.jsonl")


if __name__ == "__main__":
    main()
