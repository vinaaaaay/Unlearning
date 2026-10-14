"""Score pilot generations and build the validation table.

Answer extraction:
  * If the generation contains "the answer is", the text after the LAST such
    marker (up to the end of that line) is the answer span.
  * Otherwise the whole generation is the span (models often skip the marker).
  "lenient" accuracy uses that rule; "strict" accuracy counts a probe as wrong
  when the marker is missing.
A span is correct if any gold alias occurs as a whole word (case-insensitive),
after removing negative phrases such as "North Korea".

Statistics: questions are clustered by author (5 phrasings x 20 authors), so 95%
CIs come from a bootstrap that resamples AUTHORS, not questions. The gate
Acc(unlearn) > Acc(clean) is tested with the same paired author bootstrap.

Usage: python score.py
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PROBES = HERE / "data" / "probes.jsonl"
ORDER = ["orig", "clean", "graddiff", "npo", "rmu"]
UNLEARNED = ["graddiff", "npo", "rmu"]
N_BOOT = 10_000

MARKER = re.compile(r"answer is\s*:?", re.I)


def has_alias(text, aliases, neg):
    for n in neg:
        text = re.sub(re.escape(n), " ", text, flags=re.I)
    return any(
        re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", text, re.I) for a in aliases
    )


def extract(gen):
    hits = list(MARKER.finditer(gen))
    if not hits:
        return gen, False
    return gen[hits[-1].end():].split("\n")[0], True


def score_row(gen, probe):
    span, marked = extract(gen)
    ok = has_alias(span, probe["gold_aliases"], probe["neg_phrases"])
    return {
        "lenient": ok,
        "strict": ok and marked,
        "marked": marked,
        # does the reasoning name the true (forgotten) birth city anywhere?
        "city_in_cot": has_alias(gen, [probe["bridge_city"].replace(" City", "")], []),
    }


def load():
    probes = {p["id"]: p for p in map(json.loads, open(PROBES))}
    scored = {}
    for m in ORDER:
        path = RESULTS / f"gen_{m}.jsonl"
        if not path.exists():
            continue
        rows = {}
        for r in map(json.loads, open(path)):
            p = probes[r["id"]]
            rows[r["id"]] = {**score_row(r["generation"], p), "hop": p["hop"],
                             "author": p["author_idx"], "truncated": r["truncated"]}
        if len(rows) == len(probes):
            scored[m] = rows
        else:
            print(f"skipping {m}: {len(rows)}/{len(probes)} generations")
    return probes, scored


def author_means(rows, hop, key):
    """Per-author mean of `key` over the 5 phrasings of a given hop."""
    acc = defaultdict(list)
    for r in rows.values():
        if r["hop"] == hop:
            acc[r["author"]].append(r[key])
    return {a: sum(v) / len(v) for a, v in acc.items()}


def boot_ci(per_author_vals, rng):
    authors = list(per_author_vals)
    stats = []
    for _ in range(N_BOOT):
        s = [per_author_vals[rng.choice(authors)] for _ in authors]
        stats.append(sum(s) / len(s))
    stats.sort()
    return stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT)]


def main():
    rng = random.Random(0)
    probes, scored = load()
    models = [m for m in ORDER if m in scored]
    hops = [1, 2, 3]
    out = {"accuracy": {}, "gate": {}, "leftover_knowledge": {}, "diagnostics": {}}

    print("\n=== Accuracy (lenient; 95% author-bootstrap CI)  [strict]")
    print(f"{'model':<10}" + "".join(f"{f'hop {h}':>26}" for h in hops))
    for m in models:
        line = f"{m:<10}"
        for h in hops:
            pa = author_means(scored[m], h, "lenient")
            mean = sum(pa.values()) / len(pa)
            lo, hi = boot_ci(pa, rng)
            strict = sum(author_means(scored[m], h, "strict").values()) / len(pa)
            out["accuracy"].setdefault(m, {})[h] = {
                "lenient": mean, "ci95": [lo, hi], "strict": strict}
            line += f"{mean:>8.0%} [{lo:4.0%},{hi:4.0%}] [{strict:4.0%}]"
        print(line)

    if "clean" in scored:
        print("\n=== Decision gate: Acc(unlearn) - Acc(clean), paired author bootstrap")
        for m in [m for m in UNLEARNED if m in scored]:
            for h in hops:
                u = author_means(scored[m], h, "lenient")
                c = author_means(scored["clean"], h, "lenient")
                diff = {a: u[a] - c[a] for a in u}
                d = sum(diff.values()) / len(diff)
                lo, hi = boot_ci(diff, rng)
                # one-sided bootstrap p-value for H1: diff > 0
                authors = list(diff)
                le0 = sum(
                    sum(diff[rng.choice(authors)] for _ in authors) <= 0
                    for _ in range(N_BOOT)
                ) / N_BOOT
                verdict = ("UNLEARN > CLEAN" if lo > 0 else
                           "UNLEARN < CLEAN" if hi < 0 else "not distinguishable")
                out["gate"].setdefault(m, {})[h] = {
                    "diff": d, "ci95": [lo, hi], "p_one_sided": le0, "verdict": verdict}
                print(f"{m:<10} hop {h}: {d:+6.1%}  CI [{lo:+.1%}, {hi:+.1%}]  "
                      f"p={le0:.4f}  -> {verdict}")

    if {"orig", "clean"} <= scored.keys():
        print("\n=== Leftover Knowledge = (U - C) / (O - C)   (clipped to [0,1])")
        for m in [m for m in UNLEARNED if m in scored]:
            parts = []
            for h in hops:
                o, c, u = (out["accuracy"][x][h]["lenient"] for x in ("orig", "clean", m))
                if abs(o - c) < 0.01:
                    lk, txt = None, "n/a (O~C)"
                else:
                    raw = (u - c) / (o - c)
                    lk = min(max(raw, 0.0), 1.0)
                    txt = f"{lk:.0%} (raw {raw:+.2f})"
                out["leftover_knowledge"].setdefault(m, {})[h] = lk
                parts.append(f"hop {h}: {txt:<18}")
            print(f"{m:<10} " + "  ".join(parts))

    print("\n=== Diagnostics: answer-marker rate / truncation rate / true city named in CoT")
    for m in models:
        rows = scored[m].values()
        n = len(scored[m])
        mk = sum(r["marked"] for r in rows) / n
        tr = sum(r["truncated"] for r in rows) / n
        city = {h: sum(author_means(scored[m], h, "city_in_cot").values()) / 20 for h in (2, 3)}
        out["diagnostics"][m] = {"marker_rate": mk, "trunc_rate": tr, "city_in_cot": city}
        print(f"{m:<10} marker {mk:4.0%}  truncated {tr:4.0%}  "
              f"city-in-CoT hop2 {city[2]:4.0%} hop3 {city[3]:4.0%}")

    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "scored.jsonl", "w") as f:
        for m in models:
            for pid, r in scored[m].items():
                f.write(json.dumps({"model": m, "id": pid, **r}) + "\n")
    print(f"\nwrote {RESULTS / 'summary.json'} and scored.jsonl")


if __name__ == "__main__":
    main()
