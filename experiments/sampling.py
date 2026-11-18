"""Sampling analysis on the target multi-hop chains.

Reads results/<size>/gen/<tag>_T<t>.jsonl (written by evaluate.py with --temperature) and
reports, for every model *including the clean model*:
  per-sample   mean strict accuracy over all samples
  any-of-N     share of chains where at least one of the N samples is correct
both with a spine-bootstrap 95% CI, and the paired difference to the clean model.

  python sampling.py --size 0.6B --seed s1
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from analyze import boot_ci, fmt
from common import DATA, RESULTS, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--seed", default="s1")
    ap.add_argument("--clean", default="clean_s1")
    args = ap.parse_args()

    probes = {p["id"]: p for p in read_jsonl(DATA / "probes.jsonl") if p["set"] == "mh_target"}
    ids = sorted(probes)
    spines = sorted({probes[i]["spine_id"] for i in ids})
    gen = RESULTS / args.size / "gen"
    models = [f"orig_{args.seed}", args.clean, f"graddiff_{args.seed}", f"npo_{args.seed}", f"rmu_{args.seed}"]
    temps = sorted({p.stem.split("_T")[1] for p in gen.glob("*_T*.jsonl")}, key=float)

    def per_spine(values):
        num, den = defaultdict(float), defaultdict(float)
        for i in ids:
            num[probes[i]["spine_id"]] += values[i]
            den[probes[i]["spine_id"]] += 1
        return [num[s] for s in spines], [den[s] for s in spines]

    def load(tag, t):
        """-> {id: [strict of each sample]}"""
        path = gen / (f"{tag}.jsonl" if t == "0" else f"{tag}_T{t}.jsonl")
        if not path.exists():
            return None
        d = defaultdict(list)
        for r in read_jsonl(path):
            if r["set"] == "mh_target":
                d[r["id"]].append(float(r["strict"]))
        return d

    out = {}
    md = [f"# Sampling on target multi-hop chains: Qwen3-{args.size}, seed {args.seed}", "",
          f"Strict accuracy in %, {len(ids)} chains, 95% CI from a bootstrap over spines. "
          f"'minus clean' is the paired difference to {args.clean} at the same temperature.", "",
          "| T | N | model | per-sample | any-of-N | per-sample minus clean | any-of-N minus clean |",
          "|---|---|---|---|---|---|---|"]
    for t in ["0"] + temps:
        data = {m: load(m, t) for m in models}
        if data[args.clean] is None:
            continue
        stat = {m: {"mean": {i: np.mean(d[i]) for i in ids}, "any": {i: max(d[i]) for i in ids},
                    "n": len(d[ids[0]])} for m, d in data.items() if d is not None}
        out[t] = {}
        for m, s in stat.items():
            c = stat[args.clean]
            row = {"n_samples": s["n"],
                   "per_sample": boot_ci(*per_spine(s["mean"])),
                   "any_of_n": boot_ci(*per_spine(s["any"])),
                   "per_sample_minus_clean": boot_ci(*per_spine({i: s["mean"][i] - c["mean"][i] for i in ids})),
                   "any_of_n_minus_clean": boot_ci(*per_spine({i: s["any"][i] - c["any"][i] for i in ids}))}
            out[t][m] = row
            md.append(f"| {t} | {s['n']} | {m} | {fmt(row['per_sample'])} | {fmt(row['any_of_n'])} | "
                      + ("- | - |" if m == args.clean else
                         f"{fmt(row['per_sample_minus_clean'], signed=True)} | {fmt(row['any_of_n_minus_clean'], signed=True)} |"))
    md.append("")
    json.dump(out, open(RESULTS / args.size / f"sampling_{args.seed}.json", "w"), indent=1)
    (RESULTS / args.size / f"sampling_{args.seed}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
