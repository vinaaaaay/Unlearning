"""Step 9: accuracy, leakage above the clean model, and Leftover Knowledge.

  python analyze.py --size 0.6B

Reads results/<size>/gen/<tag>.jsonl and writes results/<size>/summary.json and
results/<size>/report.md.

Definitions (strict scoring, greedy decoding; A = accuracy):
  signal        A(M_learn) - A(M_clean,seed1) on target chains. A slice is reported as a
                leakage test only if the paired 95% CI of the signal is above zero.
  leak          A(M_unl) - A(M_clean,seed1), unclipped, paired over the same chains.
  LK            leak / signal (Leftover Knowledge), unclipped; "n/a" if |signal| < 0.01.
  kept chains   chains that M_learn answers and M_clean,seed1 does not (Step 5 filter).
  per-question  A(M_unl) on the kept chains minus A(M_clean,seed2) on the kept chains
  leak          (seed 2 did not select the chains, so its accuracy there is unbiased).
All CIs: bootstrap over spines (the unit of independence), 10,000 resamples.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from common import DATA, RESULTS, read_jsonl

N_BOOT = 10_000
RNG = np.random.default_rng(0)


def boot_ci(per_spine_num, per_spine_den):
    """Mean = sum(num)/sum(den) with a spine-bootstrap 95% CI."""
    num, den = np.asarray(per_spine_num, float), np.asarray(per_spine_den, float)
    if den.sum() == 0:
        return None
    w = RNG.multinomial(len(num), np.full(len(num), 1 / len(num)), size=N_BOOT)
    stats = (w @ num) / np.maximum(w @ den, 1e-9)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(num.sum() / den.sum()), "lo": float(lo), "hi": float(hi), "n": int(den.sum())}


class Scores:
    """Greedy strict scores of every model on every probe."""

    def __init__(self, size):
        self.probes = {p["id"]: p for p in read_jsonl(DATA / "probes.jsonl")}
        self.rows = {}
        for path in sorted((RESULTS / size / "gen").glob("*.jsonl")):
            if "_T" in path.stem:
                continue
            self.rows[path.stem] = {r["id"]: r for r in read_jsonl(path)}

    def ids(self, pred):
        return [i for i, p in self.probes.items() if pred(p)]

    def vec(self, tag, ids, key="strict"):
        return np.array([float(self.rows[tag][i][key]) for i in ids])

    def spine_sums(self, values, ids):
        num, den = defaultdict(float), defaultdict(float)
        for v, i in zip(values, ids):
            s = self.probes[i].get("spine_id", i)
            num[s] += v
            den[s] += 1
        keys = sorted(num, key=str)
        return [num[k] for k in keys], [den[k] for k in keys]

    def acc(self, tag, ids, key="strict"):
        return boot_ci(*self.spine_sums(self.vec(tag, ids, key), ids)) if ids else None

    def diff(self, a, b, ids):
        return boot_ci(*self.spine_sums(self.vec(a, ids) - self.vec(b, ids), ids)) if ids else None


def fmt(ci, pct=True, signed=False):
    if ci is None:
        return "n/a"
    m = 100 if pct else 1
    s = "+" if signed else ""
    return f"{ci['mean'] * m:{s}.1f} [{ci['lo'] * m:{s}.1f}, {ci['hi'] * m:{s}.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--seed", default="s1", help="seed tag of M_learn and of its unlearned models")
    ap.add_argument("--clean", default="clean_s1", help="clean model that defines guessing and the filter")
    ap.add_argument("--clean2", default="clean_s2", help="independent clean model for kept chains")
    args = ap.parse_args()

    S = Scores(args.size)
    orig = f"orig_{args.seed}"
    unl = [t for t in (f"graddiff_{args.seed}", f"npo_{args.seed}", f"rmu_{args.seed}") if t in S.rows]
    models = [t for t in ["base", orig, args.clean, args.clean2] if t in S.rows] + unl
    P = S.probes
    out = {"size": args.size, "models": models}
    md = [f"# Results: Qwen3-{args.size}, seed {args.seed}", "",
          "Strict accuracy in %, greedy decoding, 95% CI from a bootstrap over spines.", ""]

    # ---- 1. memorisation and utility ---------------------------------------------------
    slices = {
        "1-hop train Q, forget": lambda p: p["set"] == "qa_train" and p["is_forget"],
        "1-hop train Q, retain": lambda p: p["set"] == "qa_train" and not p["is_forget"],
        "1-hop held-out Q, forget": lambda p: p["set"] == "qa_heldout" and p["is_forget"],
        "1-hop held-out Q, retain": lambda p: p["set"] == "qa_heldout" and not p["is_forget"],
        "held-out statement, forget": lambda p: p["set"] == "stmt_heldout" and p["is_forget"],
        "held-out statement, retain": lambda p: p["set"] == "stmt_heldout" and not p["is_forget"],
        "multi-hop, target (through forgotten edge)": lambda p: p["set"] == "mh_target",
        "multi-hop, neighbour (same spine, not through)": lambda p: p["set"] == "mh_neighbour",
        "multi-hop, control spines": lambda p: p["set"] == "mh_control",
        "multi-hop, demo spines (trained)": lambda p: p["set"] == "mh_demo",
        "world knowledge": lambda p: p["set"] == "world",
    }
    out["overview"] = {}
    md += ["## 1. Accuracy of every model", "", "| probe set | n | " + " | ".join(models) + " |",
           "|---|---|" + "---|" * len(models)]
    for name, pred in slices.items():
        ids = S.ids(pred)
        accs = {m: S.acc(m, ids) for m in models}
        out["overview"][name] = accs
        md.append(f"| {name} | {len(ids)} | " + " | ".join(fmt(accs[m]) for m in models) + " |")
    md.append("")

    if not (orig in S.rows and args.clean in S.rows):
        (RESULTS / args.size / "report.md").write_text("\n".join(md))
        print("\n".join(md))
        return

    # ---- 2. leakage per slice -------------------------------------------------------------
    target = S.ids(lambda p: p["set"] == "mh_target")
    onehop = S.ids(lambda p: p["set"] == "qa_heldout" and p["is_forget"])
    stmt = S.ids(lambda p: p["set"] == "stmt_heldout" and p["is_forget"])
    direct = S.ids(lambda p: p["set"] == "qa_train" and p["is_forget"])
    groups = {"direct: 1-hop, training questions (used for tuning)": direct,
              "1-hop, held-out questions": onehop, "1-hop, held-out statements": stmt,
              "multi-hop, all (k=2..5)": target}
    for k in range(2, 6):
        groups[f"k={k}"] = [i for i in target if P[i]["length_k"] == k]
    for pos in ("start", "middle", "end"):
        groups[f"position={pos}"] = [i for i in target if P[i]["forget_position"] == pos]
    for k in range(2, 6):
        for pos in ("start", "middle", "end"):
            ids = [i for i in target if P[i]["length_k"] == k and P[i]["forget_position"] == pos]
            if ids:
                groups[f"k={k}, {pos}"] = ids
    for layer in range(1, 6):
        groups[f"forgotten edge layer {layer} ({P[next(i for i in target if P[i]['forget_edge_layer'] == layer)]['relations'][P[next(i for i in target if P[i]['forget_edge_layer'] == layer)]['forget_relative_pos'] - 1]})"] = \
            [i for i in target if P[i]["forget_edge_layer"] == layer]

    out["leakage"] = {}
    md += ["## 2. Leakage above the clean model", "",
           f"signal = A({orig}) - A({args.clean}); leak = A(unlearned) - A({args.clean}); "
           "LK = leak / signal. A slice is a valid test only if the CI of the signal is above zero.", "",
           "| slice | n | A(learn) | A(clean) | signal | " +
           " | ".join(f"A({u}) | leak({u}) | LK" for u in unl) + " |",
           "|---|---|---|---|---|" + "---|---|---|" * len(unl)]
    for name, ids in groups.items():
        sig = S.diff(orig, args.clean, ids)
        row = {"n": len(ids), "A_learn": S.acc(orig, ids), "A_clean": S.acc(args.clean, ids),
               "signal": sig, "valid": sig["lo"] > 0, "unlearned": {}}
        cells = []
        for u in unl:
            leak = S.diff(u, args.clean, ids)
            lk = None if abs(sig["mean"]) < 0.01 else leak["mean"] / sig["mean"]
            row["unlearned"][u] = {"A": S.acc(u, ids), "leak": leak, "LK": lk}
            cells += [fmt(row["unlearned"][u]["A"]), fmt(leak, signed=True),
                      "n/a" if lk is None else f"{lk:.2f}"]
        out["leakage"][name] = row
        md.append(f"| {name}{'' if row['valid'] else ' (no signal)'} | {len(ids)} | {fmt(row['A_learn'])} | "
                  f"{fmt(row['A_clean'])} | {fmt(sig, signed=True)} | " + " | ".join(cells) + " |")
    md.append("")

    # ---- 3. per-question leakage on kept chains ---------------------------------------------
    if args.clean2 in S.rows:
        out["kept"] = {}
        md += ["## 3. Per-question leakage on kept chains", "",
               f"kept = answered by {orig} and not by {args.clean}. Reference: {args.clean2} on the same chains.", "",
               "| slice | kept / all | A(clean2) | " + " | ".join(f"A({u}) | A({u}) - A(clean2)" for u in unl) + " |",
               "|---|---|---|" + "---|---|" * len(unl)]
        for name, ids in groups.items():
            keep = [i for i in ids if S.rows[orig][i]["strict"] and not S.rows[args.clean][i]["strict"]]
            row = {"n_all": len(ids), "n_kept": len(keep), "A_clean2": S.acc(args.clean2, keep), "unlearned": {}}
            cells = []
            for u in unl:
                row["unlearned"][u] = {"A": S.acc(u, keep), "leak": S.diff(u, args.clean2, keep)}
                cells += [fmt(row["unlearned"][u]["A"]), fmt(row["unlearned"][u]["leak"], signed=True)]
            out["kept"][name] = row
            md.append(f"| {name} | {len(keep)} / {len(ids)} | {fmt(row['A_clean2'])} | " + " | ".join(cells) + " |")
        md.append("")

    # ---- 4. where does the reasoning break? -----------------------------------------------------
    md += ["## 4. Hop-level trace on target chains", "",
           "Share of target chains (%) whose reasoning states the correct object of a hop, for the "
           "forgotten hop and for the other (retained) hops before and after it.", "",
           "| model | forgotten hop | retained hops before it | retained hops after it | marker present |",
           "|---|---|---|---|---|"]
    out["trace"] = {}
    for m in [orig, args.clean] + unl:
        f, before, after, marked = [], [], [], []
        for i in target:
            r, p = S.rows[m][i], P[i]
            j = p["forget_relative_pos"] - 1
            f.append(r["hops_ok"][j])
            before += r["hops_ok"][:j]
            after += r["hops_ok"][j + 1:]
            marked.append(r["marked"])
        out["trace"][m] = {k: float(np.mean(v)) for k, v in
                           dict(forgotten=f, before=before, after=after, marked=marked).items()}
        t = out["trace"][m]
        md.append(f"| {m} | {100 * t['forgotten']:.1f} | {100 * t['before']:.1f} | {100 * t['after']:.1f} | {100 * t['marked']:.1f} |")
    md.append("")

    # ---- 5. unlearning runs -------------------------------------------------------------------------
    md += ["## 5. Selected unlearning checkpoints (chosen with direct questions only)", "",
           "| model | status | lr | step | forget (direct) | retain | world |", "|---|---|---|---|---|---|---|"]
    out["unlearn"] = {}
    for u in unl:
        path = RESULTS / args.size / "unlearn" / f"{u}.json"
        if path.exists():
            s = json.load(open(path))
            out["unlearn"][u] = {k: s[k] for k in ("status", "selected", "reference", "args")}
            sel = s["selected"]
            md.append(f"| {u} | {s['status']} | {sel['lr']:g} | {sel['step']} | {100 * sel['forget']:.1f} | "
                      f"{100 * sel['retain']:.1f} (learned: {100 * s['reference']['retain']:.1f}) | "
                      f"{100 * sel['world']:.1f} (learned: {100 * s['reference']['world']:.1f}) |")
    md.append("")

    json.dump(out, open(RESULTS / args.size / f"summary_{args.seed}.json", "w"), indent=1)
    (RESULTS / args.size / f"report_{args.seed}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
