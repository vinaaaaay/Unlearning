"""Extra analyses for the mid-term report (round-1 grader comments).

Uses the existing greedy and sweep outputs, plus the retain-author and
world-knowledge controls when their files exist. All CIs: paired bootstrap over
authors, as in score.py.

  1. Orig - clean, and unlearned - clean, with lenient and strict scoring
  2. Hop 3 with chain-consistent scoring (the true country must also be named)
  3. Item-level leak: items an unlearned model answers and the clean model does not
  4. "Hard" authors: clean 0/5 and orig >= 4/5 (hop 1)
  5. Base-model refusals: accuracy on outputs without the uncertainty flag
  6. New hits under sampling: are they on items that clean already answers?
  7. Retain-author control and world-knowledge control

Writes results/analysis_extra.json and prints a report.
Usage: python analysis_extra.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from score import N_BOOT, boot_ci, extract, has_alias, score_row

HERE = Path(__file__).parent
RES = HERE / "results"
MODELS = ["base", "orig", "clean", "graddiff", "npo", "rmu"]
UNL = ["graddiff", "npo", "rmu"]
HOPS = [1, 2, 3]
rng = random.Random(0)


def load_probes(name):
    return {p["id"]: p for p in map(json.loads, open(HERE / "data" / name))}


def load_gen(path):
    out = defaultdict(list)
    for r in map(json.loads, open(path)):
        out[r["id"]].append(r["generation"])
    return out


def score_set(probes, suffix=""):
    """{model: {id: row}} for greedy files gen_<m><suffix>.jsonl."""
    data = {}
    for m in MODELS:
        path = RES / f"gen_{m}{suffix}.jsonl"
        if not path.exists():
            continue
        gens = load_gen(path)
        if len(gens) != len(probes):
            print(f"skip {path.name}: {len(gens)}/{len(probes)}")
            continue
        rows = {}
        for pid, p in probes.items():
            g = gens[pid][0]
            if "bridge_city" in p:
                r = score_row(g, p)
            else:  # world-knowledge control: same answer rule, no chain fields
                span, marked = extract(g)
                ok = has_alias(span, p["gold_aliases"], p["neg_phrases"])
                r = {"lenient": ok, "strict": ok and marked, "marked": marked, "refusal": False}
            if p.get("hop") == 3:
                co = p["bridge_country"]
                from gold import COUNTRY_ALIASES
                c = COUNTRY_ALIASES[co]
                r["chain"] = r["lenient"] and has_alias(g, c["pos"], c.get("neg", []))
            else:
                r["chain"] = r["lenient"]
            rows[pid] = {**r, "hop": p.get("hop"), "author": p.get("author_idx"),
                         "kind": p.get("kind")}
        data[m] = rows
    return data


def per_author(rows, hop, key):
    acc = defaultdict(list)
    for r in rows.values():
        if r["hop"] == hop:
            acc[r["author"]].append(float(r[key]))
    return {a: sum(v) / len(v) for a, v in acc.items()}


def mean(d):
    return sum(d.values()) / len(d)


def paired(rows_a, rows_b, hop, key):
    a, b = per_author(rows_a, hop, key), per_author(rows_b, hop, key)
    diff = {k: a[k] - b[k] for k in a}
    lo, hi = boot_ci(diff, rng)
    return {"diff": mean(diff), "ci95": [lo, hi]}


def fmt(r):
    lo, hi = r["ci95"]
    mark = "*" if lo > 0 or hi < 0 else " "
    return f"{r['diff'] * 100:+5.0f} [{lo * 100:+4.0f},{hi * 100:+4.0f}]{mark}"


def main():
    out = {}
    probes = load_probes("probes.jsonl")
    d = score_set(probes)

    # 1. orig/unlearned - clean, lenient and strict
    print("=== 1. Model - clean (points, paired author bootstrap; * = CI excludes 0)")
    out["vs_clean"] = {}
    for key in ("lenient", "strict"):
        print(f"--- {key}")
        for m in ["orig"] + UNL:
            cells = []
            for h in HOPS:
                r = paired(d[m], d["clean"], h, key)
                out["vs_clean"].setdefault(key, {}).setdefault(m, {})[h] = r
                cells.append(fmt(r))
            print(f"{m:<9} " + "  ".join(cells))
    out["acc"] = {key: {m: {h: mean(per_author(d[m], h, key)) for h in HOPS} for m in MODELS}
                  for key in ("lenient", "strict", "chain")}
    print("strict accuracy:", {m: [round(out["acc"]["strict"][m][h] * 100) for h in HOPS] for m in MODELS})

    # detection limit: mean CI half-width of the unlearned - clean differences
    hw = [(r["ci95"][1] - r["ci95"][0]) / 2 for m in UNL for r in out["vs_clean"]["lenient"][m].values()]
    out["ci_halfwidth_mean"] = sum(hw) / len(hw)
    print(f"mean CI half-width of unlearned - clean (lenient): {out['ci_halfwidth_mean'] * 100:.0f} points")

    # 2. hop 3 chain-consistent
    print("\n=== 2. Hop 3, chain-consistent (language correct AND true country named)")
    out["chain_h3"] = {}
    for m in MODELS:
        out["chain_h3"][m] = out["acc"]["chain"][m][3]
    print({m: round(v * 100) for m, v in out["chain_h3"].items()})
    for m in ["orig"] + UNL:
        r = paired(d[m], d["clean"], 3, "chain")
        out.setdefault("chain_h3_vs_clean", {})[m] = r
        print(f"{m:<9} - clean: {fmt(r)}")

    # 3. item-level leak
    print("\n=== 3. Item level (greedy, lenient): b = model right & clean wrong; c = clean right & model wrong")
    out["items"] = {}
    for m in ["orig"] + UNL + ["base"]:
        for h in HOPS:
            ids = [i for i, r in d[m].items() if r["hop"] == h]
            b = [i for i in ids if d[m][i]["lenient"] and not d["clean"][i]["lenient"]]
            c = [i for i in ids if d["clean"][i]["lenient"] and not d[m][i]["lenient"]]
            b_leaky = [i for i in b if d["orig"][i]["lenient"] and not d["base"][i]["lenient"]]
            out["items"].setdefault(m, {})[h] = {"b": len(b), "c": len(c), "b_orig_right_base_wrong": len(b_leaky)}
        print(f"{m:<9} " + "  ".join(f"H{h}: b={out['items'][m][h]['b']:>2} c={out['items'][m][h]['c']:>2} "
                                     f"(b&orig&!base={out['items'][m][h]['b_orig_right_base_wrong']})" for h in HOPS))

    # accuracy on items that clean gets wrong and orig gets right ("forgotten-only" items)
    print("\n--- accuracy on items where clean is wrong and orig is right (per hop)")
    out["clean_wrong_orig_right"] = {}
    for h in HOPS:
        ids = [i for i, r in d["clean"].items() if r["hop"] == h and not r["lenient"] and d["orig"][i]["lenient"]]
        row = {m: sum(d[m][i]["lenient"] for i in ids) for m in MODELS}
        out["clean_wrong_orig_right"][h] = {"n": len(ids), **row}
        print(f"H{h}: n={len(ids):>2}  " + "  ".join(f"{m}={row[m]}" for m in MODELS))

    # 4. hard authors at hop 1
    pa = {m: defaultdict(int) for m in MODELS}
    for m in MODELS:
        for r in d[m].values():
            if r["hop"] == 1:
                pa[m][r["author"]] += r["lenient"]
    hard = [a for a in range(20) if pa["clean"][a] == 0 and pa["orig"][a] >= 4]
    from gold import AUTHORS
    out["hard_authors"] = {"authors": [AUTHORS[a][0] for a in hard],
                           "correct_of": 5 * len(hard),
                           **{m: sum(pa[m][a] for a in hard) for m in MODELS}}
    print("\n=== 4. Hard authors (hop 1: clean 0/5, orig >= 4/5):", out["hard_authors"])

    # 5. base refusals
    print("\n=== 5. Base model: uncertainty flag and accuracy on unflagged outputs")
    out["base_refusal"] = {}
    for h in HOPS:
        rows = [r for r in d["base"].values() if r["hop"] == h]
        flagged = [r for r in rows if r["refusal"]]
        unflag = [r for r in rows if not r["refusal"]]
        acc_un = sum(r["lenient"] for r in unflag) / len(unflag)
        out["base_refusal"][h] = {"flagged": len(flagged), "unflagged": len(unflag),
                                  "acc_unflagged": acc_un,
                                  "correct_unflagged": sum(r["lenient"] for r in unflag)}
        print(f"H{h}: flagged {len(flagged)}/100, accuracy on unflagged {sum(r['lenient'] for r in unflag)}/{len(unflag)} = {acc_un:.0%}")
    out["clean_flagged"] = sum(r["refusal"] for r in d["clean"].values())
    out["orig_flagged"] = sum(r["refusal"] for r in d["orig"].values())
    print(f"clean flagged {out['clean_flagged']}/300, orig flagged {out['orig_flagged']}/300")

    # 6. new hits under sampling (T=0.7, any-of-5) on hop 1
    print("\n=== 6. Hop-1 items wrong with greedy but right in any-of-5 at T=0.7")
    out["new_hits"] = {}
    for m in ["rmu", "npo", "base", "clean"]:
        path = RES / f"gen_{m}_T0.7.jsonl"
        gens = load_gen(path)
        new = [i for i, p in probes.items() if p["hop"] == 1 and not d[m][i]["lenient"]
               and any(score_row(g, p)["lenient"] for g in gens[i])]
        easy = [i for i in new if d["clean"][i]["lenient"]]
        out["new_hits"][m] = {"new": len(new), "clean_greedy_right": len(easy)}
        print(f"{m:<6} new hits {len(new):>2}, of which clean already right with greedy: {len(easy)}")

    # 7a. retain-author control
    rprobes = load_probes("probes_retain.jsonl")
    rd = score_set(rprobes, "_retain")
    if rd:
        print("\n=== 7a. Retain-author control (greedy, lenient accuracy, 95% CI)")
        out["retain"] = {}
        for m in MODELS:
            if m not in rd:
                continue
            cells = []
            for h in HOPS:
                pa_ = per_author(rd[m], h, "lenient")
                lo, hi = boot_ci(pa_, rng)
                out["retain"].setdefault(m, {})[h] = {"acc": mean(pa_), "ci95": [lo, hi]}
                cells.append(f"{mean(pa_) * 100:4.0f} [{lo * 100:3.0f},{hi * 100:3.0f}]")
            print(f"{m:<9} " + "  ".join(cells))
        if "clean" in rd:
            print("--- retain: model - clean")
            for m in ["orig"] + UNL:
                if m in rd:
                    rr = [paired(rd[m], rd["clean"], h, "lenient") for h in HOPS]
                    out.setdefault("retain_vs_clean", {})[m] = dict(zip(HOPS, rr))
                    print(f"{m:<9} " + "  ".join(fmt(r) for r in rr))
            print("--- forget minus retain accuracy (what an observer without the clean model sees)")
            for m in MODELS:
                if m in rd:
                    g = [out["acc"]["lenient"][m][h] - out["retain"][m][h]["acc"] for h in HOPS]
                    out.setdefault("forget_minus_retain", {})[m] = dict(zip(HOPS, g))
                    print(f"{m:<9} " + "  ".join(f"{x * 100:+5.0f}" for x in g))

    # 7b. world-knowledge control
    cprobes = load_probes("probes_control.jsonl")
    cd = score_set(cprobes, "_control")
    if cd:
        print("\n=== 7b. World-knowledge control (correct / total)")
        out["control"] = {}
        for m in MODELS:
            if m not in cd:
                continue
            by = defaultdict(list)
            for r in cd[m].values():
                by[r["kind"]].append(r["lenient"])
            out["control"][m] = {k: [sum(v), len(v)] for k, v in by.items()}
            print(f"{m:<9} " + "  ".join(f"{k}: {sum(v)}/{len(v)}" for k, v in by.items()))

    (RES / "analysis_extra.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RES / 'analysis_extra.json'}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Round-2 additions: fair guess level on the selected items, concentration of
# hits, and the forget-vs-retain gap with CIs.
# ---------------------------------------------------------------------------
def round2():
    from gold import AUTHORS
    probes = load_probes("probes.jsonl")
    d = score_set(probes)
    out = json.loads((RES / "analysis_extra.json").read_text())
    r2 = {}
    print("\n=== 8. Selected items (clean wrong, orig right; greedy)")
    for h in HOPS:
        ids = [i for i, r in d["clean"].items() if r["hop"] == h and not r["lenient"] and d["orig"][i]["lenient"]]
        authors = sorted({d["clean"][i]["author"] for i in ids})
        row = {"n": len(ids), "authors": len(authors)}
        # per-sample accuracy at T=0.7 (mean of 5 samples), summed over items
        for m in ["base", "clean", "graddiff", "npo", "rmu"]:
            gens = load_gen(RES / f"gen_{m}_T0.7.jsonl")
            per = {i: sum(score_row(g, probes[i])["lenient"] for g in gens[i]) / len(gens[i]) for i in ids}
            row[f"{m}_t07_hits"] = sum(per.values())
            row[f"{m}_t07"] = per
        # author bootstrap of greedy RMU - base and RMU - clean(T0.7 per-sample) on these items
        for m, ref, key in [("rmu", "base", "greedy"), ("npo", "base", "greedy"),
                            ("rmu", "clean", "t07"), ("npo", "clean", "t07")]:
            by_a = defaultdict(list)
            for i in ids:
                a = d["clean"][i]["author"]
                mv = float(d[m][i]["lenient"])
                rv = float(d[ref][i]["lenient"]) if key == "greedy" else row[f"{ref}_t07"][i]
                by_a[a].append(mv - rv)
            means = {a: sum(v) / len(v) for a, v in by_a.items()}
            # weight authors by item count: resample authors, average over pooled items
            stats = []
            al = list(by_a)
            for _ in range(N_BOOT):
                pool = [x for a in (rng.choice(al) for _ in al) for x in by_a[a]]
                stats.append(sum(pool) / len(pool))
            stats.sort()
            pooled = sum(x for v in by_a.values() for x in v) / len(ids)
            row[f"{m}_minus_{ref}_{key}"] = {"diff": pooled, "ci95": [stats[int(.025 * N_BOOT)], stats[int(.975 * N_BOOT)]]}
        # hit concentration
        for m in ["npo", "rmu"]:
            hits = [d["clean"][i]["author"] for i in ids if d[m][i]["lenient"]]
            cnt = defaultdict(int)
            for a in hits:
                cnt[AUTHORS[a][0]] += 1
            row[f"{m}_hits_by_author"] = dict(cnt)
        for k in ["base_t07", "clean_t07", "graddiff_t07", "npo_t07", "rmu_t07"]:
            row.pop(k)
        r2[h] = row
        print(f"H{h}: n={row['n']} ({row['authors']} authors)  T0.7 hits/sample: "
              + " ".join(f"{m}={row[m + '_t07_hits']:.1f}" for m in ["base", "clean", "graddiff", "npo", "rmu"]))
        for k in ["rmu_minus_base_greedy", "npo_minus_base_greedy", "rmu_minus_clean_t07", "npo_minus_clean_t07"]:
            v = row[k]
            print(f"     {k}: {v['diff'] * 100:+.0f} [{v['ci95'][0] * 100:+.0f}, {v['ci95'][1] * 100:+.0f}]")
        print("     hits by author: npo", row["npo_hits_by_author"], " rmu", row["rmu_hits_by_author"])

    # NPO "Beijing" prior: what NPO answers for the Chinese-named authors, city question
    print("\n--- NPO greedy city answers for Chinese-named authors")
    gens = load_gen(RES / "gen_npo.jsonl")
    for i, (name, city, _) in enumerate(AUTHORS):
        if name in ("Xin Lee Williams", "Wei-Jun Chen", "Hsiao Yun-Hwa"):
            ans = [gens[f"a{i:02d}_h1_t{t}"][0][-70:].replace("\n", " ") for t in range(5)]
            beij = sum("Beijing" in a for a in ans)
            r2.setdefault("npo_beijing", {})[name] = {"true_city": city, "says_beijing": beij}
            print(f"  {name} (true {city}): says Beijing in {beij}/5")

    # forget - retain gap, unpaired author bootstrap, and its difference to clean
    print("\n=== 9. Forget-minus-retain gap (unpaired author bootstrap), and gap minus clean gap")
    rprobes = load_probes("probes_retain.jsonl")
    rd = score_set(rprobes, "_retain")
    gap = {}
    for h in HOPS:
        fa = {m: per_author(d[m], h, "lenient") for m in MODELS}
        ra = {m: per_author(rd[m], h, "lenient") for m in MODELS}
        for m in MODELS:
            st, stc = [], []
            fk, rk = list(fa[m]), list(ra[m])
            for _ in range(N_BOOT):
                fs = [rng.choice(fk) for _ in fk]
                rs = [rng.choice(rk) for _ in rk]
                g_m = sum(fa[m][a] for a in fs) / len(fs) - sum(ra[m][a] for a in rs) / len(rs)
                g_c = sum(fa["clean"][a] for a in fs) / len(fs) - sum(ra["clean"][a] for a in rs) / len(rs)
                st.append(g_m)
                stc.append(g_m - g_c)
            st.sort(); stc.sort()
            g = mean(fa[m]) - mean(ra[m])
            gc = g - (mean(fa["clean"]) - mean(ra["clean"]))
            gap.setdefault(m, {})[h] = {
                "gap": g, "ci95": [st[int(.025 * N_BOOT)], st[int(.975 * N_BOOT)]],
                "minus_clean_gap": gc, "minus_clean_ci95": [stc[int(.025 * N_BOOT)], stc[int(.975 * N_BOOT)]]}
    for m in MODELS:
        print(f"{m:<9} " + "  ".join(
            f"gap {gap[m][h]['gap'] * 100:+4.0f} [{gap[m][h]['ci95'][0] * 100:+4.0f},{gap[m][h]['ci95'][1] * 100:+4.0f}]"
            f" vs clean {gap[m][h]['minus_clean_gap'] * 100:+4.0f} [{gap[m][h]['minus_clean_ci95'][0] * 100:+4.0f},{gap[m][h]['minus_clean_ci95'][1] * 100:+4.0f}]"
            for h in HOPS))
    r2["forget_retain_gap"] = gap
    out["round2"] = r2
    (RES / "analysis_extra.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    round2()
