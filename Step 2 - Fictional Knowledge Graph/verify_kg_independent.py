"""Independent re-verification of fictional_kg.json (shares no code with generate_kg.cpp).

Re-derives every property from the raw JSON: graph structure, split, chain
extraction (recomputed from scratch and compared set-wise), labels, metadata
counts, name hygiene, and a few leakage/confound diagnostics.

Usage: python verify_kg_independent.py [fictional_kg.json]
"""
import json, sys, collections, itertools

path = sys.argv[1] if len(sys.argv) > 1 else "fictional_kg.json"
kg = json.load(open(path))
md, ents, tris, chains = kg["metadata"], kg["entities"], kg["triples"], kg["evaluation_chains"]
N = md["num_spines"]
fails = []
def check(cond, msg):
    if not cond: fails.append(msg)

TYPES = ["Person", "Person", "Person", "Company", "Company", "City"]
REL = {1: "mentor_of", 2: "mentor_of", 3: "works_at", 4: "acquired_by", 5: "headquartered_in"}
SIG = {"mentor_of": ("Person", "Person"), "works_at": ("Person", "Company"),
       "acquired_by": ("Company", "Company"), "headquartered_in": ("Company", "City")}

E = {e["id"]: e for e in ents}
T = {t["id"]: t for t in tris}
check(len(ents) == 6 * N == len(E), "entity count / id uniqueness")
check(len({e["name"] for e in ents}) == 6 * N, "name uniqueness")
check(len(tris) == 5 * N == len(T), "triple count / id uniqueness")
check(all(TYPES[e["layer"]] == e["type"] for e in ents), "entity type vs layer")
check(all(sum(1 for e in ents if e["layer"] == L) == N for L in range(6)), "layer sizes")

# structure
out, inn = collections.Counter(), collections.Counter()
adj = collections.defaultdict(list)
for t in tris:
    s, o = E[t["subject_id"]], E[t["object_id"]]
    check(t["relation"] == REL[t["hop_layer"]], f"relation/hop {t['id']}")
    check(SIG[t["relation"]] == (s["type"], o["type"]) == (t["subject_type"], t["object_type"]), f"types {t['id']}")
    check(s["layer"] == t["hop_layer"] - 1 and o["layer"] == t["hop_layer"], f"layers {t['id']}")
    check(s["name"] == t["subject_name"] and o["name"] == t["object_name"], f"denormalised names {t['id']}")
    out[t["subject_id"]] += 1; inn[t["object_id"]] += 1; adj[t["subject_id"]].append(t["object_id"])
for e in ents:
    check(out[e["id"]] == (1 if e["layer"] < 5 else 0), f"outdeg {e['id']}")
    check(inn[e["id"]] == (1 if e["layer"] > 0 else 0), f"indeg {e['id']}")

# reconstruct spines purely by walking from each layer-0 node
spines = {}
for e in ents:
    if e["layer"] != 0: continue
    path_ = [e["id"]]
    while adj[path_[-1]]:
        path_.append(adj[path_[-1]][0])
        check(len(path_) <= 6, "cycle / overlong path")
        if len(path_) > 6: break
    check(len(path_) == 6, f"spine length from {e['id']}")
    spines[e["spine_id"]] = path_
check(sorted(spines) == list(range(N)), "spine ids 0..N-1")
check(len({v for p in spines.values() for v in p}) == 6 * N, "spines not vertex-disjoint")
check(all(E[v]["spine_id"] == i for i, p in spines.items() for v in p), "entity.spine_id")

# split
forget = [t for t in tris if t["is_forget"]]
check(10 * len(forget) == len(tris), "10% forget")
check(md["num_forget_triples"] == len(forget) and md["num_retain_triples"] == len(tris) - len(forget), "metadata split")
fl = collections.Counter(t["hop_layer"] for t in forget)
check(all(fl[m] == N // 10 for m in range(1, 6)), f"forget per layer {dict(fl)}")
for i in range(N):
    f_i = [t for t in tris if t["spine_id"] == i and t["is_forget"]]
    if i < N // 2:
        check(len(f_i) == 1 and f_i[0]["hop_layer"] == i % 5 + 1, f"target spine {i}")
    else:
        check(not f_i, f"retain spine {i}")

# chains: recompute expected set from scratch
def label(k, p): return "single" if k == 1 else "start" if p == 1 else "end" if p == k else "middle"
expected = {}
for i in range(N // 2):
    ms = i % 5 + 1
    for k in range(1, 6):
        for s in range(0, 6 - k):
            if s + 1 <= ms <= s + k:
                expected[(True, i, k, s)] = label(k, ms - s)
                expected[(False, i + N // 2, k, s)] = "none"
got = {}
tri_by_spine_layer = {(t["spine_id"], t["hop_layer"]): t for t in tris}
for c in chains:
    key = (c["is_target_chain"], c["spine_id"], c["length_k"], c["start_layer"])
    check(key not in got, f"duplicate chain {key}")
    got[key] = c["forget_position"]
    i, k, s = key[1], key[2], key[3]
    check(c["node_ids"] == spines[i][s:s + k + 1], f"nodes {c['chain_id']}")
    check(c["node_names"] == [E[v]["name"] for v in c["node_ids"]], f"names {c['chain_id']}")
    exp_t = [tri_by_spine_layer[(i, m)]["id"] for m in range(s + 1, s + k + 1)]
    check(c["triple_ids"] == exp_t, f"triples {c['chain_id']}")
    check(c["relations"] == [REL[m] for m in range(s + 1, s + k + 1)], f"relations {c['chain_id']}")
    check(c["terminal_entity_id"] == c["node_ids"][-1] and c["end_layer"] == s + k, f"terminal {c['chain_id']}")
    nf = sum(T[x]["is_forget"] for x in c["triple_ids"])
    check(nf == (1 if c["is_target_chain"] else 0), f"forget count {c['chain_id']}")
    if c["is_target_chain"]:
        p = [T[x]["is_forget"] for x in c["triple_ids"]].index(True) + 1
        check(c["forget_relative_pos"] == p and c["forget_edge_layer"] == s + p, f"pos {c['chain_id']}")
        check(c["position_bucket"] == ("start" if k == 1 else c["forget_position"]), "bucket")
check(got == expected, f"chain set mismatch: missing={len(set(expected)-set(got))} extra={len(set(got)-set(expected))}")
by_id = {c["chain_id"]: c for c in chains}
for c in chains:
    m = by_id.get(c["matched_chain_id"])
    check(m is not None and m["matched_chain_id"] == c["chain_id"] and (m["length_k"], m["start_layer"]) ==
          (c["length_k"], c["start_layer"]) and m["is_target_chain"] != c["is_target_chain"], f"matching {c['chain_id']}")
nt = sum(c["is_target_chain"] for c in chains)
check(md["num_target_chains"] == nt == 35 * N // 10, "num_target_chains")
check(md["num_retain_control_chains"] == len(chains) - nt == nt, "num_retain_control_chains")

# terminal uniformity per (s,k) slice over all N spines
for k in range(1, 6):
    for s in range(0, 6 - k):
        terms = collections.Counter(spines[i][s + k] for i in range(N))
        check(len(terms) == N and set(terms.values()) == {1}, f"terminal uniformity k={k} s={s}")

# name hygiene
BL = ["real", "amer", "chin", "indi", "kor", "fran", "lond", "pari", "toky"]
check(not [e for e in ents if any(b in e["name"].lower().replace(" ", "") for b in BL)], "blacklist")

# ---------------- diagnostics (informational) ----------------
print(f"Independent verifier on {path}: {'PASS' if not fails else 'FAIL'} ({len(fails)} failures)")
for f in fails[:20]: print("  FAIL:", f)

tab = collections.Counter((c["length_k"], c["forget_position"]) for c in chains if c["is_target_chain"])
print("\n(k, position) -> set of forgotten relations contributing (confound check):")
conf = collections.defaultdict(collections.Counter)
for c in chains:
    if c["is_target_chain"]:
        conf[(c["length_k"], c["forget_position"])][REL[c["forget_edge_layer"]] + f"@L{c['forget_edge_layer']}"] += 1
for key in sorted(conf, key=lambda x: (x[0], ["single", "start", "middle", "end"].index(x[1]))):
    print(f"  k={key[0]} {key[1]:>6}: {tab[key]:4d}  {dict(conf[key])}")

try:
    import wordfreq
    hits = []
    for e in ents:
        for part in e["name"].split():
            if e["type"] == "Company" and part == e["name"].split()[-1]: continue
            z = wordfreq.zipf_frequency(part.lower(), "en")
            if z > 0: hits.append((part, round(z, 2)))
    print(f"\nwordfreq-attested name parts (zipf>0): {len(hits)}; max zipf = {max([h[1] for h in hits], default=0)}")
    print("  sample:", sorted(hits, key=lambda h: -h[1])[:12])
except ImportError:
    pass
sys.exit(1 if fails else 0)
