# Step 2 — Fictional KG: implementation & critical evaluation

## Files
| File | Purpose |
|---|---|
| `generate_kg.cpp` | C++17, standard library only. Builds the graph, runs `verify_kg()`, prints the summary + LaTeX table, writes the JSON. |
| `fictional_kg.json` | Output for `N=400, seed=42` (2400 entities, 2000 triples, 200 forget / 1800 retain, 1400 target + 1400 control chains). |
| `run_log.txt` | Console output of the run plus the independent verifier. |
| `verify_kg_independent.py` | Python re-check of the JSON that shares no code with the C++. It recomputes spines, the split, and the full chain set from scratch. |
| `make_blocklist.py` | One-off script that regenerates the lexical blocklist embedded in the C++. Needs `wordfreq`. Run it only if the syllable grammar changes. |

Build/run: `g++ -std=c++17 -O2 -Wall -Wextra generate_kg.cpp -o generate_kg && ./generate_kg`
(compiles with zero warnings even with `-Wpedantic -Wshadow -Wconversion`; runtime ≈ 30 ms).

## Step-by-step evaluation of the spec

**§2.1–2.2 Layers and bijective edges: correct as specified.** Composing 5 permutations gives N vertex-disjoint length-5 paths. Out- and in-degrees are exactly as stated, there are no chords and no diamonds, and every (s,k) slice maps the N spines bijectively onto V_{s+k}.

**§2.3 Split: correct.** 0.5N / 5N = 10% exactly. `m*(i) = (i mod 5)+1` puts N/10 forget edges in each hop layer, and N multiple of 10 is exactly the condition this needs. Each chain contains at most one forgotten link, because each spine has at most one and spines are disjoint. The forgotten fact also cannot be inferred from the retain set, since every vertex has in-degree and out-degree ≤ 1.

**§2.4 / §5 Inconsistency 1: chain counts.** The metadata example says `num_target_chains = 3000`, and 3000 = 200 spines × 15 = *all* sub-paths. The filter rule `s+1 ≤ m* ≤ s+k` keeps m*(6−m*) sub-paths per spine (5, 8, 9, 8, 5), so the correct count is **3.5N = 1400**. I followed the mathematical definition and noted the example as wrong.

**§2.4 Inconsistency 2: "matched" controls.** The spec doesn't define what "matched" means. I pair target spine i with pure-retain spine i+N/2. Every kept (s,k) slice on spine i gets one control chain on spine i+N/2 with the same slice, so the two sets have the same (k, s, relation-sequence) distribution. The result is 1400 controls, not 3000. Every control stores `matched_chain_id` and `matched_forget_position`.

**§2.4 k=1 label.** `forget_position = "single"`, and a separate field `position_bucket = "start"` handles the 1-hop baseline indexing. Controls have `"none"` in both, and `null` for `forget_edge_layer` and `forget_relative_pos`.

**§2.4 Important design caveat: position is confounded with relation type.** With fixed typed layers, `forget_position` at a given k determines which relation was forgotten (see the confound table in `run_log.txt`):
- k=5 `start` is always `mentor_of@L1`.
- k=5 `end` is always `headquartered_in@L5`.
- k=4 `end` is only L4 or L5.

A raw start-vs-middle-vs-end comparison therefore mixes up *position* with *relation / entity type*. **Recommendation:** report effects stratified by (or regressed on) `forget_edge_layer`. Compare positions only within a fixed forget layer, e.g. m*=3 appears as start, middle and end across different k and s.

**§3 Names: the spec's filter is insufficient.** The 9-substring blacklist leaves many real words and names that the grammar produces, such as *Simon, Lemon, Mason, Nolan, Melon, Solar, Brazil, Demon, Denim, Ketosis*. These bring back the very priors the spec says it removes. I added a lexical blocklist embedded in the C++ (so there are still no runtime dependencies): 4222 roots + 150 city names attested in English (wordfreq Zipf ≥ 1.0). After filtering, **0** name parts in the output are attested words. Other choices:
- Every root is globally unique across all 6N entities. This is stricter than unique full names, and it stops a shared surname from hinting at a link.
- Names are drawn at random, not enumerated in order. Ordered enumeration would make target-spine names (low indices) look systematically different from retain-spine names.
- City suffixes and company suffixes cycle by index within the layer, so they are exactly balanced.
- Company suffix balance is exact only when N is a multiple of 8. The program prints a note otherwise.

Note that subword-tokenizer priors (e.g. "-ia" sounding like a country) can't be removed fully. This is a limit of the method.

**§2.2 / §5 Determinism caveat: `std::shuffle` is not portable.** `mt19937_64` is fully specified by the standard, but `std::shuffle` is implementation-defined, so MSVC, libstdc++ and libc++ give different graphs for seed 42. The default is therefore an explicit Fisher–Yates shuffle with unbiased rejection sampling. It has the same uniform distribution and is bit-identical on any compiler. `--std-shuffle` gives the literal behaviour from the spec. Names use a separate PRNG stream (`seed XOR 0x9E37…`), so the topology depends only on the seed.

**§4 Verification.**
- The checks use a macro that is never compiled out. A `-DNDEBUG` build still verifies, and it produces an identical output hash.
- The 10% check uses integer arithmetic (`10·|F| == |E|`), not float `==`.
- The DFS check enumerates *all* paths from every u_{i,a}. Each (a,b) pair must have exactly one path, of length b−a, and nothing off the spine may be reachable.
- Spec item 6.3 ("terminal entities uniform") can only hold over *all* spines, so it is checked per (s,k) slice over all N spines. Inside the target and control sets each terminal appears at most once.
- The program also checks that the chain set is complete (no missing or extra slices) and that the count equals the closed form 3.5N.

## Tests performed
- Default run: all 10 C++ checks pass, and the independent Python verifier passes.
- N ∈ {10, 20, 1000, 4000}, seed up to 2^64−1, `--std-shuffle`: all pass in both verifiers.
- Same seed run twice: identical SHA-256. Same with `-DNDEBUG`.
- Invalid arguments (`--num-spines 15`, `0`, `abc`, missing value) are rejected.
- Fault injection: a broken bijection in π₃ and a wrong forget rule (`i mod 4`) both make `verify_kg()` FAIL and refuse to write the JSON.
- Tampered JSON: one flipped `is_forget` and one wrong `forget_position` are both caught by the Python verifier.

## Target-chain counts (N=400)
| k | start | middle | end | total |
|---|---|---|---|---|
| 1 | 200 (single) | – | – | 200 |
| 2 | 160 | – | 160 | 320 |
| 3 | 120 | 120 | 120 | 360 |
| 4 | 80 | 160 | 80 | 320 |
| 5 | 40 | 120 | 40 | 200 |
| all | 600 | 400 | 400 | 1400 |
