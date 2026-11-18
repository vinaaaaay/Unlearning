"""Step 5/7: run a model on the probe sets and score every answer.

Writes results/<size>/gen/<tag>.jsonl (greedy) or <tag>_T<t>.jsonl (sampling), one row
per probe (and per sample), with the generation and its scores.

Usage:
  python evaluate.py --size 0.6B --tag orig_s1                 # runs/<size>/orig_s1
  python evaluate.py --size 0.6B --tag base --model Qwen/Qwen3-0.6B
  python evaluate.py --size 0.6B --tag orig_s1 --temperature 0.7 --num-samples 5 --sets mh_target
"""

import argparse
import time

from common import (DATA, MODELS, RESULTS, RUNS, generate, has_name, load_model, load_tokenizer,
                    read_jsonl, score_generation, write_jsonl)

MAX_NEW = {"qa": 16, "completion": 16, "world": 16, "multihop": 128}
ALL_SETS = ["qa_train", "qa_heldout", "stmt_heldout", "mh_target", "mh_neighbour", "mh_control",
            "mh_demo", "world"]


def score(gen, probe):
    if probe["kind"] == "world":
        ok = any(has_name(gen, a) for a in probe["aliases"])
        return {"strict": ok, "lenient": ok}
    return score_generation(gen, probe)


def run_probes(model, tok, probes, temperature=0.0, num_samples=1, seed=0, batch_size=256):
    """Generate and score. Returns one row per (probe, sample)."""
    rows = []
    for kind, max_new in MAX_NEW.items():
        sub = [p for p in probes if p["kind"] == kind]
        if not sub:
            continue
        gens = generate(model, tok, [p["prompt"] for p in sub], max_new,
                        batch_size=batch_size if kind != "multihop" else batch_size // 2,
                        temperature=temperature, num_samples=num_samples, seed=seed)
        for p, samples in zip(sub, gens):
            for j, g in enumerate(samples):
                rows.append({"id": p["id"], "set": p["set"], "sample": j, "gen": g, **score(g, p)})
    return rows


def accuracy(rows, key="strict"):
    return sum(r[key] for r in rows) / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(MODELS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=None, help="path or HF id; default runs/<size>/<tag>")
    ap.add_argument("--sets", nargs="+", default=ALL_SETS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    path = args.model or str(RUNS / args.size / args.tag)
    probes = [p for p in read_jsonl(DATA / "probes.jsonl") if p["set"] in args.sets]
    tok = load_tokenizer(path)
    model = load_model(path).eval()
    t0 = time.time()
    rows = run_probes(model, tok, probes, args.temperature, args.num_samples, args.seed,
                      args.batch_size)
    sfx = "" if args.temperature == 0 else f"_T{args.temperature}"
    out = RESULTS / args.size / "gen" / f"{args.tag}{sfx}.jsonl"
    if out.exists() and set(args.sets) != set(ALL_SETS):  # partial run: keep the other sets
        rows = [r for r in read_jsonl(out) if r["set"] not in args.sets] + rows
    write_jsonl(out, rows)
    print(f"[{args.tag}{sfx}] {len(rows)} rows in {time.time() - t0:.0f}s -> {out}")
    for s in args.sets:
        sub = [r for r in rows if r["set"] == s]
        print(f"  {s:14s} n={len(sub):5d} strict={accuracy(sub):.3f} lenient={accuracy(sub, 'lenient'):.3f}")


if __name__ == "__main__":
    main()
