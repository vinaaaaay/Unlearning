"""Build the training corpora and the probe sets for Steps 4-7 from the KG (Step 2)
and the template corpus (Step 3). Deterministic; no randomness.

Spine roles (N = 400):
  target   spines   0..199  one forgotten edge each (from the KG)
  demo     spines 200..399, odd   multi-hop CoT demonstrations are in the training data
  control  spines 200..399, even  never appear in a multi-hop training text

The demo spines teach the answer format ("... Therefore, the answer is: X") and how to
chain facts. They contain only retain facts, and the same demonstrations go into the
corpus of the learned and of the clean model. No demonstration touches a target or a
control spine, so every evaluated multi-hop question is new to all models.

Outputs (experiments/data/):
  train_full.jsonl      statements + train 1-hop QA of all triples + demos
                        + real-world anchors (world_anchors.py)              -> M_learn
  train_retain.jsonl    the same without every text of a forget triple       -> M_clean
  forget.jsonl          statements + train QA of the forget triples, with the character
                        span of the answer (unlearning losses use only these tokens)
                        (the retain term of the unlearning losses uses train_retain.jsonl)
  probes.jsonl          all evaluation probes (see `sets` below)
"""

import json
from collections import Counter

from world_anchors import anchors
from common import (CORPUS, DATA, KG_PATH, multihop_cot, multihop_question, qa_prompt,
                    read_jsonl, write_jsonl)

WORLD = [  # world-knowledge control: (question, accepted answers)
    ("What is the capital of France?", ["Paris"]),
    ("What is the capital of Japan?", ["Tokyo"]),
    ("What is the capital of Italy?", ["Rome"]),
    ("What is the capital of Germany?", ["Berlin"]),
    ("What is the capital of Spain?", ["Madrid"]),
    ("What is the capital of Russia?", ["Moscow"]),
    ("What is the capital of Egypt?", ["Cairo"]),
    ("What is the capital of Canada?", ["Ottawa"]),
    ("What is the capital of China?", ["Beijing"]),
    ("What is the capital of the United Kingdom?", ["London"]),
    ("In which country is the city of Paris?", ["France"]),
    ("In which country is the city of Tokyo?", ["Japan"]),
    ("In which country is the city of Mumbai?", ["India"]),
    ("In which country is the city of Sydney?", ["Australia"]),
    ("In which country is the city of Toronto?", ["Canada"]),
    ("In which country is the city of Berlin?", ["Germany"]),
    ("Which company makes the iPhone?", ["Apple"]),
    ("Which company develops the Windows operating system?", ["Microsoft"]),
    ("Which company created the Android operating system?", ["Google"]),
    ("Which company owns Instagram?", ["Meta", "Facebook"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"]),
    ("Who painted the Mona Lisa?", ["Leonardo", "da Vinci"]),
    ("Who developed the theory of general relativity?", ["Einstein"]),
    ("Who was the first President of the United States?", ["Washington"]),
    ("Who wrote the novel Pride and Prejudice?", ["Austen"]),
    ("Who discovered penicillin?", ["Fleming"]),
    ("What is the largest planet in the Solar System?", ["Jupiter"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("What is the longest river in Africa?", ["Nile"]),
    ("What is the tallest mountain on Earth?", ["Everest"]),
    ("What language is mainly spoken in Brazil?", ["Portuguese"]),
    ("What language is mainly spoken in Mexico?", ["Spanish"]),
    ("What is the currency of Japan?", ["yen"]),
    ("How many days are there in a week?", ["7", "seven"]),
    ("What is two plus three?", ["5", "five"]),
    ("What colour do you get when you mix blue and yellow?", ["green"]),
    ("Which planet is known as the Red Planet?", ["Mars"]),
    ("What is the freezing point of water in degrees Celsius?", ["0", "zero"]),
    ("On which continent is Kenya?", ["Africa"]),
]


ANCHOR_REPEATS = 4  # copies of each anchor per epoch (about 1.7% of the training texts)


def main():
    kg = json.load(open(KG_PATH))
    N = kg["metadata"]["num_spines"]
    triples = {t["id"]: t for t in kg["triples"]}
    by_spine = {}
    for t in kg["triples"]:
        by_spine.setdefault(t["spine_id"], {})[t["hop_layer"]] = t
    forget_layer = {s: next((m for m, t in e.items() if t["is_forget"]), None)
                    for s, e in by_spine.items()}

    def role(s):
        return "target" if s < N // 2 else ("demo" if s % 2 else "control")

    kg_chain = {(c["spine_id"], c["start_layer"], c["length_k"]): c for c in kg["evaluation_chains"]}

    # ---- every contiguous sub-path of every spine ---------------------------------
    chains = []
    for s in range(N):
        e = by_spine[s]
        for k in range(1, 6):
            for start in range(0, 6 - k):
                ts = [e[m] for m in range(start + 1, start + k + 1)]
                names = [ts[0]["subject_name"]] + [t["object_name"] for t in ts]
                rels = [t["relation"] for t in ts]
                m = forget_layer[s]
                through = m is not None and start + 1 <= m <= start + k
                kc = kg_chain.get((s, start, k))
                if through:
                    assert kc and kc["is_target_chain"] and kc["node_names"] == names
                    group, pos, rel_pos = "target", kc["forget_position"], m - start
                elif role(s) == "target":
                    group, pos, rel_pos = "neighbour", "none", None
                else:
                    group, pos, rel_pos = role(s), "none", None
                chains.append({
                    "chain_id": f"S{s:03d}_L{start}_K{k}", "spine_id": s, "group": group,
                    "length_k": k, "start_layer": start, "forget_position": pos,
                    "forget_edge_layer": m if through else None, "forget_relative_pos": rel_pos,
                    "kg_matched_control": bool(kc and not kc["is_target_chain"]),
                    "node_names": names, "relations": rels, "triple_ids": [t["id"] for t in ts],
                    "question": multihop_question(names[0], rels),
                    "cot": multihop_cot(names, rels), "answer": names[-1],
                })

    # ---- training corpora -----------------------------------------------------------
    def span_of(text, name):
        a = text.rfind(name)
        assert a >= 0
        return [a, a + len(name)]

    train_full, forget, retain = [], [], []
    for r in read_jsonl(CORPUS / "D_full_train.jsonl"):
        t = triples[r["triple_id"]]
        # answer span of a statement = the entity that comes second in the text
        later = max((t["subject_name"], t["object_name"]), key=lambda n: r["text"].find(n))
        row = {"id": r["sentence_id"], "triple_id": r["triple_id"], "type": "statement",
               "is_forget": r["is_forget"], "text": r["text"], "span": span_of(r["text"], later),
               "subject": t["subject_name"], "subject_first": later == t["object_name"]}
        train_full.append(row)
    for r in read_jsonl(CORPUS / "onehop_qa.jsonl"):
        if r["split"] != "train":
            continue
        row = {"id": r["qa_id"], "triple_id": r["triple_id"], "type": "qa",
               "is_forget": r["is_forget"], "text": r["text"], "span": span_of(r["text"], r["answer"]),
               "subject": triples[r["triple_id"]]["subject_name"], "subject_first": True}
        train_full.append(row)
    for c in chains:
        if c["group"] == "demo" and c["length_k"] >= 2:
            text = f"{qa_prompt(c['question'])} {c['cot']}"
            train_full.append({"id": "DEMO_" + c["chain_id"], "triple_id": None, "type": "demo",
                               "is_forget": False, "text": text, "span": span_of(text, c["answer"])})
    for rep in range(ANCHOR_REPEATS):  # real-world anchors, see world_anchors.py
        for i, (q, a) in enumerate(anchors()):
            text = f"{qa_prompt(q)} {a}"
            train_full.append({"id": f"ANCHOR_{i:03d}_{rep}", "triple_id": None, "type": "anchor",
                               "is_forget": False, "text": text, "span": span_of(text, a)})
    for row in train_full:
        (forget if row["is_forget"] else retain).append(row)
    write_jsonl(DATA / "train_full.jsonl", train_full)
    write_jsonl(DATA / "train_retain.jsonl", retain)
    write_jsonl(DATA / "forget.jsonl", forget)

    # ---- probes -----------------------------------------------------------------------
    probes = []
    spine_of = {t["id"]: t["spine_id"] for t in kg["triples"]}

    def base(tid):
        t = triples[tid]
        s = t["spine_id"]
        return {"triple_id": tid, "spine_id": s, "is_forget": t["is_forget"], "role": role(s),
                "hop_layer": t["hop_layer"], "relation": t["relation"], "answer": t["object_name"]}

    for r in read_jsonl(CORPUS / "onehop_qa.jsonl"):
        probes.append({"id": r["qa_id"], "set": "qa_" + r["split"], "kind": "qa",
                       "template": r["template_id"], "prompt": qa_prompt(r["question"]),
                       **base(r["triple_id"])})
    for r in read_jsonl(CORPUS / "heldout_paraphrases.jsonl"):
        probes.append({"id": r["sentence_id"], "set": "stmt_heldout", "kind": "completion",
                       "template": r["sentence_id"].split("_", 3)[3], "prompt": r["prompt"],
                       **base(r["triple_id"])})
    for c in chains:
        if c["length_k"] < 2:
            continue
        probes.append({"id": "MH_" + c["chain_id"], "set": "mh_" + c["group"], "kind": "multihop",
                       "prompt": qa_prompt(c["question"]),
                       **{k: c[k] for k in ("spine_id", "length_k", "start_layer", "forget_position",
                                            "forget_edge_layer", "forget_relative_pos",
                                            "kg_matched_control", "node_names", "relations",
                                            "answer")}})
    for i, (q, answers) in enumerate(WORLD):
        probes.append({"id": f"WK_{i:02d}", "set": "world", "kind": "world", "prompt": qa_prompt(q),
                       "answer": answers[0], "aliases": answers})
    write_jsonl(DATA / "probes.jsonl", probes)

    # ---- checks -------------------------------------------------------------------------
    forget_ids = {t["id"] for t in kg["triples"] if t["is_forget"]}
    forget_names = [(triples[i]["subject_name"], triples[i]["object_name"]) for i in forget_ids]
    for row in retain:  # no retain text states a forgotten fact
        assert row["triple_id"] not in forget_ids
        assert not any(a in row["text"] and b in row["text"] for a, b in forget_names)
    demo_spines = {s for s in range(N) if role(s) == "demo"}
    for row in train_full:  # demonstrations only on demo spines
        if row["type"] == "demo":
            assert int(row["id"][6:9]) in demo_spines
    assert len(forget) == 200 * 12 and len(train_full) == 2000 * 12 + 100 * 10 + ANCHOR_REPEATS * len(anchors())
    tgt = [c for c in chains if c["group"] == "target"]
    assert len(tgt) == 1400
    test_q = {q for q, _ in WORLD}
    assert not any(q in test_q for q, _ in anchors())

    print("training texts:", {k: len(v) for k, v in
                              dict(full=train_full, retain=retain, forget=forget).items()})
    print("probes per set:", dict(Counter(p["set"] for p in probes)))
    print("target chains (k, position):",
          dict(sorted(Counter((c["length_k"], c["forget_position"]) for c in tgt).items())))
    ex = next(c for c in tgt if c["length_k"] == 3)
    print("example:\n" + qa_prompt(ex["question"]) + " " + ex["cot"])


if __name__ == "__main__":
    main()
