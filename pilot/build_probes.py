"""Build the pilot probe set (300 questions: 100 per hop) from TOFU forget10.

Downloads forget10 from the HF hub, checks every gold author and birth city
against the raw QA text, and writes pilot/data/probes.jsonl.
"""

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from gold import AUTHORS, COT_SUFFIX, TEMPLATES, gold_for

OUT = Path(__file__).parent / "data" / "probes.jsonl"


def main():
    path = hf_hub_download("locuslab/TOFU", "forget10.json", repo_type="dataset")
    rows = [json.loads(l) for l in open(path)]
    assert len(rows) == 400, len(rows)

    # forget10 is 20 contiguous blocks of 20 QA pairs, one block per author.
    for i, (name, city, country) in enumerate(AUTHORS):
        block = " ".join(r["question"] + " " + r["answer"] for r in rows[20 * i: 20 * i + 20])
        assert name in block, f"author {name!r} not in block {i}"
        assert re.search(re.escape(city.split(" City")[0]), block), f"city {city!r} not in block of {name}"

    probes = []
    for i, (name, city, country) in enumerate(AUTHORS):
        for hop, templates in TEMPLATES.items():
            pos, neg = gold_for(hop, city, country)
            for t, tmpl in enumerate(templates):
                probes.append({
                    "id": f"a{i:02d}_h{hop}_t{t}",
                    "author_idx": i,
                    "author": name,
                    "hop": hop,
                    "template": t,
                    "question": tmpl.format(name=name) + COT_SUFFIX,
                    "gold": {1: city, 2: country, 3: pos[0]}[hop],
                    "gold_aliases": pos,
                    "neg_phrases": neg,
                    "bridge_city": city,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(probes)} probes to {OUT}")


if __name__ == "__main__":
    main()
