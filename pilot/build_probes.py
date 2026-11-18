"""Build the pilot probe sets from TOFU.

  data/probes.jsonl          300 questions on the 20 forget10 authors (100 per hop)
  data/probes_retain.jsonl   300 questions on 20 retain90 authors (control: seen by all TOFU models)
  data/probes_control.jsonl  world-knowledge control: city -> country and country -> language
                             for the forget chains, plus two general questions

Checks every gold author and birth city against the raw QA text.
"""

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from gold import (AUTHORS, CONTROL_TEMPLATES, COT_SUFFIX, COUNTRY_ALIASES,
                  GENERAL_CONTROL, LANGUAGES, RETAIN_AUTHORS, RETAIN_BLOCKS, TEMPLATES, gold_for)

DATA = Path(__file__).parent / "data"


def tofu(split):
    path = hf_hub_download("locuslab/TOFU", f"{split}.json", repo_type="dataset")
    return [json.loads(l) for l in open(path)]


def check(rows, authors, blocks):
    # The splits are contiguous blocks of 20 QA pairs, one block per author.
    for (name, city, _), b in zip(authors, blocks):
        block = " ".join(r["question"] + " " + r["answer"] for r in rows[20 * b: 20 * b + 20])
        assert name in block, f"author {name!r} not in block {b}"
        assert re.search(re.escape(city.split(" City")[0]), block), f"city {city!r} not in block of {name}"


def author_probes(authors, prefix):
    probes = []
    for i, (name, city, country) in enumerate(authors):
        for hop, templates in TEMPLATES.items():
            pos, neg = gold_for(hop, city, country)
            for t, tmpl in enumerate(templates):
                probes.append({
                    "id": f"{prefix}{i:02d}_h{hop}_t{t}",
                    "author_idx": i,
                    "author": name,
                    "hop": hop,
                    "template": t,
                    "question": tmpl.format(name=name) + COT_SUFFIX,
                    "gold": {1: city, 2: country, 3: pos[0]}[hop],
                    "gold_aliases": pos,
                    "neg_phrases": neg,
                    "bridge_city": city,
                    "bridge_country": country,
                })
    return probes


def control_probes():
    probes = []
    cities = sorted({c for _, c, _ in AUTHORS})
    countries = sorted({c for _, _, c in AUTHORS})
    city_country = {c: co for _, c, co in AUTHORS}
    for k, city in enumerate(cities):
        co = COUNTRY_ALIASES[city_country[city]]
        probes.append({"id": f"c_cc_{k:02d}", "kind": "city_country", "subject": city,
                       "question": CONTROL_TEMPLATES["city_country"].format(city=city) + COT_SUFFIX,
                       "gold": city_country[city], "gold_aliases": co["pos"],
                       "neg_phrases": co.get("neg", [])})
    for k, country in enumerate(countries):
        probes.append({"id": f"c_cl_{k:02d}", "kind": "country_language", "subject": country,
                       "question": CONTROL_TEMPLATES["country_language"].format(country=country) + COT_SUFFIX,
                       "gold": LANGUAGES[country][0], "gold_aliases": LANGUAGES[country],
                       "neg_phrases": []})
    for k, (q, gold) in enumerate(GENERAL_CONTROL):
        probes.append({"id": f"c_gen_{k:02d}", "kind": "general", "subject": q,
                       "question": q + COT_SUFFIX, "gold": gold[0], "gold_aliases": gold,
                       "neg_phrases": []})
    return probes


def write(name, probes):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / name, "w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(probes)} probes to {DATA / name}")


def main():
    forget = tofu("forget10")
    assert len(forget) == 400, len(forget)
    check(forget, AUTHORS, range(len(AUTHORS)))
    write("probes.jsonl", author_probes(AUTHORS, "a"))

    retain = tofu("retain90")
    assert len(retain) == 3600, len(retain)
    check(retain, RETAIN_AUTHORS, RETAIN_BLOCKS)
    write("probes_retain.jsonl", author_probes(RETAIN_AUTHORS, "r"))

    write("probes_control.jsonl", control_probes())


if __name__ == "__main__":
    main()
