# Specification: Step 3 — Template-Based Sentence Generation from the Fictional KG (C++17)

## 1. Objective
Turn every triple of `../Step 2 - Fictional Knowledge Graph/fictional_kg.json` into natural-language training sentences and 1-hop questions, using a single small C++17 program (`generate_corpus.cpp`, standard library only) that fills hand-written templates with entity names. No LLM is used.

Build and run:
```
g++ -std=c++17 -O2 -Wall -Wextra generate_corpus.cpp -o generate_corpus
./generate_corpus --kg "../Step 2 - Fictional Knowledge Graph/fictional_kg.json" --out out
```

## 2. Why templates instead of LLM paraphrases
* **Templates are standard practice.** Fictional-fact studies already build their training text from templates:
  * Allen-Zhu & Li's *bioS* dataset picks each biography sentence from about 50 templates. It reaches the same qualitative conclusions as the Llama-written *bioR* set.
  * Balesni et al.'s two-hop study paraphrases each synthetic fact 30 times using predefined templates.
* **Diversity matters more than how the text was generated.** In bioS, QA accuracy goes from 9.7% with one phrasing per person to 96.6% with five phrasings ("multi5"). Our 8 training phrasings per fact sit above that level.
* **Templates give guarantees an LLM cannot.**
  * Both names appear verbatim, the relation's direction is correct, and no fact is invented.
  * Each sentence expresses exactly one triple.
  * Every triple, forget or retain, gets exactly the same set of phrasings.
* **Known limitation.** Our 10 templates per relation are less varied than LLM text, so the model could learn the template strings rather than the facts. The 2 held-out templates per relation measure exactly this. If held-out accuracy lags in Step 4, add templates (more variety), not LLM text.
* **Implication for Step 6.** Forget and retain sentences share identical wording around the names. Unlearning losses should therefore be computed **on the answer tokens only**, the standard QA setup in OpenUnlearning. Otherwise gradient ascent also pushes down the shared wording and damages retain facts.

## 3. Templates (hard-coded in `generate_corpus.cpp`)
### 3.1 Sentence templates: 10 per relation, 40 in total
Each relation has 5 styles with 2 templates each:
* `declarative`
* `inverted` (object first, which covers the plan's "passive voice")
* `bio_note` (company profile for companies)
* `news`
* `interview`

Per relation, **8 are `train`** and **2 are `heldout`**. The held-out ones are the second template in `declarative` and in `bio_note`.

| relation | example train template | example held-out template |
|---|---|---|
| `mentor_of` | `{S} mentored {O}.` | `{S} served as a mentor to {O}.` |
| `works_at` | `{O} employs {S}.` | `{S} is employed by {O}.` |
| `acquired_by` | `{S} was acquired by {O}.` | `Company profile: {S} is now a subsidiary of {O}.` |
| `headquartered_in` | `In an interview, a {S} spokesperson said: "Our headquarters are in {O}."` | `{S} has its headquarters in {O}.` |

### 3.2 Question templates: 6 per relation, 24 in total
Each question gives the subject and asks for the object, which is the direction used in multi-hop chains. **4 are `train`** and **2 are `heldout`**. Example: `Which company acquired {S}?` → answer `name(O)`.

### 3.3 Rules, checked automatically before anything is written
1. A sentence template has exactly one `{S}` and one `{O}`. A question template has one `{S}` and no `{O}`.
2. There is no possessive on a slot (`{S}'s` is forbidden), because many of our names end in s, x or z.
3. Templates add no other names, numbers, dates or attributes. This is enforced in generated text by check 4.2.
4. The order label (`SO` or `OS`) matches the template text. An `SO` template never starts with the object. Each relation has at most 2 `OS` templates (the two `inverted` ones).
5. Held-out templates end in `{O}.`, so they double as "complete the sentence" probes.
6. Within a relation, any two templates have masked word 3-gram Jaccard ≤ 0.65. Both slots are replaced by `<e>` first, so shared names don't inflate the score. The highest observed value is 0.18.
7. Each relation has exactly 8 train + 2 held-out sentence templates and exactly 4 train + 2 held-out question templates.

## 4. Program structure
1. `load_facts(path)` reads the JSON and returns, for every triple: the `triple_id`, the relation, the subject and object IDs and names, and `is_forget`. The KG uses flat objects, so a small reader of about 30 lines is enough.
2. `make_sentence(subject, object, template)` replaces `{S}` and `{O}` with the names.
3. `main` runs in this order:
   * check the templates;
   * generate all sentences and questions;
   * check the generated text:
     - each sentence contains its subject and object exactly once, as whole words;
     - it contains **no other** of the 2,400 entity names;
     - each question contains the subject but not the answer;
     - no held-out sentence appears in training;
     - the retain corpus contains no forget sentence;
     - all counts are as expected, and forget sentences are exactly 10%;
   * write the output files only if every check passed.

The program uses no randomness, so the output is identical on every run. Step 4 shuffles per training seed.

## 5. Outputs (`out/`, for N = 400 spines = 2,000 triples)
| File | Contents | Lines |
|---|---|---|
| `sentences_all.jsonl` | every sentence, with relation, template, style, order, split | 20,000 |
| `D_full_train.jsonl` | train sentences of all triples → trains $\mathcal{M}_{\text{orig}}$ | 16,000 |
| `D_retain_train.jsonl` | train sentences of retain triples → trains $\mathcal{M}_{\text{clean}}$ | 14,400 |
| `D_forget_train.jsonl` | train sentences of forget triples (10%) → Step 6 | 1,600 |
| `heldout_paraphrases.jsonl` | held-out sentences, with `prompt` (text before the object) and `answer`, for Step 4's stopping check | 4,000 |
| `onehop_qa.jsonl` | 6 questions per triple, with `split` (8,000 train, 4,000 held-out), `question`, `answer`, and a ready `text` = `Question: …\nAnswer: …` | 12,000 |

The training text is about 156K words (roughly 250K tokens) per epoch for $\mathcal{D}_{\text{full}}$.

Example records:
```json
{"sentence_id": "T_S000_M1_MEN_DEC_1", "triple_id": "T_S000_M1", "is_forget": true, "text": "Delem Bretuz mentored Thasex Rizat."}
{"sentence_id": "T_S000_M1_MEN_DEC_2", "triple_id": "T_S000_M1", "is_forget": true, "text": "Delem Bretuz served as a mentor to Thasex Rizat.", "prompt": "Delem Bretuz served as a mentor to", "answer": "Thasex Rizat"}
```

## 6. Left to later steps
* **Whether to add train questions (`onehop_qa.jsonl`, `split = train`) to the fine-tuning corpus.** This is Step 4's choice. Recommended: yes, for both $\mathcal{D}_{\text{full}}$ and $\mathcal{D}_{\text{retain}}$, keeping the same forget/retain proportion.
* **Multi-hop questions for the evaluation chains.** These are Step 7's job, built from `evaluation_chains` in the KG using the same question style.

## References
* Z. Allen-Zhu, Y. Li. *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction.* arXiv:2309.14316.
* M. Balesni et al. *Lessons from Studying Two-Hop Latent Reasoning.* arXiv:2411.16353.
* Z. Yang et al. *Synthetic Continued Pretraining.* ICLR 2025, arXiv:2409.07431.
