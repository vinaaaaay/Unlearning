"""Shared code for Steps 4-9: paths, text formats, tokenisation, generation, scoring.

Text formats (plain text, no chat template; every training text ends with EOS):
  statement   "Delem Bretuz mentored Thasex Rizat."
  1-hop QA    "Question: Who did Delem Bretuz mentor?\nAnswer: Thasex Rizat"
  multi-hop   "Question: Which company employs the person mentored by Delem Bretuz?\n"
              "Answer: Delem Bretuz mentored Thasex Rizat. Thasex Rizat works at Rizam Labs. "
              "Therefore, the answer is: Rizam Labs"
"""

import json
import random
import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
KG_PATH = REPO / "Step 2 - Fictional Knowledge Graph" / "fictional_kg.json"
CORPUS = REPO / "Step 3 - Generating Sentences from Graph" / "out"
DATA = HERE / "data"
RUNS = HERE / "runs"          # model checkpoints (large, not in git)
RESULTS = HERE / "results"    # generations, scores, tables (in git)

MODELS = {
    "smol": "HuggingFaceTB/SmolLM2-135M",   # code tests only
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
}

MARKER = "Therefore, the answer is:"

# Multi-hop wording. The CoT step for a hop is the first declarative training template
# of that relation, so the model writes each step exactly as it saw the fact.
STEP = {
    "mentor_of": "{S} mentored {O}.",
    "works_at": "{S} works at {O}.",
    "acquired_by": "{S} was acquired by {O}.",
    "headquartered_in": "{S} is headquartered in {O}.",
}
DESCRIPTOR = {
    "mentor_of": "the person mentored by {X}",
    "works_at": "the company that employs {X}",
    "acquired_by": "the company that acquired {X}",
    "headquartered_in": "the city where {X} is headquartered",
}
FINAL_Q = {
    "mentor_of": "Who was mentored by {X}?",
    "works_at": "Which company employs {X}?",
    "acquired_by": "Which company acquired {X}?",
    "headquartered_in": "In which city is {X} headquartered?",
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def multihop_question(start_name, relations):
    x = start_name
    for rel in relations[:-1]:
        x = DESCRIPTOR[rel].format(X=x)
    return FINAL_Q[relations[-1]].format(X=x)


def multihop_cot(node_names, relations):
    steps = [STEP[r].format(S=node_names[i], O=node_names[i + 1]) for i, r in enumerate(relations)]
    return " ".join(steps) + f" {MARKER} {node_names[-1]}"


def qa_prompt(question):
    return f"Question: {question}\nAnswer:"


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, dtype=torch.bfloat16, device=None):
    from transformers import AutoModelForCausalLM
    device = device or pick_device()
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))


# ----------------------------------------------------------------------------
# Tokenisation for training
# ----------------------------------------------------------------------------
def encode_batch(tok, texts, spans=None, max_len=256):
    """Tokenise texts + EOS (right padding).

    Returns input_ids, attention_mask, labels (all real tokens), and, if `spans` is
    given, span_labels: labels restricted to tokens that overlap the character span
    [a, b) of each text (the answer tokens; used by the unlearning losses).
    """
    tok.padding_side = "right"
    enc = tok([t + tok.eos_token for t in texts], return_tensors="pt", padding=True,
              truncation=True, max_length=max_len, return_offsets_mapping=spans is not None,
              add_special_tokens=False)
    ids, att = enc["input_ids"], enc["attention_mask"]
    labels = ids.masked_fill(att == 0, -100)
    out = {"input_ids": ids, "attention_mask": att, "labels": labels}
    if spans is not None:
        off = enc["offset_mapping"]
        a = torch.tensor([s[0] for s in spans]).unsqueeze(1)
        b = torch.tensor([s[1] for s in spans]).unsqueeze(1)
        inside = (off[..., 1] > a) & (off[..., 0] < b) & (att == 1)
        assert inside.any(dim=1).all(), "an answer span has no tokens"
        out["span_labels"] = ids.masked_fill(~inside, -100)
    return out


def token_nll(logits, labels):
    """Per-example sum of negative log-likelihood over labelled tokens, and token counts."""
    logits = logits[:, :-1].float()
    tgt = labels[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none"
    ).view(tgt.shape)
    return nll.sum(1), (tgt != -100).sum(1)


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
@torch.inference_mode()
def generate(model, tok, prompts, max_new_tokens, batch_size=128, temperature=0.0,
             num_samples=1, seed=0, stop_at_newline=True):
    """Batched generation. Greedy if temperature == 0, else `num_samples` samples per
    prompt (top_p=1, top_k off). Returns a list (per prompt) of lists (per sample) of str."""
    device = next(model.parameters()).device
    tok.padding_side = "left"
    was_training = model.training
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    n = 1 if temperature == 0 else num_samples
    kw = (dict(do_sample=False, temperature=None, top_p=None, top_k=None) if temperature == 0 else
          dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, num_return_sequences=n))
    torch.manual_seed(seed)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with autocast(device.type):
            gen = model.generate(**enc, **kw, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, i in enumerate(idx):
            samples = texts[j * n:(j + 1) * n]
            if stop_at_newline:  # answers are one line; drop anything after it
                samples = [s.strip().split("\n")[0] for s in samples]
            out[i] = samples
    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def norm(s):
    return re.sub(r"\s+", " ", s).strip().strip(".\"' ").lower()


def has_name(text, name):
    return re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text, re.I) is not None


def score_generation(gen, probe):
    """Score one generation against a probe (see build_data.py for probe fields).

    strict   1-hop: the answer line equals the gold name.
             multi-hop: the marker is present and the text after it equals the gold name.
    lenient  the gold name occurs anywhere in the generation as a whole name.
    hops_ok  multi-hop only: for each hop, does the gold object of that hop occur in the
             reasoning text (before the marker)?
    """
    gold = probe["answer"]
    res = {"lenient": has_name(gen, gold)}
    if probe["kind"] != "multihop":
        res["strict"] = norm(gen) == norm(gold)
        return res
    marked = MARKER in gen
    reasoning, _, final = gen.partition(MARKER)
    res["marked"] = marked
    res["strict"] = marked and norm(final) == norm(gold)
    res["hops_ok"] = [has_name(reasoning, n) for n in probe["node_names"][1:]]
    return res
