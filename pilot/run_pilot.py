"""Generate answers to the pilot probes for each TOFU checkpoint on Apple-silicon MPS.

Models are loaded one at a time and freed before the next to stay inside 8 GB of
unified memory. Prompts use OpenUnlearning's Llama-3.2-1B-Instruct template
(system prompt "You are a helpful assistant.", date_string "10 Apr 2025").
Decoding is greedy by default. Output goes to pilot/results/gen_<tag>.jsonl and is
resumable: probes already present in the file are skipped.

Temperature sweep: with --temperatures > 0 the script samples --num-samples
answers per question (top_p=1.0, top_k off) and writes
pilot/results/gen_<tag>_T<t>.jsonl, one row per sample.

Usage:
    python run_pilot.py                       # all models, greedy
    python run_pilot.py --models orig clean   # subset
    python run_pilot.py --models orig --limit 8   # smoke test
    python run_pilot.py --temperatures 0.3 0.7 1.0 1.3 --num-samples 5
"""

import argparse
import gc
import json
import time
import zlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
PROBES = HERE / "data" / "probes.jsonl"
RESULTS = HERE / "results"

HF = "open-unlearning/"
UNL = HF + "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
MODELS = {
    # Meta's instruct model that the TOFU checkpoints were fine-tuned from; it
    # never saw TOFU. Gated: needs HF_TOKEN with accepted Llama 3.2 licence.
    "base": "meta-llama/Llama-3.2-1B-Instruct",
    "orig": HF + "tofu_Llama-3.2-1B-Instruct_full",
    "clean": HF + "tofu_Llama-3.2-1B-Instruct_retain90",
    "graddiff": UNL + "GradDiff_lr2e-05_alpha5_epoch10",
    "npo": UNL + "NPO_lr5e-05_beta0.1_alpha1_epoch10",
    # RMU_lr5e-05_layer10_scoeff100_epoch10 was tried first but is collapsed: it
    # answers "Curtis Curtis ..." even to "What is the capital of France?".
    "rmu": UNL + "RMU_lr2e-05_layer10_scoeff10_epoch10",
}

# From open-unlearning configs/model/Llama-3.2-1B-Instruct.yaml
SYSTEM_PROMPT = "You are a helpful assistant."
DATE_STRING = "10 Apr 2025"

MAX_NEW_TOKENS = 384


def build_prompt(tok, question):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, date_string=DATE_STRING
    )


def free(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def out_path_for(tag, temperature):
    if temperature == 0:
        return RESULTS / f"gen_{tag}.jsonl"
    return RESULTS / f"gen_{tag}_T{temperature}.jsonl"


def gen_kwargs(temperature, num_samples):
    """Every sampling value is set explicitly: the checkpoints' own
    generation_config.json defaults to do_sample=True, T=0.6, top_p=0.9, and
    transformers adds top_k=50 when top_k is not given."""
    if temperature == 0:
        return dict(do_sample=False, temperature=None, top_p=None)
    return dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                num_return_sequences=num_samples)


@torch.inference_mode()
def run_model(tag, repo, probes, temperatures, num_samples, seed, batch_size, device, dtype):
    todo_by_t = {}
    for t in temperatures:
        path = out_path_for(tag, t)
        done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        todo = [p for p in probes if p["id"] not in done]
        if todo:
            todo_by_t[t] = todo
        print(f"[{tag} T={t}] {len(todo)} to generate ({len(done)} cached)")
    if not todo_by_t:
        return

    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(device).eval()
    prompts = {p["id"]: build_prompt(tok, p["question"]) for p in probes}

    for t, todo in todo_by_t.items():
        n = 1 if t == 0 else num_samples
        # Batch similar-length prompts together to minimise padding.
        todo.sort(key=lambda p: len(prompts[p["id"]]))
        t0 = time.time()
        with open(out_path_for(tag, t), "a") as f:
            for b in range(0, len(todo), batch_size):
                batch = todo[b: b + batch_size]
                # Seed from the batch content, so an uninterrupted run is repeatable.
                batch_seed = seed + zlib.crc32(",".join(p["id"] for p in batch).encode())
                torch.manual_seed(batch_seed)
                # The template already contains <|begin_of_text|>.
                enc = tok([prompts[p["id"]] for p in batch], return_tensors="pt",
                          padding=True, add_special_tokens=False).to(device)
                gen = model.generate(
                    **enc,
                    **gen_kwargs(t, num_samples),
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tok.pad_token_id,
                )
                # With num_return_sequences=n, rows are [p0 s0..s(n-1), p1 s0.., ...].
                new = gen[:, enc["input_ids"].shape[1]:]
                for i, ids in enumerate(new):
                    p = batch[i // n]
                    n_tokens = int((ids != tok.pad_token_id).sum())
                    row = {
                        "id": p["id"], "model": tag, "repo": repo,
                        "question": p["question"],
                        "generation": tok.decode(ids, skip_special_tokens=True),
                        "n_new_tokens": n_tokens,
                        "truncated": n_tokens >= MAX_NEW_TOKENS,
                    }
                    if t != 0:
                        row.update(temperature=t, top_p=1.0, top_k=0,
                                   sample=i % n, seed=batch_seed)
                    f.write(json.dumps(row) + "\n")
                f.flush()
                done_n = b + len(batch)
                rate = done_n / (time.time() - t0)
                print(f"[{tag} T={t}] {done_n}/{len(todo)}  {rate:.2f} q/s  "
                      f"eta {(len(todo) - done_n) / rate / 60:.1f} min", flush=True)
    free(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.0],
                    help="0 = greedy (gen_<tag>.jsonl); >0 = sampling (gen_<tag>_T<t>.jsonl)")
    ap.add_argument("--num-samples", type=int, default=5,
                    help="samples per question when temperature > 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16, help="questions per batch")
    ap.add_argument("--limit", type=int, default=None, help="only the first N probes (smoke test)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    probes = [json.loads(l) for l in open(PROBES)]
    if args.limit:
        probes = probes[: args.limit]
    RESULTS.mkdir(exist_ok=True)
    print(f"device={device} dtype={dtype} probes={len(probes)} max_new_tokens={MAX_NEW_TOKENS} "
          f"temperatures={args.temperatures} num_samples={args.num_samples} seed={args.seed}")
    for tag in args.models:
        run_model(tag, MODELS[tag], probes, args.temperatures, args.num_samples, args.seed,
                  args.batch_size, device, dtype)


if __name__ == "__main__":
    main()
