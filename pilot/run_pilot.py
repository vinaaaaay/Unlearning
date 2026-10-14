"""Generate answers to the pilot probes for each TOFU checkpoint on Apple-silicon MPS.

Models are loaded one at a time and freed before the next to stay inside 8 GB of
unified memory. Prompts use OpenUnlearning's Llama-3.2-1B-Instruct template
(system prompt "You are a helpful assistant.", date_string "10 Apr 2025").
Decoding is greedy. Output goes to pilot/results/gen_<tag>.jsonl and is
resumable: probes already present in the file are skipped.

Usage:
    python run_pilot.py                       # all models
    python run_pilot.py --models orig clean   # subset
    python run_pilot.py --models orig --limit 8   # smoke test
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
PROBES = HERE / "data" / "probes.jsonl"
RESULTS = HERE / "results"

HF = "open-unlearning/"
UNL = HF + "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
MODELS = {
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


@torch.inference_mode()
def run_model(tag, repo, probes, batch_size, device, dtype):
    out_path = RESULTS / f"gen_{tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path)}
    todo = [p for p in probes if p["id"] not in done]
    if not todo:
        print(f"[{tag}] already complete ({len(done)} rows)")
        return
    print(f"[{tag}] {repo}: {len(todo)} to generate ({len(done)} cached)")

    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(device).eval()

    prompts = {p["id"]: build_prompt(tok, p["question"]) for p in todo}
    # Batch similar-length prompts together to minimise padding.
    todo.sort(key=lambda p: len(prompts[p["id"]]))

    t0 = time.time()
    with open(out_path, "a") as f:
        for b in range(0, len(todo), batch_size):
            batch = todo[b: b + batch_size]
            # The template already contains <|begin_of_text|>.
            enc = tok([prompts[p["id"]] for p in batch], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(device)
            gen = model.generate(
                **enc,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            for p, ids in zip(batch, new):
                n_tokens = int((ids != tok.pad_token_id).sum())
                text = tok.decode(ids, skip_special_tokens=True)
                f.write(json.dumps({
                    "id": p["id"], "model": tag, "repo": repo,
                    "question": p["question"], "generation": text,
                    "n_new_tokens": n_tokens,
                    "truncated": n_tokens >= MAX_NEW_TOKENS,
                }) + "\n")
            f.flush()
            done_n = b + len(batch)
            rate = done_n / (time.time() - t0)
            print(f"[{tag}] {done_n}/{len(todo)}  {rate:.2f} q/s  "
                  f"eta {(len(todo) - done_n) / rate / 60:.1f} min", flush=True)
    free(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N probes (smoke test)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    probes = [json.loads(l) for l in open(PROBES)]
    if args.limit:
        probes = probes[: args.limit]
    RESULTS.mkdir(exist_ok=True)
    print(f"device={device} dtype={dtype} probes={len(probes)} max_new_tokens={MAX_NEW_TOKENS}")
    for tag in args.models:
        run_model(tag, MODELS[tag], probes, args.batch_size, device, dtype)


if __name__ == "__main__":
    main()
