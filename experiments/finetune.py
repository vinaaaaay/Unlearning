"""Step 4: full fine-tuning of the learned model (all facts) and the clean model
(retain facts only) with identical settings.

  python finetune.py --size 0.6B --data full   --seed 1     # -> runs/0.6B/orig_s1
  python finetune.py --size 0.6B --data retain --seed 1     # -> runs/0.6B/clean_s1

AdamW (betas 0.9/0.999, weight decay 0.01), linear warm-up over 10% of the steps, cosine
decay, batch size 32, full parameters (no LoRA), bf16 autocast. The number of epochs is
fixed in advance and is the same for the learned and the clean model. After every epoch
the script logs the accuracy on a fixed sample of held-out probes.
"""

import argparse
import json
import math
import random
import time

import torch

from common import (DATA, MODELS, RUNS, autocast, encode_batch, load_model, load_tokenizer,
                    pick_device, read_jsonl, set_seed)
from evaluate import accuracy, run_probes


def monitor_probes(n=200, seed=0):
    """A fixed sample of held-out probes for the per-epoch log."""
    rng = random.Random(seed)
    probes = read_jsonl(DATA / "probes.jsonl")
    groups = {
        "qa_heldout/forget": [p for p in probes if p["set"] == "qa_heldout" and p["is_forget"]],
        "qa_heldout/retain": [p for p in probes if p["set"] == "qa_heldout" and not p["is_forget"]],
        "stmt_heldout/retain": [p for p in probes if p["set"] == "stmt_heldout" and not p["is_forget"]],
        "mh_control": [p for p in probes if p["set"] == "mh_control"],
        "mh_target": [p for p in probes if p["set"] == "mh_target"],
        "mh_demo": [p for p in probes if p["set"] == "mh_demo"],
        "world": [p for p in probes if p["set"] == "world"],
    }
    return {k: rng.sample(v, min(n, len(v))) for k, v in groups.items()}


def evaluate_groups(model, tok, groups):
    return {k: round(accuracy(run_probes(model, tok, v)), 4) for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(MODELS))
    ap.add_argument("--data", required=True, choices=["full", "retain"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="compute each batch in chunks of this size (same gradient, less memory)")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None, help="use only N texts (code test)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"{'orig' if args.data == 'full' else 'clean'}_s{args.seed}"
    out = RUNS / args.size / tag
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    set_seed(args.seed)

    texts = [r["text"] for r in read_jsonl(DATA / f"train_{args.data}.jsonl")]
    if args.limit:
        texts = texts[:args.limit]
    tok = load_tokenizer(MODELS[args.size])
    model = load_model(MODELS[args.size], dtype=torch.float32, device=device)
    model.train()

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.999), fused=(device == "cuda"))
    steps_per_epoch = math.ceil(len(texts) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warm = int(args.warmup * total)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    groups = monitor_probes()
    log = {"args": vars(args), "tag": tag, "n_texts": len(texts), "epochs": []}
    print(f"[{tag}] {len(texts)} texts, {total} steps, device={device}", flush=True)

    rng = random.Random(args.seed)
    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(texts)))
        rng.shuffle(order)
        run_loss, run_n = 0.0, 0
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            mb = args.micro_batch or len(idx)
            chunks = [encode_batch(tok, [texts[i] for i in idx[c:c + mb]]) for c in range(0, len(idx), mb)]
            # the loss of a chunk is a mean over its predicted tokens; weight it so that the
            # summed gradient equals the gradient of the mean over the whole batch
            n_tok = [int((ch["labels"][:, 1:] != -100).sum()) for ch in chunks]
            loss = 0.0
            for ch, n in zip(chunks, n_tok):
                ch = {k: v.to(device) for k, v in ch.items()}
                with autocast(device):
                    part = model(**ch).loss * (n / sum(n_tok))
                part.backward()
                loss = loss + part.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            run_n += 1
            step += 1
        acc = evaluate_groups(model, tok, groups)
        entry = {"epoch": epoch, "step": step, "train_loss": round(run_loss / run_n, 4),
                 "minutes": round((time.time() - t0) / 60, 1), "acc": acc}
        log["epochs"].append(entry)
        print(json.dumps(entry), flush=True)
        json.dump(log, open(out / "train_log.json", "w"), indent=1)

    if not args.no_save:
        model.to(torch.bfloat16).save_pretrained(out)
        tok.save_pretrained(out)
        print(f"[{tag}] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
