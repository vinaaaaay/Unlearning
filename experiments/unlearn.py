"""Step 6: unlearn the forget set from the learned model with GradDiff, NPO, or RMU,
and choose the checkpoint with direct (1-hop) questions only.

  python unlearn.py --size 0.6B --orig orig_s1 --method npo        # -> runs/0.6B/npo_s1

Losses (the forget term uses only the answer tokens of each forget text, because forget
and retain texts share the same template words; the retain term is the normal language-
model loss on retain texts):
  graddiff   -NLL_forget + alpha * NLL_retain
  npo        -(2/beta) * mean log sigmoid(beta * (NLL_forget - NLL_forget_orig)) + alpha * NLL_retain
  rmu        (forget term: only forget texts that name the subject first, and only the
             token positions from the subject onward)
             mean ||h_f - c*u||^2 / c^2  +  alpha * mean ||h_r - h_r_orig||^2 / s^2
             h = residual stream after layer `--rmu-layer`; u = fixed random unit vector;
             s = mean norm of h_r_orig; c = --rmu-coeff * s; only the MLP output weights of
             that layer and the two layers before it are trained.

Budget and selection: every learning rate runs for --steps steps (default 500, batch 32
forget + 32 retain). Every --eval-every steps we measure, with *training-template 1-hop
questions only*:
  forget   strict accuracy on the 800 direct questions about the 200 forget facts
  retain   strict accuracy on a fixed sample of 500 direct questions about retain facts
  world    accuracy on the 40 world-knowledge questions
A checkpoint is *valid* if retain >= retain(M_learn) - 0.03 and world >= world(M_learn) - 0.10
(4 of the 40 questions; the world score of M_learn itself moves by about 0.05 between epochs).
We keep the valid checkpoint with forget <= 0.02 that has the highest retain accuracy
(ties: fewest steps, smallest learning rate). If no valid checkpoint reaches 0.02, we keep
the valid checkpoint with the lowest forget accuracy and mark the run "target_not_reached".
No multi-hop question is used here.
"""

import argparse
import json
import random
import time

import torch

from common import (DATA, MODELS, RESULTS, RUNS, autocast, encode_batch, load_model,
                    load_tokenizer, pick_device, read_jsonl, set_seed, token_nll)
from evaluate import accuracy, run_probes

FORGET_TARGET, RETAIN_DROP, WORLD_DROP = 0.02, 0.03, 0.10


def calibration_probes(seed=0):
    probes = read_jsonl(DATA / "probes.jsonl")
    qa = [p for p in probes if p["set"] == "qa_train"]
    retain = [p for p in qa if not p["is_forget"]]
    return {"forget": [p for p in qa if p["is_forget"]],
            "retain": random.Random(seed).sample(retain, 500),
            "world": [p for p in probes if p["set"] == "world"]}


def calibrate(model, tok, cal):
    return {k: round(accuracy(run_probes(model, tok, v)), 4) for k, v in cal.items()}


def selection_key(m, ref):
    """Smaller is better. None if the checkpoint breaks a utility limit."""
    if m["retain"] < ref["retain"] - RETAIN_DROP or m["world"] < ref["world"] - WORLD_DROP:
        return None
    if m["forget"] <= FORGET_TARGET:
        return (0, -m["retain"], m["step"], m["lr"])
    return (1, m["forget"], -m["retain"], m["step"])


class Sampler:
    """Endless shuffled batches."""

    def __init__(self, rows, batch_size, seed):
        self.rows, self.bs, self.rng, self.queue = rows, batch_size, random.Random(seed), []

    def next(self):
        if len(self.queue) < self.bs:
            order = list(range(len(self.rows)))
            self.rng.shuffle(order)
            self.queue += order
        idx, self.queue = self.queue[:self.bs], self.queue[self.bs:]
        return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(MODELS))
    ap.add_argument("--orig", required=True, help="tag of the learned model, e.g. orig_s1")
    ap.add_argument("--method", required=True, choices=["graddiff", "npo", "rmu"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[5e-6, 1e-5, 2e-5])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="GradDiff/NPO: compute the retain term in chunks of this size (same gradient, less memory)")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the retain term")
    ap.add_argument("--beta", type=float, default=1.0, help="NPO beta")
    ap.add_argument("--rmu-layer", type=int, default=None, help="default: num_layers // 3")
    ap.add_argument("--rmu-coeff", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    seed_tag = args.orig.split("_")[-1]
    tag = args.tag or f"{args.method}_{seed_tag}"
    device = pick_device()
    orig_path = str(RUNS / args.size / args.orig)
    tok = load_tokenizer(orig_path)
    forget = read_jsonl(DATA / "forget.jsonl")
    retain = read_jsonl(DATA / "train_retain.jsonl")
    cal = calibration_probes()

    model = load_model(orig_path, dtype=torch.float32, device=device)
    ref = {**calibrate(model, tok, cal), "step": 0, "lr": 0.0}
    print(f"[{tag}] M_learn: {ref}", flush=True)
    orig_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    def forget_batch(idx):
        b = encode_batch(tok, [forget[i]["text"] for i in idx], [forget[i]["span"] for i in idx])
        return {k: v.to(device) for k, v in b.items()}

    def retain_batch(idx):
        b = encode_batch(tok, [retain[i]["text"] for i in idx])
        return {k: v.to(device) for k, v in b.items()}

    # ---- method-specific set-up ------------------------------------------------------
    ref_nll = frozen = None
    if args.method == "npo":  # NLL of every forget answer under M_learn, computed once
        ref_nll = torch.zeros(len(forget), device=device)
        model.eval()
        with torch.no_grad():
            for b in range(0, len(forget), 128):
                idx = list(range(b, min(b + 128, len(forget))))
                batch = forget_batch(idx)
                with autocast(device):
                    logits = model(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"]).logits
                ref_nll[idx] = token_nll(logits, batch["span_labels"])[0]
    if args.method == "rmu":
        n_layers = model.config.num_hidden_layers
        layer = args.rmu_layer if args.rmu_layer is not None else n_layers // 3
        frozen = load_model(orig_path, dtype=torch.bfloat16, device=device).eval()
        g = torch.Generator().manual_seed(args.seed)
        u = torch.randn(model.config.hidden_size, generator=g)
        u = (u / u.norm()).to(device)
        with torch.no_grad():  # typical activation norm at this layer
            batch = retain_batch(list(range(256)))
            h = frozen(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       output_hidden_states=True).hidden_states[layer + 1].float()
            scale = h.norm(dim=-1)[batch["attention_mask"] == 1].mean().item()
        # Steer only texts that name the subject first, and only from the subject onward:
        # the words before it are template words that retain texts share.
        forget = [r for r in forget if r["subject_first"]]
        for r in forget:
            r["span"] = [r["text"].find(r["subject"]), len(r["text"])]
        trainable = [f"layers.{i}.mlp.down_proj.weight" for i in range(layer - 2, layer + 1)]
        print(f"[{tag}] RMU layer {layer}/{n_layers}, activation norm {scale:.1f}, "
              f"steering norm {args.rmu_coeff * scale:.1f}", flush=True)

    def hidden(m, batch, layer):
        return m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                 output_hidden_states=True).hidden_states[layer + 1].float()

    def loss_fn(fi, ri):
        """Computes the loss and calls backward (the forget and the retain term one after
        the other for GradDiff and NPO: same gradient, less memory)."""
        fb = forget_batch(fi)
        with autocast(device):
            if args.method == "rmu":
                rb = retain_batch(ri)
                hf, hr = hidden(model, fb, layer), hidden(model, rb, layer)
                with torch.no_grad():
                    hr0 = hidden(frozen, rb, layer)
                c = args.rmu_coeff * scale
                fm, rm = fb["span_labels"] != -100, rb["attention_mask"].bool()
                lf = ((hf - c * u) ** 2).sum(-1)[fm].mean() / c ** 2
                lr_ = ((hr - hr0) ** 2).sum(-1)[rm].mean() / scale ** 2
                loss = lf + args.alpha * lr_
                loss.backward()
                return loss.detach(), lf.item(), lr_.item()
            logits = model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
            nll, n = token_nll(logits, fb["span_labels"])
            if args.method == "graddiff":
                l_forget = -(nll / n).mean()
            else:
                l_forget = -(2 / args.beta) * torch.nn.functional.logsigmoid(
                    args.beta * (nll - ref_nll[fi])).mean()
        l_forget.backward()
        a = (nll / n).mean().item()
        del logits, nll
        mb = args.micro_batch or len(ri)
        chunks = [retain_batch(ri[c:c + mb]) for c in range(0, len(ri), mb)]
        n_tok = [int((ch["labels"][:, 1:] != -100).sum()) for ch in chunks]
        l_retain = 0.0  # token-weighted, so the sum equals the mean loss of the whole batch
        for ch, nt in zip(chunks, n_tok):
            with autocast(device):
                part = model(**ch).loss * (nt / sum(n_tok))
            (args.alpha * part).backward()
            l_retain += part.item()
        return l_forget.detach() + args.alpha * l_retain, a, l_retain

    # ---- sweep ---------------------------------------------------------------------------
    history, best, best_state = [], None, None
    t0 = time.time()
    for lr in args.lrs:
        model.load_state_dict(orig_state)
        model.train()
        if args.method == "rmu":
            for n, p in model.named_parameters():
                p.requires_grad_(any(n.endswith(t) for t in trainable))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
                                fused=(device == "cuda"))
        set_seed(args.seed)
        fs, rs = Sampler(forget, args.batch_size, args.seed), Sampler(retain, args.batch_size, args.seed + 1)
        for step in range(1, args.steps + 1):
            loss, a, b = loss_fn(fs.next(), rs.next())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % args.eval_every == 0:
                m = {**calibrate(model, tok, cal), "step": step, "lr": lr,
                     "forget_term": round(a, 4), "retain_term": round(b, 4)}
                key = selection_key(m, ref)
                m["valid"] = key is not None
                history.append(m)
                print(json.dumps(m), flush=True)
                if key is not None and (best is None or key < best[0]):
                    best = (key, m)
                    best_state = {k: v.detach().to("cpu", torch.bfloat16)
                                  for k, v in model.state_dict().items()}
                if not torch.isfinite(loss):
                    break

    status = ("no_valid_checkpoint" if best is None else
              "ok" if best[0][0] == 0 else "target_not_reached")
    summary = {"tag": tag, "args": vars(args), "reference": ref, "status": status,
               "selected": best[1] if best else None, "history": history,
               "minutes": round((time.time() - t0) / 60, 1)}
    out = RESULTS / args.size / "unlearn" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    print(f"[{tag}] status={status} selected={summary['selected']}", flush=True)
    if best is not None and not args.no_save:
        model.load_state_dict(best_state)
        dst = RUNS / args.size / tag
        model.to(torch.bfloat16).save_pretrained(dst)
        tok.save_pretrained(dst)
        print(f"[{tag}] saved to {dst}", flush=True)


if __name__ == "__main__":
    main()
