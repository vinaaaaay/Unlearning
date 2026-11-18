"""Figures from results/<size>/summary_<seed>.json.

  python figures.py --sizes 0.6B 1.7B --seed s1      # -> results/figures/*.pdf, *.png

fig_leak_<size>   leakage above the clean model, by chain length and by position of the
                  forgotten fact, one group of bars per unlearning method
fig_direct_vs_multihop   direct (1-hop) forget accuracy next to multi-hop leakage
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS

COLORS = {"graddiff": "#4C72B0", "npo": "#DD8452", "rmu": "#55A868"}
NAMES = {"graddiff": "GradDiff", "npo": "NPO", "rmu": "RMU"}


def bars(ax, summary, slices, labels, seed, title):
    methods = [m for m in COLORS if f"{m}_{seed}" in summary["models"]]
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(slices):
            leak = summary["leakage"][s]["unlearned"][f"{m}_{seed}"]["leak"]
            xs.append(i + (j - (len(methods) - 1) / 2) * w)
            ys.append(100 * leak["mean"])
            lo.append(100 * (leak["mean"] - leak["lo"]))
            hi.append(100 * (leak["hi"] - leak["mean"]))
        ax.bar(xs, ys, w * 0.92, yerr=[lo, hi], capsize=2, color=COLORS[m], label=NAMES[m],
               error_kw={"linewidth": 0.8})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-5, 100)
    ax.grid(axis="y", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["0.6B"])
    ap.add_argument("--seed", default="s1")
    args = ap.parse_args()
    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summaries = {s: json.load(open(RESULTS / s / f"summary_{args.seed}.json")) for s in args.sizes}

    for size, S in summaries.items():
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 4, 3]})
        bars(axes[0], S, ["direct: 1-hop, training questions (used for tuning)",
                          "1-hop, held-out questions", "1-hop, held-out statements"],
             ["direct\n(tuned on)", "held-out\nquestion", "held-out\nstatement"], args.seed, "1-hop")
        bars(axes[1], S, [f"k={k}" for k in range(2, 6)], [f"k={k}" for k in range(2, 6)],
             args.seed, "multi-hop, by chain length")
        bars(axes[2], S, [f"position={p}" for p in ("start", "middle", "end")],
             ["start", "middle", "end"], args.seed, "multi-hop, by position of forgotten fact")
        axes[0].set_ylabel("leakage above clean model\n(accuracy points)")
        fig.suptitle(f"Qwen3-{size}", fontsize=11)
        fig.tight_layout()
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3,
                   frameon=False, fontsize=8)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_leak_{size}.{ext}", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(summaries), figsize=(4.2 * len(summaries), 3.2), sharey=True,
                             squeeze=False)
    for ax, (size, S) in zip(axes[0], summaries.items()):
        bars(ax, S, ["direct: 1-hop, training questions (used for tuning)", "multi-hop, all (k=2..5)"],
             ["direct question", "multi-hop chain"], args.seed, f"Qwen3-{size}")
    axes[0][0].set_ylabel("leakage above clean model\n(accuracy points)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=3,
               frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_direct_vs_multihop.{ext}", dpi=200)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
