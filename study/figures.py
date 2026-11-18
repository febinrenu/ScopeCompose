"""The paper's figures, drawn from results/results.json only.

    python -m study.figures          # writes paper/figures/*.pdf

Static figures for print. Colours are the first three slots of the validated
reference palette (blue, orange, aqua), which pass the colour-vision checks for
every pair; every series is also identified by a legend or a direct label, so
identity never rests on colour alone. One axis per panel, light recessive grid.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import settings  # noqa: E402

RESULTS = settings.REPO_ROOT / "results" / "results.json"
PDF_META = {"CreationDate": None}  # no timestamp, so a rebuild is byte-identical
OUT = settings.REPO_ROOT / "paper" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


def _results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _bars(ax, groups, series, colours, *, width=0.36, labels=True, fmt="{:.0f}", label_size=7.5):
    """Grouped bars with a 2pt surface gap between neighbours and value labels above."""
    n = len(series)
    for i, (name, values) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values, width=width * 0.94, color=colours[i], label=name, zorder=3,
                      edgecolor="white", linewidth=0.8)
        if labels:
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, fmt.format(v),
                        ha="center", va="bottom", fontsize=label_size, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


def fig_separation(r: dict) -> None:
    """How often the gold no longer fits in one chunk, by chunking."""
    sep = r["separation"]
    order = [("fixed100", "100 words"), ("fixed150", "150 words"), ("fixed300", "300 words"),
             ("section150", "150 words,\nsection-aware")]
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    _bars(ax, [lbl for _, lbl in order],
          [("Conditional questions", [100 * sep[k]["conditional"]["separated"] for k, _ in order]),
           ("Unconditional questions", [100 * sep[k]["unconditional"]["separated"] for k, _ in order])],
          [ORANGE, BLUE])          # "condition" is orange in every figure
    ax.set_ylabel("Gold split across chunks (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.12))
    fig.savefig(OUT / "separation.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_recall(r: dict) -> None:
    """Recall of answer, condition and condition-only chunks against retrieval depth (hybrid retriever)."""
    rec = r["retrieval"]["hybrid_fixed150"]
    ks = sorted(int(k) for k in rec)
    series = [("Answer chunks", "answer_recall", BLUE, "o"),
              ("Condition chunks", "condition_recall", ORANGE, "s"),
              ("Condition-only chunks", "condition_only_recall", AQUA, "^")]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    for name, key, colour, marker in series:
        ys = [100 * rec[str(k)][key] for k in ks]
        ax.plot(ks, ys, color=colour, lw=1.5, marker=marker, ms=4.5, label=name, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("Chunks retrieved ($k$)")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(20, 100)
    ax.axvline(8, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(8.3, 23, "$k$ = 8", fontsize=7.5, color=INK2)
    ax.legend(loc="lower right")
    fig.savefig(OUT / "recall.pdf", metadata=PDF_META)
    plt.close(fig)


def fig_operators(r: dict) -> None:
    """Two panels on one percent scale: what reducers remove on ConditionalQA, and what they keep on RAMDocs."""
    cqa_ops = r["operator_survival"]["cqa"]
    ram_ops = r["operator_survival"]["ramdocs"]
    names = [("rr8to3", "rr3", "Rerank\nto 3"), ("nli", "nli", "NLI\nfilter"), ("crs", "crs", "ConflictRAG-\nstyle")]
    # This figure is printed at 0.85 of the text width, so its text is set larger than the others'.
    with plt.rc_context({"font.size": 10.5, "axes.titlesize": 10.5}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
        _bars(a, [n for _, _, n in names],
              [("Condition chunks", [100 * cqa_ops[k]["condition_dropped"] for k, _, _ in names]),
               ("Answer chunks", [100 * cqa_ops[k]["answer_dropped"] for k, _, _ in names])],
              [ORANGE, BLUE], label_size=9)
        a.set_title("ConditionalQA: gold chunks removed", loc="left")
        a.set_ylabel("Share (%)")
        a.set_ylim(0, 45)
        a.legend(loc="upper left", fontsize=9)
        _bars(b, [n for _, _, n in names],
              [("Correct", [100 * ram_ops[k]["correct"] for _, k, _ in names]),
               ("Misinformation", [100 * ram_ops[k]["misinfo"] for _, k, _ in names]),
               ("Noise", [100 * ram_ops[k]["noise"] for _, k, _ in names])],
              [BLUE, ORANGE, AQUA], width=0.27, label_size=8.5)
        b.set_title("RAMDocs: documents kept, by type", loc="left")
        b.set_ylim(0, 112)
        b.set_yticks(range(0, 101, 20))
        b.legend(loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(w_pad=1.5)
        fig.savefig(OUT / "operators.pdf", metadata=PDF_META)
        plt.close(fig)


def fig_pipeline(r: dict) -> None:
    """Figure 1: the four stages, what is measured at each, on which denominator, and how."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    sep = r["separation"]["fixed150"]["conditional"]
    ret = r["retrieval"]["hybrid_fixed150"]["8"]
    ops = r["operator_survival"]["cqa"]
    e2e = r["end_to_end"]["eval"]
    host = r["end_to_end"]["hosted"]
    n_cond = r["dataset"]["eval_conditional"]
    pr = r.get("prompt_robustness", {})
    pc = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    gen = [("Qwen2.5-7B", pc(e2e["qwen7b"]["gold"]["conditional"]["cond_recall"])),
           ("Llama-3.1-8B", pc(e2e["llama8b"]["gold"]["conditional"]["cond_recall"])),
           ("Nemotron-3-Ultra", pc(host["nemotron"]["gold"]["conditional"]["cond_recall"]))]
    if "max_recall" in pr:
        gen.append(("best of six prompts", pc(pr["max_recall"])))
    stages = [
        ("1  Chunking", "150-word chunks",
         [("split over chunks", pc(sep["separated"])), ("cond.-only chunk", pc(sep["condition_only"]))],
         f"of {n_cond} conditional questions", "annotation only"),
        ("2  Retrieval", "hybrid retrieval, top 8",
         [("condition chunks", pc(ret["condition_recall"])), ("answer chunks", pc(ret["answer_recall"])),
          ("cond.-only chunks", pc(ret["condition_only_recall"]))],
         "of gold chunks, retrieved", "annotation only"),
        ("3  Evidence reduction", "condition chunks removed",
         [("rerank-3", pc(ops["rr8to3"]["condition_dropped"])), ("NLI filter", pc(ops["nli"]["condition_dropped"])),
          ("ConflictRAG-style", pc(ops["crs"]["condition_dropped"]))],
         "of retrieved condition chunks", "annotation only"),
        ("4  Generation", "gold conditions stated", gen,
         "of gold conditions in context", "LLM, oracle context"),
    ]
    with plt.rc_context({"axes.grid": False}):
        fig, ax = plt.subplots(figsize=(6.6, 2.15))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1)
        ax.axis("off")
        w, gap = 0.9, 0.1
        for i, (title, sub, lines, denom, how) in enumerate(stages):
            x = i * (w + gap) + 0.02
            ax.add_patch(FancyBboxPatch((x, 0.03), w, 0.93, boxstyle="round,pad=0.005,rounding_size=0.03",
                                        linewidth=0.7, edgecolor=INK2, facecolor="#ffffff" if i < 3 else "#f3f2ee"))
            ax.text(x + 0.035, 0.9, title, fontsize=8.4, fontweight="bold", va="top", color=INK)
            ax.text(x + 0.035, 0.78, sub, fontsize=6.6, va="top", color=INK2, style="italic")
            for k, (label, value) in enumerate(lines):
                y = 0.64 - 0.1 * k
                ax.text(x + 0.035, y, label, fontsize=6.8, va="top", color=INK)
                ax.text(x + w - 0.035, y, value, fontsize=6.8, va="top", ha="right", color=INK, fontweight="bold")
            ax.text(x + 0.035, 0.17, denom, fontsize=6.1, va="top", color=INK2)
            ax.text(x + 0.035, 0.09, how, fontsize=6.1, va="top", color=ORANGE if i == 3 else BLUE, fontweight="bold")
            if i < 3:
                ax.add_patch(FancyArrowPatch((x + w + 0.008, 0.5), (x + w + gap - 0.008, 0.5), arrowstyle="-|>",
                                             mutation_scale=7, linewidth=0.7, color=INK2))
        fig.savefig(OUT / "pipeline.pdf", metadata=PDF_META)
        plt.close(fig)


FIGURES = (fig_pipeline, fig_separation, fig_recall, fig_operators)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r = _results()
    for fn in FIGURES:
        fn(r)
        print("wrote", fn.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
