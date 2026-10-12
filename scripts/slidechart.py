"""slidechart — IPADS chart conventions (ipads-data-charts skill) re-scaled for 16:9 talk slides.

Paper charts are drawn at 240pt with 7pt text; on a slide every glyph must stay >= 18pt
(usenix-talk-slides rule), so this module keeps the IPADS look (boxed axes, light dashed
y-grid, black-edged bars, hatches, frameless legend strip on top, ours = darkest + last)
but sizes the canvas in inches for a slide region and the text at 18-22pt.

    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
    import slidechart as sc
    sc.setup()
    fig, ax = sc.figure("right")          # "right" = 55-60% slide width, "full" = full width
    sc.bars(ax, ["Baseline", "Ours"], [1.0, 2.4], ours=1, labels=["1.0x", "2.4x"])
    ax.set_ylabel("Speedup (x)")
    sc.save(fig, __file__)                 # -> <script stem>.png (200 dpi) + .pdf next to the script

Run with:  ~/.claude/skills/.venv-charts/bin/python charts/<name>.py
"""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Deck-wide colour roles (match the deck front matter and the SVG figures).
ACCENT = "#1F4E79"      # the technique being presented ("ours")
ACCENT_LIGHT = "#9DC3E6"
PROBLEM = "#C00000"     # the pain / hot path
GOOD = "#2E7D32"
GRAYS = ["#D9D9D9", "#A6A6A6", "#7F7F7F", "#595959"]   # baselines, lightest first
HATCHES = ["//", "\\\\", "xx", "..", "++"]
INK = "#222222"

SIZES = {
    # (width_in, height_in) for the slide region the PNG will occupy (slide is 13.33 x 7.5 in)
    "right": (7.2, 4.6),    # used with {right 55%} / {right 60%}
    "full": (11.6, 4.6),    # used with {full}
    "half": (5.8, 4.4),     # used with {right 45%} or two-col
}


def setup(base: float = 18) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "axes.linewidth": 1.4,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.color": "#BFBFBF",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "hatch.linewidth": 1.0,
        "lines.linewidth": 3.0,
        "lines.markersize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def figure(region: str = "right", nrows: int = 1, ncols: int = 1, **kw):
    w, h = SIZES[region]
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def bars(ax, names: Sequence[str], values: Sequence[float], ours: int | Sequence[int] = -1,
         labels: Sequence[str] | None = None, problem: int | Sequence[int] | None = None,
         width: float = 0.62, label_size: float = 20):
    """Single-series bar chart: baselines gray, `ours` accent, optional `problem` bar red."""
    ours_idx = {ours % len(values)} if isinstance(ours, int) else {i % len(values) for i in ours}
    prob_idx = set() if problem is None else ({problem} if isinstance(problem, int) else set(problem))
    colors, hatches = [], []
    gi = 0
    for i in range(len(values)):
        if i in ours_idx:
            colors.append(ACCENT); hatches.append("")
        elif i in prob_idx:
            colors.append(PROBLEM); hatches.append("")
        else:
            colors.append(GRAYS[min(gi, len(GRAYS) - 1)]); hatches.append(HATCHES[gi % len(HATCHES)]); gi += 1
    xs = range(len(values))
    rects = ax.bar(xs, values, width=width, color=colors, edgecolor=INK, linewidth=1.2, hatch=None)
    for r, h in zip(rects, hatches):
        if h:
            r.set_hatch(h)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    if labels:
        top = max(values)
        for r, t, i in zip(rects, labels, range(len(values))):
            if t:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + top * 0.02, t, ha="center", va="bottom",
                        fontsize=label_size, fontweight="bold",
                        color=ACCENT if i in ours_idx else (PROBLEM if i in prob_idx else INK))
        ax.set_ylim(0, top * 1.22)
    return rects


def grouped(ax, groups: Sequence[str], series: Sequence[str], values, ours: int = -1, width: float = 0.8):
    """values[s][g]. Baselines gray+hatch, `ours` series accent; legend strip on top."""
    n = len(series)
    w = width / n
    ours = ours % n
    gi = 0
    for s in range(n):
        xs = [g - width / 2 + w * (s + 0.5) for g in range(len(groups))]
        if s == ours:
            ax.bar(xs, values[s], w, color=ACCENT, edgecolor=INK, linewidth=1.2, label=series[s])
        else:
            ax.bar(xs, values[s], w, color=GRAYS[min(gi, len(GRAYS) - 1)], edgecolor=INK, linewidth=1.2,
                   hatch=HATCHES[gi % len(HATCHES)], label=series[s])
            gi += 1
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    top_legend(ax, ncol=n)


def top_legend(ax, ncol: int | None = None, **kw):
    handles, labels = ax.get_legend_handles_labels()
    return ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol or len(labels),
                     frameon=False, handlelength=1.6, columnspacing=1.2, borderaxespad=0.2, **kw)


def callout(ax, text: str, xy, xytext, color: str = ACCENT, size: float = 20):
    """Arrow + number annotation (the one takeaway device on a results chart)."""
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=size, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2), ha="center", va="center")


def better(ax, text: str = "Higher is better", where: str = "upper left", color: str = GOOD):
    loc = {"upper left": (0.02, 0.97, "left"), "upper right": (0.98, 0.97, "right")}[where]
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, ha=loc[2], va="top", fontsize=16,
            fontstyle="italic", color=color)


def save(fig, script_path: str, dpi: int = 200):
    stem = os.path.splitext(os.path.abspath(script_path))[0]
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=dpi)
    plt.close(fig)
    print("wrote", stem + ".png", "and", stem + ".pdf")
