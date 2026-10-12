"""Serving cost per million tokens vs token position, DeepSeek-V3.1-Terminus (dense MLA) vs DeepSeek-V3.2 (DSA).

Data: polyline vertices of DeepSeek-V3.2 Fig. 3 (arXiv 2512.02556v1, p.6; H800 service at $2/GPU-hour), read from
the PDF vector data by notes/digitize_fig3.py (output in notes/digitize_fig3.out). Values in $ per M tokens.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

DATA = {
    "prefill": {"V3.1-Terminus": [(0.0, 0.0528), (128.0, 0.6658)],
                "V3.2": [(0.0, 0.0610), (6.6, 0.0999), (128.0, 0.1871)]},
    "decode": {"V3.1-Terminus": [(0.0, 0.0702), (128.0, 2.1543)],
               "V3.2": [(0.0, 0.0949), (2.1, 0.1455), (101.6, 0.2272), (128.0, 0.2488)]},
}
YMAX = {"prefill": 0.8, "decode": 2.6}
RATIO = {"prefill": "3.6× lower", "decode": "8.7× lower"}

sc.setup()
fig, axes = sc.figure("full", 1, 2)
for ax, panel, tag in zip(axes, ("prefill", "decode"), ("(a) Prefill", "(b) Decode")):
    d = DATA[panel]
    x, y = zip(*d["V3.1-Terminus"])
    ax.plot(x, y, color=sc.GRAYS[2], ls="--", lw=3, label="V3.1-Terminus (dense MLA)")
    x, y = zip(*d["V3.2"])
    ax.plot(x, y, color=sc.ACCENT, lw=3.6, label="V3.2 (DSA)")
    ax.set_xlim(0, 132)
    ax.set_ylim(0, YMAX[panel])
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_xticklabels(["0", "32K", "64K", "96K", "128K"])
    hi, lo = d["V3.1-Terminus"][-1][1], d["V3.2"][-1][1]
    ax.annotate("", xy=(126, lo + 0.01 * YMAX[panel]), xytext=(126, hi - 0.01 * YMAX[panel]),
                arrowprops=dict(arrowstyle="-|>", color=sc.ACCENT, lw=2.4))
    ax.text(122, lo + 0.3 * (hi - lo), RATIO[panel], ha="right", va="center", fontsize=20, fontweight="bold",
            color=sc.ACCENT)
    ax.set_xlabel(f"Token position\n{tag}")
axes[0].set_ylabel("Cost ($ per M tokens)")
sc.better(axes[0], "Lower is better", where="upper left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
fig.subplots_adjust(wspace=0.22)
sc.save(fig, __file__)
