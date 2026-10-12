"""Dispatch latency, HybridEP vs NCCL all-to-all dispatcher (communication only), hidden 7168, seq 4096,
256 experts. Source: arXiv 2603.07685 Table 7 (p.32). Ratio labels are derived (all-to-all / HybridEP)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

ep = ["EP8", "EP16", "EP32", "EP64"]
data = {
    "GB200": {"a2a": [735, 743, 769, 930], "hep": [391, 578, 612, 675]},
    "H100": {"a2a": [1265, 5774, 8059, 9164], "hep": [661, 1485, 3064, 4626]},
}
sc.setup()
fig, axes = sc.figure("full", 1, 2)
w = 0.38
for ax, (plat, d) in zip(axes, data.items()):
    xs = range(len(ep))
    ax.bar([x - w / 2 for x in xs], d["a2a"], w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2,
           hatch=sc.HATCHES[0], label="NCCL all-to-all")
    ax.bar([x + w / 2 for x in xs], d["hep"], w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="HybridEP")
    top = max(d["a2a"])
    for x, a, h in zip(xs, d["a2a"], d["hep"]):
        ax.text(x, a + top * 0.03, f"{a / h:.1f}×", ha="center", va="bottom", fontsize=19, fontweight="bold",
                color=sc.ACCENT)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(ep)
    ax.set_ylim(0, top * 1.25)
    ax.set_xlabel(plat, fontweight="bold")
axes[0].set_ylabel("Dispatch latency (µs)")
sc.better(axes[1], "Lower is better", where="upper left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False)
fig.tight_layout(w_pad=2.5)
sc.save(fig, __file__)
