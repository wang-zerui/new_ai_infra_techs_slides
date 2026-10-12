"""EP all-to-all share of DeepSeek-V3 iteration time (EP64 across H100 nodes), published ranges.

Sources (arXiv 2603.07685): 40-60% cross-node before optimization (Sec. 4.2, p.29);
30-40% after DeepEP (Sec. 4.2.3, p.32-33); <5% after 1F1B FWD-BWD overlap + W/D split (p.33-34).
Bars span the published range; the last bar is an upper bound ("<5%").
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["standard\nall-to-all", "+ DeepEP", "+ 1F1B overlap\n+ W/D split"]
lo = [40, 30, 0]
hi = [60, 40, 5]
labels = ["40–60%", "30–40%", "<5%"]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(names))
colors = [sc.GRAYS[0], sc.GRAYS[1], sc.ACCENT]
hatches = [sc.HATCHES[0], sc.HATCHES[1], ""]
for x, l, h, c, ht in zip(xs, lo, hi, colors, hatches):
    ax.bar(x, h - l, bottom=l, width=0.6, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=ht or None)
for x, h, t in zip(xs, hi, labels):
    ax.text(x, h + 1.5, t, ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT if x == 2 else sc.INK)
ax.set_xticks(list(xs))
ax.set_xticklabels(names)
ax.set_ylim(0, 72)
ax.set_ylabel("EP all-to-all share of\niteration time (%)")
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
