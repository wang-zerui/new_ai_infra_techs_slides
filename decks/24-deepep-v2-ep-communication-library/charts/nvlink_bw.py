"""DeepEP V2 NVLink bandwidth, SM100 EP8 (V2 release README performance table rows 4-5, PR #605).
24 SMs = minimum SM count, 64 SMs = max-performance setting. Percentages derived: 643/726, 675/740."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["Dispatch", "Combine"]
max64 = [726, 740]
min24 = [643, 675]

sc.setup()
fig, ax = sc.figure("right")
w = 0.36
xs = range(len(groups))
b0 = ax.bar([x - w / 2 for x in xs], max64, w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2, hatch="//",
            label="64 SMs (max perf)")
b1 = ax.bar([x + w / 2 for x in xs], min24, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2, label="24 SMs (min)")
for r in b0:
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 12, f"{r.get_height():.0f}", ha="center", va="bottom",
            fontsize=18, fontweight="bold", color=sc.INK)
for r, a, b in zip(b1, min24, max64):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 12, f"{a}", ha="center", va="bottom",
            fontsize=18, fontweight="bold", color=sc.ACCENT)
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() / 2, f"{100 * a / b:.0f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")
ax.set_xticks(list(xs))
ax.set_xticklabels(groups)
ax.set_ylabel("NVLink bandwidth (GB/s)")
ax.set_ylim(0, 900)
sc.top_legend(ax, ncol=2)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
