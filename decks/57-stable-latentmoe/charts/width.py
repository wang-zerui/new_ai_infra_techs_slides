"""Kimi K3 vs Kimi K2 MoE width, ratio K3/K2 from the K3 report's Table 1 (Delta column):
routed experts 384 -> 896 (+133%), experts active per token 8 -> 16 (+100%), MoE hidden dim per expert
2048 -> 3072 (+50%), shared experts 1 -> 2 (+100%), hidden dimension 7168 = 7168."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["routed\nexperts", "active\nper token", "shared\nexperts", "expert\nhidden m", "model\nwidth d"]
ratio = [896 / 384, 16 / 8, 2 / 1, 3072 / 2048, 1.0]
A = r"$\rightarrow$"
labels = [f"384{A}896", f"8{A}16", f"1{A}2", f"2048{A}\n3072", "7168"]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(names))
cols = [sc.ACCENT, sc.ACCENT, sc.ACCENT, sc.ACCENT, sc.GRAYS[0]]
ax.bar(xs, ratio, 0.62, color=cols, edgecolor=sc.INK, linewidth=1.2)
for x, r, t in zip(xs, ratio, labels):
    ax.text(x, r + 0.06, t, ha="center", va="bottom", fontsize=17, fontweight="bold",
            color=sc.ACCENT if x < 4 else sc.INK)
ax.axhline(1.0, color=sc.INK, lw=1.4, ls="--")
ax.set_xticks(list(xs))
ax.set_xticklabels(names)
ax.set_ylim(0, 3.0)
ax.set_ylabel("Kimi K3 / Kimi K2 (×)")
sc.save(fig, __file__)
