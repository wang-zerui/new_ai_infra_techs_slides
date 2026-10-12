"""Combine-forward latency vs routing imbalance (labeled values of MoonEP README figure comm_vs_deepep.png):
H20, EP=8, E=384, H=7168, K=8, S=8192/GPU, 32 SMs. See notes/sources.md, slide 13."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

MAXVIO = [0.2, 1, 10, 20]
deep = [2126, 2142, 2306, 2468]
moon = [1975, 1932, 1955, 1954]

sc.setup()
fig, ax = sc.figure("right")
xs = range(4)
ax.plot(xs, [v / 1000 for v in deep], marker="s", color=sc.GRAYS[2], lw=3, label="DeepEP v2")
ax.plot(xs, [v / 1000 for v in moon], marker="o", color=sc.ACCENT, lw=3.5, label="MoonEP")
for x in xs:
    ax.text(x, deep[x] / 1000 + 0.03, f"{deep[x] / 1000:.2f}", ha="center", va="bottom", fontsize=17, color=sc.INK)
    ax.text(x, moon[x] / 1000 - 0.03, f"{moon[x] / 1000:.2f}", ha="center", va="top", fontsize=17,
            color=sc.ACCENT, fontweight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels([str(m) for m in MAXVIO])
ax.set_ylim(1.7, 2.65)
ax.set_xlim(-0.35, 3.35)
ax.set_xlabel("Routing imbalance (maxvio)")
ax.set_ylabel("Combine fwd (ms)")
sc.top_legend(ax, ncol=2)
sc.better(ax, "Lower is better", "upper left")
sc.save(fig, __file__)
