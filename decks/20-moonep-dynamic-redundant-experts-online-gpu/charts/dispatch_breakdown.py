"""Dispatch-forward latency, H20, EP=8, E=384, H=7168, K=8, S=8192/GPU, 32 SMs (MoonEP README figure
comm_vs_deepep.png). Totals are the chart's labels; MoonEP's comm / planning / prefetch split is ESTIMATED from
bar-segment pixel heights (linear fit on the labeled tops). See notes/sources.md, slide 12."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

MAXVIO = ["0.2", "1", "10", "20"]
deep_total = [2296, 2299, 2408, 2604]          # labeled
moon_total = [2598, 2644, 2629, 2413]          # labeled
moon_comm = [1832, 1852, 1849, 1839]           # estimated
moon_plan = [42, 42, 52, 52]                   # estimated
moon_pref = [t - c - p for t, c, p in zip(moon_total, moon_comm, moon_plan)]   # remainder (est. 724-750, 522)

sc.setup()
fig, ax = sc.figure("right")
w = 0.38
xs = range(4)
ax.bar([x - w / 2 for x in xs], [v / 1000 for v in deep_total], w, color=sc.GRAYS[0], edgecolor=sc.INK, lw=1.2,
       hatch="//", label="DeepEP v2")
b0 = [v / 1000 for v in moon_comm]
b1 = [v / 1000 for v in moon_plan]
b2 = [v / 1000 for v in moon_pref]
ax.bar([x + w / 2 for x in xs], b0, w, color=sc.ACCENT, edgecolor=sc.INK, lw=1.2, label="MoonEP comm")
ax.bar([x + w / 2 for x in xs], b1, w, bottom=b0, color=sc.INK, edgecolor=sc.INK, lw=1.2, label="plan")
ax.bar([x + w / 2 for x in xs], b2, w, bottom=[a + b for a, b in zip(b0, b1)], color=sc.ACCENT_LIGHT,
       edgecolor=sc.INK, lw=1.2, label="prefetch")
for x in xs:
    ax.text(x - w / 2, deep_total[x] / 1000 + 0.04, f"{deep_total[x] / 1000:.2f}", ha="center", va="bottom",
            fontsize=16, color=sc.INK)
    ax.text(x + w / 2, moon_total[x] / 1000 + 0.04, f"{moon_total[x] / 1000:.2f}", ha="center", va="bottom",
            fontsize=16, color=sc.ACCENT, fontweight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels(MAXVIO)
sc.better(ax, "Lower is better", "upper left")
ax.set_ylim(0, 3.4)
ax.set_xlabel("Routing imbalance (maxvio)")
ax.set_ylabel("Dispatch fwd (ms)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=16, handlelength=1.1,
          columnspacing=0.8, borderaxespad=0.2)
sc.save(fig, __file__)
