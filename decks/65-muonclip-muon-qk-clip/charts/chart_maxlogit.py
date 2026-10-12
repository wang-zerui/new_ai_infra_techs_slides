"""chart_maxlogit: max attention logit, log scale.
Vanilla Muon, 53B-total / 9B-active MoE: "quickly exceed a magnitude of 1000" (K2 §2.1, Fig. 2 left) -> lower bound.
Kimi K2 with MuonClip, tau = 100: "capped at 100" (K2 §2.1, Fig. 2 right). Different model scales."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["Vanilla Muon\n(53B MoE)", "MuonClip\n(Kimi K2)"]
r = ax.bar([0, 1], [1000, 100], width=0.56, color=[sc.PROBLEM, sc.ACCENT], edgecolor=sc.INK, linewidth=1.2)
r[0].set_hatch("//"); r[0].set_facecolor("#F4C7C7"); r[0].set_edgecolor(sc.PROBLEM)
ax.set_yscale("log")
ax.set_ylim(10, 6000)
ax.set_yticks([10, 100, 1000]); ax.set_yticklabels(["10", "100", "1000"]); ax.minorticks_off()
ax.set_xticks([0, 1]); ax.set_xticklabels(names)
ax.set_xlim(-0.6, 1.6)
ax.annotate("", xy=(0, 3300), xytext=(0, 1050), arrowprops=dict(arrowstyle="-|>", color=sc.PROBLEM, lw=2.6))
ax.text(0.33, 2000, "> 1000", color=sc.PROBLEM, fontsize=22, fontweight="bold", ha="left", va="center")
ax.text(1, 125, "100 = τ", color=sc.ACCENT, fontsize=22, fontweight="bold", ha="center", va="bottom")
ax.axhline(100, color=sc.ACCENT, lw=1.6, ls="--")
ax.set_ylabel("Max attention logit (log)")
sc.better(ax, "Lower is safer", where="upper right")
sc.save(fig, __file__)
