"""Derived: where GLM-5's 744B parameters sit — routed experts vs everything else.

256 routed experts x 3 matrices x 6144 x 2048 x 75 MoE layers = 724.8e9 (97.4%); the remaining
19.2e9 covers attention, the shared experts, the indexer and the dense MLP blocks. Parameter
shapes from the GLM-5 report Table 10; derivation in notes/sources.md.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

P, E = 744.0, 724.8     # in billions of parameters
rest = P - E

sc.setup()
fig, ax = sc.figure("right")
fig.set_size_inches(7.2, 3.0)
ax.barh([0], [E], height=0.45, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2,
        label="routed experts")
ax.barh([0], [rest], left=[E], height=0.45, color=sc.GRAYS[1], edgecolor=sc.INK, linewidth=1.2,
        hatch="//", label="attention + MLP + indexer")
ax.text(E / 2, 0, "724.8B  ·  97%", ha="center", va="center", fontsize=22, fontweight="bold",
        color="#FFFFFF")
ax.annotate("19.2B  ·  3%", xy=(P - rest / 2, 0.24), xytext=(P - 150, 0.50),
            fontsize=20, fontweight="bold", color=sc.INK,
            arrowprops=dict(arrowstyle="-|>", color=sc.INK, lw=2.0), ha="center")
ax.set_xlim(0, P * 1.02)
ax.set_ylim(-0.34, 0.66)
ax.set_yticks([])
ax.set_xlabel("Parameters (billions)")
ax.grid(False)
sc.top_legend(ax, ncol=1)
sc.save(fig, __file__)
