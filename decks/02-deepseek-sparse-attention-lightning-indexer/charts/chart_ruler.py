"""RULER accuracy vs context length for GLM-4.7-Flash (MLA) before and after DSA conversion.

Source: GLM-5 technical report, arXiv 2602.15763v2, Table 6 (p.8). Warm-up only = indexer trained 1,000 steps with
the base model frozen; + DSA = warm-up followed by joint training on 150B tokens.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

ctx = ["4K", "8K", "16K", "32K", "64K", "128K"]
base = [97.44, 96.72, 95.83, 92.96, 85.34, 79.21]
warm = [97.51, 96.54, 95.40, 90.09, 84.05, 71.35]
dsa = [96.77, 96.25, 96.69, 93.45, 87.06, 78.86]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(ctx))
ax.plot(xs, base, color=sc.GRAYS[3], ls="--", marker="s", lw=2.6, label="GLM-4.7-Flash (dense MLA)")
ax.plot(xs, warm, color=sc.GRAYS[1], ls=":", marker="^", lw=2.6, label="+ indexer warm-up only")
ax.plot(xs, dsa, color=sc.ACCENT, marker="o", lw=3.4, label="+ DSA (warm-up + 150B tokens)")
ax.set_xticks(list(xs))
ax.set_xticklabels(ctx)
ax.set_ylim(66, 101)
ax.set_xlim(-0.3, 5.75)
ax.set_ylabel("RULER accuracy (%)")
ax.set_xlabel("Context length")
ax.text(5.12, 78.86, "78.9", color=sc.ACCENT, fontsize=20, fontweight="bold", va="top", ha="left")
ax.text(5.12, 80.4, "79.2", color=sc.GRAYS[3], fontsize=18, va="bottom", ha="left")
ax.text(5.12, 71.35, "71.4", color=sc.GRAYS[2], fontsize=18, va="center", ha="left")
sc.better(ax, "Higher is better", where="upper right")
ax.legend(loc="lower left", frameon=False, fontsize=17, handlelength=1.8)
sc.save(fig, __file__)
