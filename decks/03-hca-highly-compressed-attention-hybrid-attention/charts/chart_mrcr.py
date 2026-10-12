"""MRCR 8-needle average MMR vs input length ([V4] Fig. 9)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
x = ["8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"]
pro = [0.90, 0.85, 0.94, 0.90, 0.92, 0.82, 0.66, 0.59]
flash = [0.91, 0.84, 0.87, 0.85, 0.87, 0.76, 0.60, 0.49]
xs = range(len(x))
ax.plot(xs, pro, "-o", color=sc.ACCENT, label="V4-Pro-Max")
ax.plot(xs, flash, "--s", color=sc.ACCENT_LIGHT, label="V4-Flash-Max")
ax.set_xticks(list(xs)); ax.set_xticklabels(x, fontsize=16)
ax.set_ylim(0, 1.15)
ax.set_ylabel("MRCR 8-needle (avg MMR)")
ax.set_xlabel("Input tokens")
sc.top_legend(ax)
ax.axvspan(4.5, 7.4, color="#FBE5E5", zorder=0)
ax.annotate("0.92 $\\rightarrow$ 0.59", xy=(7, 0.59), xytext=(4.9, 0.25), color=sc.PROBLEM, fontsize=20, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=sc.PROBLEM, lw=2.2))
sc.better(ax, "Higher is better", "upper left")
sc.save(fig, __file__)
