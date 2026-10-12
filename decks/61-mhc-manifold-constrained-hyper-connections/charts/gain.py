"""Peak composite residual gain (Amax gain magnitude) in the 27B model: HC ~3000 ([mHC] §3.1, Fig. 3b),
mHC ~1.6 ([mHC] §5.4, Fig. 7b); identity residual = 1 by definition."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["Residual\n(identity)", "HC", "mHC"]
vals = [1, 3000, 1.6]
sc.bars(ax, names, vals, ours=2, problem=1, labels=None)
ax.set_yscale("log")
ax.set_ylim(0.5, 2e4)
for i, (v, t) in enumerate(zip(vals, ["1", "≈ 3000", "≈ 1.6"])):
    ax.text(i, v * 1.35, t, ha="center", va="bottom", fontsize=22, fontweight="bold",
            color=[sc.INK, sc.PROBLEM, sc.ACCENT][i])
ax.axhline(1, color=sc.GOOD, lw=1.6, ls="--")
ax.set_ylabel("Peak composite gain (log)")
sc.better(ax, "Ideal = 1", where="upper right")
sc.save(fig, __file__)
