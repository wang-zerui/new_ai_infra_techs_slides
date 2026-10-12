"""Validation loss of residual variants on the 16-layer model (AttnRes paper, arXiv 2603.15031, Table 4):
PreNorm baseline 1.766, DenseFormer (static cross-layer weights) 1.767, SWA (last 8 layers + embedding) 1.764,
input-independent mixing (learned scalars) 1.749, mHC 1.747, Full AttnRes 1.737."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["PreNorm\nbaseline", "Dense-\nFormer", "window\n(8 layers)", "static\nmixing", "mHC", "AttnRes\n(Full)"]
loss = [1.766, 1.767, 1.764, 1.749, 1.747, 1.737]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(names))
cols = [sc.GRAYS[2], sc.PROBLEM, sc.GRAYS[0], sc.GRAYS[0], sc.GRAYS[1], sc.ACCENT]
hat = [None, None, "//", "\\\\", "xx", None]
for x, v, c, h in zip(xs, loss, cols, hat):
    ax.bar(x, v - 1.72, bottom=1.72, width=0.64, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=h)
    ax.text(x, v + 0.0008, f"{v:.3f}", ha="center", va="bottom", fontsize=17,
            fontweight="bold" if x in (1, 5) else "normal",
            color=sc.PROBLEM if x == 1 else (sc.ACCENT if x == 5 else sc.INK))
ax.set_xticks(list(xs))
ax.set_xticklabels(names, fontsize=16)
ax.set_ylim(1.72, 1.775)
ax.set_ylabel("Validation loss")
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
