"""Residual-mechanism memory access per token per layer, in units of d (AttnRes paper, Table 1, typical
L = 128, N = 8, S = 16, m = 4): standard residual 3d, Block AttnRes 5.5d, Full AttnRes 24d, mHC (4 streams) 34d."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

names = ["standard\nresidual", "Block\nAttnRes", "Full\nAttnRes", "mHC\n(4 streams)"]
io = [3, 5.5, 24, 34]

sc.setup()
fig, ax = sc.figure("right")
xs = range(len(names))
cols = [sc.GRAYS[0], sc.ACCENT, sc.ACCENT_LIGHT, sc.GRAYS[2]]
hat = ["//", None, None, "xx"]
for x, v, c, h in zip(xs, io, cols, hat):
    ax.bar(x, v, width=0.62, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=h)
    ax.text(x, v + 0.6, f"{v:g}d", ha="center", va="bottom", fontsize=19, fontweight="bold",
            color=sc.ACCENT if x == 1 else sc.INK)
ax.set_xticks(list(xs))
ax.set_xticklabels(names)
ax.set_ylim(0, 40)
ax.set_ylabel("Residual I/O per layer (× d)")
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
