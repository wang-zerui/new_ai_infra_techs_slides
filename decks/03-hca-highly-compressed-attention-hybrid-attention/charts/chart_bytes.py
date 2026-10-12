"""Analytic KV cache for one 2^20-token request at V4-Pro depth (61 layers); see notes/analytic.py.
'All CSA' / 'All HCA' are hypothetical uniform stacks used only to show what the mix buys."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notes"))
import slidechart as sc
import analytic as an

sc.setup()
fig, ax = sc.figure("right")
L = an.L
vals = [an.GQA8 * 61 * L / 1e9, an.stack_bytes(61, 0, 0) / 1e9, an.stack_bytes(30, 31, 0) / 1e9,
        an.stack_bytes(0, 61, 0) / 1e9]
names = ["BF16\nGQA8", "All\nCSA*", "V4-Pro\nmix", "All\nHCA*"]
labels = ["262 GB", "10.2 GB", "5.2 GB", "0.3 GB"]
cols = [sc.GRAYS[1], sc.GRAYS[0], sc.ACCENT, sc.GRAYS[0]]
hat = ["\\\\", "//", None, ".."]
for i, (v, c, h) in enumerate(zip(vals, cols, hat)):
    ax.bar(i, v, 0.62, color=c, edgecolor=sc.INK, linewidth=1.2, hatch=h)
    ax.text(i, v * 1.25, labels[i], ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT if i == 2 else sc.INK)
ax.set_yscale("log")
ax.set_ylim(0.1, 3000)
ax.set_xticks(range(4)); ax.set_xticklabels(names)
ax.set_ylabel("KV cache per 1M-token\nrequest (GB, log)")
ax.grid(axis="y", which="minor", visible=False)
sc.better(ax, "Lower is better", "upper right")
sc.save(fig, __file__)
