"""Analytic per-token elements moved by the residual write-back at n = 4, in units of C (hidden size).
[mHC] Table 2: plain residual merge reads 2C, writes C. HC post + res + merge (unfused): reads (3n+1)C = 13C,
writes 3nC = 12C (C+n, nC+n^2, 2nC reads; nC x3 writes; small n-terms dropped). [mHC] §4.3.1: the fused
F_post,res kernel reads (n+1)C = 5C and writes nC = 4C."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
names = ["Residual\nmerge", "HC\nunfused", "mHC\nfused"]
reads, writes = [2, 13, 5], [1, 12, 4]
cols = [sc.GRAYS[1], sc.PROBLEM, sc.ACCENT]
kw = dict(edgecolor=sc.INK, linewidth=1.2, width=0.58)
for i in range(3):
    ax.bar(i, reads[i], color=cols[i], **kw)
    ax.bar(i, writes[i], bottom=reads[i], color="white", hatch="//", **kw)
    ax.text(i, reads[i] + writes[i] + 0.5, f"{reads[i] + writes[i]}C", ha="center", va="bottom", fontsize=22,
            fontweight="bold", color=cols[i] if i else sc.INK)
ax.bar([0], [0], color=sc.GRAYS[3], edgecolor=sc.INK, label="read (solid)")
ax.bar([0], [0], color="white", hatch="//", edgecolor=sc.INK, label="write (hatched)")
ax.set_xticks(range(3)); ax.set_xticklabels(names)
ax.set_ylabel("Elements per token (× C)")
ax.set_ylim(0, 31); ax.set_yticks([0, 10, 20, 30])
sc.top_legend(ax, ncol=2)
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
