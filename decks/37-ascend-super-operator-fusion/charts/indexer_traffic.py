"""Analytic: HBM traffic of the Lightning Indexer per decoded token at a 128K context.

Assumptions (notes/sources.md): bf16 (2 B) index keys and scores; one index key of dim 128 per
context token; 32 index heads; S = 131,072. An op-by-op implementation materializes the 32 x S
score matrix (write + read) and the aggregated score vector (write + read); the fused kernel
keeps both on chip. Index keys are streamed from HBM either way.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

S, HEADS, B = 131072, 32, 2
keys = S * 128 * B / 1e6                      # 33.55 MB
inter = (HEADS * S * B * 2 + S * B * 2) / 1e6  # 17.30 MB

sc.setup()
fig, ax = sc.figure("right")
names = ["score, ReLU, TopK\nas three ops", "one fused\nkernel"]
base = [keys, keys]
top = [inter, 0.0]
ax.bar([0, 1], base, width=0.56, color=[sc.GRAYS[1], sc.ACCENT], edgecolor=sc.INK, linewidth=1.2,
       hatch=["//", ""], label="index keys streamed (unavoidable)")
ax.bar([0], top[:1], bottom=base[:1], width=0.56, color="#FBE5E5", edgecolor=sc.PROBLEM,
       linewidth=1.2, label="score matrix written + read back")
ax.set_xticks([0, 1])
ax.set_xticklabels(names)
ax.set_ylabel("HBM traffic per token (MB)")
ax.set_ylim(0, 68)
ax.text(0, keys + inter + 1.5, f"{keys + inter:.0f} MB", ha="center", va="bottom", fontsize=20,
        fontweight="bold", color=sc.PROBLEM)
ax.text(1, keys + 1.5, f"{keys:.0f} MB", ha="center", va="bottom", fontsize=20, fontweight="bold",
        color=sc.ACCENT)
sc.top_legend(ax, ncol=1)
sc.better(ax, "Lower is better", "upper right")
sc.save(fig, __file__)
