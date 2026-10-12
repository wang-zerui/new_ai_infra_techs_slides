"""Analytic weight-memory of GLM-5 (744B params) at three precisions vs one Atlas 800 A3 node.

Derivation is written out in notes/sources.md ("Derived numbers"):
  expert weights = 3 x 6144 x 2048 x (256 + 1) x 75 = 727.6B params (97.8% of 744B)
  BF16  = 744B x 2B                        = 1386 GiB
  W8A8  = 744B x 1B                        =  693 GiB
  W4A8  = 727.6B x 0.5B + 16.4B x 1B       =  354 GiB
  node  = 8 x 128 GB                       = 1024 GiB
Excludes quantization scales, KV cache and activations.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

NAMES = ["BF16", "W8A8\n(INT8)", "W4A8\n(mixed)"]
GIB = [1386, 693, 354]
NODE = 1024

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, NAMES, GIB, ours=2, problem=0, labels=["1,386", "693", "354"])
ax.axhline(NODE, color=sc.INK, linestyle="--", linewidth=2.0)
ax.text(2.42, NODE + 40, "one node: 1,024 GiB", ha="right", va="bottom", fontsize=18,
        fontweight="bold", color=sc.INK)
ax.set_ylabel("Weight memory (GiB)")
ax.set_ylim(0, 1750)
sc.better(ax, "Lower is better", "upper right")
sc.save(fig, __file__)
