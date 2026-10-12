"""Mega MoE speedup over the non-fused pipeline, EP8, averaged over 8 ranks (DeepGEMM PR #316).

Numbers: sources.md S3.3.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

BATCH = ["1", "512", "8192", "32768"]
FLASH = [1.96, 1.73, 1.56, 1.62]
PRO = [1.61, 1.54, 1.50, 1.54]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, BATCH, ["V4-Flash", "V4-Pro"], [FLASH, PRO], ours=0, width=0.72)
ax.axhline(1.0, color=sc.PROBLEM, linewidth=2.2, linestyle="--")
ax.set_xlim(-0.55, 4.25)
ax.text(3.52, 1.02, "non-fused\npipeline", color=sc.PROBLEM, fontsize=17, ha="left", va="bottom")
ax.set_ylabel("Speedup (x)")
ax.set_xlabel("Tokens per rank")
ax.set_ylim(0, 2.3)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
