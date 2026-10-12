"""Speedups reported for the fused kernels in the 2026-09 DeepGEMM release (PR #432).

Numbers: sources.md S3.4 — Mega Gate "30-75% speedups", Mega mHC "45-85% speedups",
MoE backward GEMMs "~20% gains". Ranges are drawn as floating bars, the single value as a marker.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

RANGES = [("Mega Gate", 30, 75), ("Mega mHC", 45, 85)]
POINT = ("MoE backward", 20)

sc.setup()
fig, ax = sc.figure("right")
ys = [2, 1]
for y, (_, lo, hi) in zip(ys, RANGES):
    ax.barh(y, hi - lo, left=lo, height=0.5, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2)
    ax.text(hi + 2, y, f"{lo}-{hi}%", va="center", ha="left", fontsize=20,
            fontweight="bold", color=sc.ACCENT)
ax.plot([POINT[1]], [0], marker="D", markersize=14, color=sc.ACCENT)
ax.text(POINT[1] + 5, 0, "~20%", va="center", ha="left", fontsize=20,
        fontweight="bold", color=sc.ACCENT)

ax.set_yticks([2, 1, 0])
ax.set_yticklabels([RANGES[0][0], RANGES[1][0], POINT[0]])
ax.set_xlabel("Faster than the previous release (%)")
ax.set_xlim(0, 108)
ax.set_ylim(-0.6, 2.6)
ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#BFBFBF")
ax.grid(axis="y", visible=False)
sc.better(ax, "Further right is better", where="upper right")
sc.save(fig, __file__)
