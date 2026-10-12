"""Grouped (contiguous) FP8 GEMM vs a dense FP8 GEMM with the same per-expert shape, H800 SXM5.
Source: DeepGEMM README @ 8002b769 (2025-03-25), dense and contiguous-grouped tables (notes/sources.md)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["gate + up\nN 4096, K 7168", "down\nN 7168, K 2048"]
dense = [1265, 1168]          # dense, M = 4096
g8 = [1346, 1214]             # 8 experts x 4096 rows
g4 = [1346, 1214]             # 4 experts x 8192 rows

sc.setup()
fig, ax = sc.figure("right")
w = 0.26
for i, (vals, name, col, hatch) in enumerate([
        (dense, "dense 4096", sc.GRAYS[0], "//"),
        (g8, "8 × 4096", sc.ACCENT_LIGHT, ""),
        (g4, "4 × 8192", sc.ACCENT, "")]):
    xs = [g + (i - 1) * w for g in range(len(groups))]
    bars = ax.bar(xs, vals, w * 0.95, color=col, edgecolor=sc.INK, linewidth=1.2, hatch=hatch or None, label=name)
    for x, v in zip(xs, vals):
        ax.text(x, v + 18, f"{v}", ha="center", va="bottom", fontsize=18, fontweight="bold",
                color=sc.ACCENT if i else sc.INK)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups)
ax.set_ylim(0, 1650)
ax.set_ylabel("FP8 TFLOPS on H800")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.2,
          columnspacing=0.9, borderaxespad=0.2)
sc.better(ax, where="upper right")
sc.save(fig, __file__)
