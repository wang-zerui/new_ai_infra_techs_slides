"""Masked grouped FP8 GEMM on H800 at decode-like sizes, with speedup over DeepSeek's internal CUTLASS 3.6-based
implementation. Source: DeepGEMM README @ 8002b769 (2025-03-25), masked-layout table (notes/sources.md)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

cfg = ["1 × 1024", "2 × 512", "4 × 256"]          # experts × rows per expert
up = [1233, 1040, 932]                            # N 4096, K 7168 (gate + up)
up_x = ["1.2×", "1.2×", "1.1×"]
down = [925, 916, 815]                            # N 7168, K 2048 (down)
down_x = ["1.2×", "1.2×", "1.2×"]

sc.setup()
fig, ax = sc.figure("right")
w = 0.36
for i, (vals, lab, name, col) in enumerate([(up, up_x, "gate + up (N 4096, K 7168)", sc.ACCENT),
                                            (down, down_x, "down (N 7168, K 2048)", sc.ACCENT_LIGHT)]):
    xs = [g + (i - 0.5) * w for g in range(len(cfg))]
    ax.bar(xs, vals, w * 0.95, color=col, edgecolor=sc.INK, linewidth=1.2, label=name)
    for x, v, t in zip(xs, vals, lab):
        ax.text(x, v + 20, t, ha="center", va="bottom", fontsize=18, fontweight="bold", color=sc.ACCENT)
ax.set_xticks(range(len(cfg)))
ax.set_xticklabels(cfg)
ax.set_xlabel("Experts × M per expert")
ax.set_ylabel("FP8 TFLOPS on H800")
ax.set_ylim(0, 1600)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, handlelength=1.2,
          borderaxespad=0.2)
ax.text(2 - 0.5 * w, 470, "2064\nGB/s", ha="center", va="center", fontsize=18, fontweight="bold", color="white")
sc.better(ax, where="upper right")
sc.save(fig, __file__)
