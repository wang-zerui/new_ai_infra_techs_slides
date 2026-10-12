"""FP8 grouped Wgrad (4 experts) vs a dense FP8 Wgrad with the same per-expert (M, N, K).
Source: DeepGEMM PR #95 (2025-05-14) dense and grouped weight-gradient tables; Hopper; kernel time
(notes/sources.md). Grouped/dense ratios are derived."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["gate+up\nK 4096", "down\nK 4096", "gate+up\nK 8192", "down\nK 8192"]
dense = [942, 920, 1061, 1037]
grouped = [939, 900, 1044, 1033]

sc.setup()
fig, ax = sc.figure("right")
w = 0.38
for i, (vals, name, col, hatch) in enumerate([(dense, "dense Wgrad", sc.GRAYS[0], "//"),
                                              (grouped, "grouped, 4 experts", sc.ACCENT, None)]):
    xs = [g + (i - 0.5) * w for g in range(len(groups))]
    ax.bar(xs, vals, w * 0.95, color=col, edgecolor=sc.INK, linewidth=1.2, hatch=hatch, label=name)
for g, (d, v) in enumerate(zip(dense, grouped)):
    ax.text(g + 0.5 * w, v + 18, f"{100 * v / d:.1f}%", ha="center", va="bottom", fontsize=18,
            fontweight="bold", color=sc.ACCENT)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups)
ax.set_ylim(0, 1450)
ax.set_ylabel("FP8 TFLOPS (Hopper)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.2,
          columnspacing=1.0, borderaxespad=0.2)
sc.better(ax, where="upper right")
sc.save(fig, __file__)
