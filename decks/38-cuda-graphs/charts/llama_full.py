"""Speedup from a full-iteration CUDA graph (FullCudaGraphWrapper), Llama 3.1 405B, FP4, 512 GPUs,
MLPerf Training v5.1. Source: NVIDIA dl-cuda-graph docs, "Llama 3.1 405B" example, Performance Impact table.
The eager baseline is 1.0 by definition. The same benchmark in FP8 shows no speedup on B200/GB200 (text)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

plats = ["B200", "GB200", "GB300"]
speed = [1.15, 1.54, 1.79]
sc.setup()
fig, ax = sc.figure("right")
w = 0.36
xs = range(len(plats))
ax.bar([x - w / 2 for x in xs], [1.0] * 3, w, color=sc.GRAYS[0], edgecolor=sc.INK, linewidth=1.2,
       hatch=sc.HATCHES[0], label="eager")
ax.bar([x + w / 2 for x in xs], speed, w, color=sc.ACCENT, edgecolor=sc.INK, linewidth=1.2,
       label="full-iteration graph")
for x, v in zip(xs, speed):
    ax.text(x + w / 2, v + 0.03, f"{v:.2f}×", ha="center", va="bottom", fontsize=20, fontweight="bold",
            color=sc.ACCENT)
ax.set_xticks(list(xs))
ax.set_xticklabels(plats)
ax.set_ylim(0, 2.15)
ax.set_ylabel("Relative training speed")
sc.top_legend(ax, ncol=2)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
