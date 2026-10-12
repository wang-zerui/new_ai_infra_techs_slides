"""SM90 FP8 MQA-logits kernel throughput on H200 at production indexer shapes (D = 128).
Source: DeepGEMM PR #452 (community, open, 2026-09-18), 'Performance (H200, 132 SMs)' table: before = DeepGEMM main,
after = ReLU as x + |x| (FADD) instead of fmaxf (FMNMX). Peak line: H200 SXM FP8 3,958 TFLOPS 'with sparsity'
(NVIDIA H200 page) / 2 = 1,979 dense (derived)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

shapes = ["H32\nKV 140K", "H64\nKV 131K", "H32\nKV 131K", "H64\nKV 8K"]
before = [1018, 1042, 941, 980]
after = [1093, 1126, 989, 1027]
sc.setup()
fig, ax = sc.figure("right")
w = 0.38
xs = range(len(shapes))
ax.bar([x - w / 2 for x in xs], before, w, color=sc.ACCENT, edgecolor=sc.INK, lw=1.2, label="DeepGEMM main")
ax.bar([x + w / 2 for x in xs], after, w, color=sc.ACCENT_LIGHT, edgecolor=sc.INK, lw=1.2,
       label="+ FADD ReLU (PR #452)")
for x, b, a in zip(xs, before, after):
    ax.text(x - w / 2 - 0.03, b + 25, f"{b:,}", ha="center", va="bottom", fontsize=15, fontweight="bold", color=sc.ACCENT)
    ax.text(x + w / 2 + 0.05, a + 25, f"+{(a / b - 1) * 100:.0f}%", ha="center", va="bottom", fontsize=15, color=sc.INK)
ax.axhline(1979, color=sc.GRAYS[3], ls="--", lw=2)
ax.text(3.45, 1979 - 40, "H200 dense FP8 peak", ha="right", va="top", fontsize=16, color=sc.GRAYS[3])
ax.set_xticks(list(xs)); ax.set_xticklabels(shapes, fontsize=16)
ax.set_ylim(0, 2200)
ax.set_xlim(-0.6, 3.6)
ax.set_ylabel("Throughput (TFLOPS)")
sc.top_legend(ax, ncol=2, fontsize=16)
sc.save(fig, __file__)
