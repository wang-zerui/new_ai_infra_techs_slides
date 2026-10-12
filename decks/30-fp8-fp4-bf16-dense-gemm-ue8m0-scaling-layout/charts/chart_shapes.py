"""FP8 GEMM speedup over DeepSeek's tuned CUTLASS 3.6 implementation, H800 SXM5, NVCC 12.8.

Numbers: sources.md S6 (DeepGEMM README @ commit 8002b769, 2025-03-25). Bar labels are the measured
TFLOPS of the same table.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

M = ["M = 64", "M = 128", "M = 4096"]
SHAPE_A = [2.7, 2.4, 1.1]           # N=2112, K=7168
SHAPE_B = [1.4, 1.4, 1.3]           # N=7168, K=16384
LABEL_A = ["206 TF", "352 TF", "1009 TF"]
LABEL_B = ["336 TF", "645 TF", "1426 TF"]

sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, M, ["N=2112, K=7168", "N=7168, K=16384"], [SHAPE_A, SHAPE_B], ours=0, width=0.7)
for container, labels in zip(ax.containers, [LABEL_A, LABEL_B]):
    ax.bar_label(container, labels=labels, fontsize=17, padding=4)
ax.axhline(1.0, color=sc.PROBLEM, linewidth=2.2, linestyle="--")
ax.set_ylabel("Speedup (x)")
ax.set_ylim(0, 3.4)
sc.better(ax, "Higher is better", where="upper right")
sc.save(fig, __file__)
