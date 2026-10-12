"""Dense FP8 GEMM throughput on H800 SXM5 at N=7168, K=16384 (DeepGEMM README @ 8002b769, 2025-03-25).

Numbers: sources.md S2.4. Speedup labels are versus DeepSeek's internally tuned CUTLASS 3.6
implementation on the same GPU (sources.md S2.3).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc  # noqa: E402

M = ["M = 64\n(decode)", "M = 128", "M = 4096\n(prefill)"]
TFLOPS = [336, 645, 1426]
SPEEDUP = ["336 TF\n1.4x", "645 TF\n1.4x", "1426 TF\n1.3x"]

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, M, TFLOPS, ours=[0, 1, 2], labels=SPEEDUP, width=0.55, label_size=19)
ax.set_ylabel("FP8 GEMM (TFLOPS)")
ax.set_ylim(0, 1900)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
