"""Max abs error vs FP32 cuBLAS (H20, K=4096, N=192 sweep). Source: LMSYS blog Router GEMM section."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
sc.bars(ax, ["TF32 cuBLAS", "Route GEMM"], [0.06464, 0.00177], ours=1, labels=["0.0646", "0.00177"])
ax.set_ylabel("Max abs error vs FP32")
ax.set_yticks([0, 0.02, 0.04, 0.06])
sc.callout(ax, "36× lower", xy=(1, 0.013), xytext=(1, 0.042))
sc.better(ax, "Lower is better", where="upper right")
sc.save(fig, __file__)
