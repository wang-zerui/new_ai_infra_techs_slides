"""LongCat-Flash-Lite-FP8 input-throughput gain from HPC-Ops Router GEMM alone, 8x H20, 1,024-in/128-out.
Source: LMSYS blog Table 10."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

sc.setup()
fig, ax = sc.figure("right")
bs = ["1", "4", "8", "16", "32", "64"]
gain = [0.5, 6.1, 5.6, 5.7, 5.5, 5.7]
sc.bars(ax, bs, gain, ours=[0, 1, 2, 3, 4, 5], labels=[f"+{g}%" for g in gain], label_size=17)
ax.set_xlabel("Batch size")
ax.set_ylabel("Input throughput gain (%)")
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
