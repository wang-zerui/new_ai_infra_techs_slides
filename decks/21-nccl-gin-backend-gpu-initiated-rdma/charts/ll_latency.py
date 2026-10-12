"""DeepEP v1.2.1 low-latency BF16 dispatch latency with NVLink enabled, NVSHMEM vs NCCL GIN, NVIDIA EOS H100.
Source: arXiv 2511.15076v2 Sec. V-C (1 node: 41.43 vs 40.62 us; 2 nodes: 157.00 vs 142.51 us, '9% lower')."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["1 node (8 GPUs)", "2 nodes (16 GPUs)"]
nvshmem = [41.43, 157.00]
gin = [40.62, 142.51]
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, ["NVSHMEM", "NCCL GIN"], [nvshmem, gin], ours=1, width=0.66)
w = 0.66 / 2
for g in range(2):
    for s, vals, col in ((0, nvshmem, sc.INK), (1, gin, sc.ACCENT)):
        x = g - 0.66 / 2 + w * (s + 0.5)
        ax.text(x, vals[g] + 3, f"{vals[g]:.1f}", ha="center", va="bottom", fontsize=18, fontweight="bold", color=col)
ax.text(1 + w / 2, 142.51 / 2, "−9%", ha="center", va="center", fontsize=22, fontweight="bold", color="white")
ax.set_ylabel("Dispatch latency (µs)")
ax.set_ylim(0, 190)
sc.better(ax, "Lower is better", where="upper left")
sc.save(fig, __file__)
