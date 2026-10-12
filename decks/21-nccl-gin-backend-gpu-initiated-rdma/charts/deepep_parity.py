"""DeepEP v1.2.1 kernels on NVSHMEM vs ported to NCCL GIN, BF16 dispatch bandwidth, NVIDIA EOS (H100, 8x400 Gb/s IB per
node, 24 SMs per GPU). Source: arXiv 2511.15076v2 Sec. V-B (HT, 2 nodes) and V-C (LL, 1 node, with and without NVLink)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
import slidechart as sc

groups = ["HT\n2 nodes", "LL\n1 node", "LL RDMA-only\n1 node"]
nvshmem = [84.97, 182.15, 46.79]
gin = [84.36, 185.28, 47.00]
sc.setup()
fig, ax = sc.figure("right")
sc.grouped(ax, groups, ["NVSHMEM", "NCCL GIN"], [nvshmem, gin], ours=1, width=0.72)
w = 0.72 / 2
for g in range(3):
    for s, vals, col in ((0, nvshmem, sc.INK), (1, gin, sc.ACCENT)):
        x = g - 0.72 / 2 + w * (s + 0.5)
        ax.text(x, vals[g] + 3, f"{vals[g]:.1f}", ha="center", va="bottom", fontsize=18, fontweight="bold", color=col)
ax.set_ylabel("Dispatch bandwidth (GB/s)")
ax.set_ylim(0, 230)
sc.better(ax, "Higher is better", where="upper left")
sc.save(fig, __file__)
