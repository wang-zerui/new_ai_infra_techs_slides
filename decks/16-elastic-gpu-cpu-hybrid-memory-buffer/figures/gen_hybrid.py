"""fig_hybrid: HybridElasticSymmetricMemory. Each node-local rank maps all node-local storage shards
back-to-back after its own GPU segment; same column = same physical host pages. A remote GET addresses one
rank of the node (rank 0 here) and names the shard by window."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_hybrid: node-wide contiguous storage shards, one RDMA peer per node")

GX, GW, CW, RH = 112, 112, 104, 40
cols = [GX + GW + k * CW for k in range(3)]
rows_y = [44, 104, 164]
PY = 236
# column headers
s.text(GX + GW / 2, 30, "own HBM", 16, MUTED, weight="bold")
for k, x in enumerate(cols):
    s.text(x + CW / 2, 30, f"shard {k}", 16, ACC, weight="bold")
# vertical "same physical pages" buses (drawn first, behind the cells)
for x in cols:
    s.line(x + CW / 2, rows_y[0] + RH, x + CW / 2, PY - 2, ACC, 1.8, dash="5,4")
for r, y in enumerate(rows_y):
    s.text(GX - 14, y + 26, f"rank {r}", 17, INK, anchor="end", weight="bold")
    s.rect(GX, y, GW, RH, NEU_F, NEU, 1.6, 0)
    s.text(GX + GW / 2, y + 26, f"GPU {r} seg", 16, INK)
    for k, x in enumerate(cols):
        own = k == r
        s.rect(x, y, CW, RH, ACC_F, ACC, 2.6 if own else 1.4, 0)
        s.text(x + CW / 2, y + 26, "own" if own else "mapped", 16, ACC if own else INK,
               weight="bold" if own else "normal")
# physical host shards
s.text(GX - 14, PY + 30, "host DRAM", 17, INK, anchor="end", weight="bold")
for k, x in enumerate(cols):
    s.rect(x + 6, PY, CW - 12, 46, ACC_F, ACC, 1.8, 4)
    s.text(x + CW / 2, PY + 29, f"s{k} pages", 16, ACC, weight="bold")
# remote reader -> rank 0, window of shard 2
s.box(700, 28, 150, 58, ["remote", "issue warp"], "#FFFFFF", NEU, size=17)
s.box(580, 28, 84, 58, "NIC", ACC_F, ACC, ACC, 18, "bold")
s.line(698, 57, 666, 57, INK, 2.2, "arr", dash="6,4")
s.path(f"M578,{57} L{cols[2] + CW + 14},{57} L{cols[2] + CW + 14},{rows_y[0] + 10} L{cols[2] + CW + 2},{rows_y[0] + 10}",
       ACC, 2.4, "arrAcc")
s.text(600, 118, "GET to rank 0", 16, ACC, anchor="start", weight="bold")
s.text(600, 140, "window of shard 2", 16, ACC, anchor="start")
# export / import note
s.text(600, 200, "export: one POSIX FD per shard", 16, INK, anchor="start")
s.text(600, 226, "import: pidfd_getfd + cuMemMap", 16, INK, anchor="start")
s.save(os.path.join(HERE, "fig_hybrid.svg"))
