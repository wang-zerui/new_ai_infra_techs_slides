"""fig_layout: overview. One reserved VA range: GPU segment backed by HBM pages (NCCL window for SMs, NVLink
peers and RDMA), storage shards backed by pinned host pages on the GPU's NUMA node (Gin-only windows for the
NIC); shards 1-3 belong to the node-local peers (hybrid mode)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 296
s = SVG(W, H, "fig_layout: VA range = HBM segment + host-DRAM storage shards")

Y, BH = 128, 50
# VA bar
s.rect(40, Y, 64, BH, NEU_F, NEU, 1.6, 0)
s.text(72, Y + 31, "ws", 16, INK)
s.rect(104, Y, 226, BH, NEU_F, NEU, 1.6, 0)
s.text(217, Y + 31, "GPU buffer", 18, INK, weight="bold")
for k in range(4):
    x = 330 + k * 125
    s.rect(x, Y, 125, BH, ACC_F, ACC, 2.4 if k == 0 else 1.6, 0)
    s.text(x + 62, Y + 31, f"shard {k}", 17, ACC if k == 0 else INK, weight="bold" if k == 0 else "normal")
s.text(24, Y + 31, "VA", 17, MUTED, anchor="middle", weight="bold")
# accessors
s.box(96, 22, 242, 48, "SMs and NVLink peers", "#FFFFFF", NEU, size=18)
s.line(217, 72, 217, Y - 3, INK, 2.2, "arr")
s.text(227, 104, "NCCL window", 16, INK, anchor="start", style="italic")
s.box(524, 22, 112, 48, "RDMA NIC", ACC_F, ACC, ACC, 18, "bold")
s.line(580, 72, 580, Y - 3, ACC, 2.4, "arrAcc")
s.text(590, 104, "Gin-only windows", 16, ACC, anchor="start", style="italic")
s.badge(370, 98, 2)
s.text(390, 104, "two window kinds", 16, INK, anchor="start")
# physical backing
PY = 232
s.rect(40, PY, 290, 52, NEU_F, NEU, 1.6, 4)
s.text(185, PY + 22, "HBM pages", 18, INK, weight="bold")
s.text(185, PY + 44, "cuMemCreate on device", 15, INK, family="Menlo, monospace")
s.rect(420, PY, 410, 52, ACC_F, ACC, 1.6, 4)
s.text(625, PY + 22, "pinned host pages on the GPU's NUMA node", 17, INK, weight="bold")
s.text(625, PY + 44, "cuMemCreate on HOST_NUMA", 15, INK, family="Menlo, monospace")
for (x0, x1) in ((40, 40), (330, 330)):
    s.line(x0, Y + BH + 2, x1, PY - 2, NEU, 1.6, dash="5,4")
for (x0, x1) in ((330, 420), (830, 830)):
    s.line(x0, Y + BH + 2, x1, PY - 2, ACC, 1.6, dash="5,4")
s.badge(62, 206, 1)
s.text(80, 212, "one VA range with two backings", 16, INK, anchor="start")
s.path(f"M{455 + 4},{Y + BH + 8} L{455 + 4},{Y + BH + 14} L{826},{Y + BH + 14} L{826},{Y + BH + 8}", INK, 1.6)
s.badge(560, 210, 3)
s.text(578, 216, "node peers' shards", 16, INK, anchor="start")
s.save(os.path.join(HERE, "fig_layout.svg"))
