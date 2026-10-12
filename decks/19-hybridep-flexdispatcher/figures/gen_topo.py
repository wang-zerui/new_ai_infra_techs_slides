"""fig_topo: EP64 on H100 (8 NVLink islands joined by InfiniBand) vs GB200 NVL72 (one NVLink domain).

Facts: EP64 spans 8 nodes on H100/B200 NVL8; EP64 stays inside the NVLink domain on GB200 NVL72
(arXiv 2603.07685 Sec. 9.2.2 p.75). Link specs: ~50 GB/s inter-node (IB NDR, p.30), 1.8 TB/s NVLink (p.75).
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 352, "fig_topo: EP64 on H100 vs GB200 NVL72")

# ---- (a) H100: 8 nodes x 8 GPUs
s.text(8, 22, "H100: 8 NVLink nodes of 8", 19, INK, anchor="start", weight="bold")
NW, NG, X0, Y0, NH = 50, 6.857, 8, 36, 58
for n in range(8):
    x = X0 + n * (NW + NG)
    s.rect(x, Y0, NW, NH, "#FFFFFF", NEU, 2, 4)
    for r in range(2):
        for c in range(4):
            s.rect(x + 4.5 + c * 10.6, Y0 + 9 + r * 21, 9, 16, ACC_F, ACC, 1.2, 1)
# IB fabric
YB = 128
s.line(X0 + NW / 2, YB, X0 + 7 * (NW + NG) + NW / 2, YB, PRB, 3)
for n in range(8):
    cx = X0 + n * (NW + NG) + NW / 2
    s.line(cx, Y0 + NH, cx, YB, PRB, 2.2)
s.text(240, YB + 24, "InfiniBand between nodes (~50 GB/s)", 17, PRB, weight="bold")

# ---- (b) GB200 NVL72: one NVLink domain, EP64 = 64 of 72 GPUs
s.text(8, 186, "GB200 NVL72: one NVLink domain", 19, INK, anchor="start", weight="bold")
BX, BY, BW, BH = 8, 198, 464, 118
s.rect(BX, BY, BW, BH, "#FFFFFF", ACC, 2.4, 6)
CW, CH, GX, GY = 30, 11, 7.0, 5.4
cx0 = BX + (BW - (12 * CW + 11 * GX)) / 2
for i in range(72):
    r, c = divmod(i, 12)
    x = cx0 + c * (CW + GX)
    y = BY + 12 + r * (CH + GY)
    if i < 64:
        s.rect(x, y, CW, CH, ACC_F, ACC, 1.2, 1.5)
    else:
        s.rect(x, y, CW, CH, "#FFFFFF", "#A6A6A6", 1.2, 1.5, dash="3,2")
s.text(240, BY + BH + 26, "NVLink across 72 GPUs (1.8 TB/s bidirectional)", 17, ACC, weight="bold")
s.rect(262, 10, 14, 10, ACC_F, ACC, 1.2, 1.5)
s.text(282, 20, "EP64 rank", 15, INK, anchor="start")
s.rect(368, 10, 14, 10, "#FFFFFF", "#A6A6A6", 1.2, 1.5, dash="3,2")
s.text(388, 20, "not in group", 15, INK, anchor="start")
s.save(os.path.join(HERE, "fig_topo.svg"))
