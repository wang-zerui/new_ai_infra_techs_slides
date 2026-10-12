"""fig_disp_1 / fig_disp_2: HybridEP dispatch kernel, one persistent warp-specialized block per SM.

Build 1 (one NVLink domain): G2S warp group TMA-loads token chunks into an SMEM cyclic FIFO; S2G warp group
TMA-stores each token only to the ranks the routing map selects.
Build 2 (several domains): an RDMA warp group sends tokens to the same-local-index GPU of every other domain
with IBGDA and receives peer tokens, which G2S/S2G then forward inside the domain.

Sources: DeepEP hybrid-ep docs/Hybrid-EP_Implementation.md Sec. 8.1; NVIDIA blog (Feb 2026) Fig. 2 text;
arXiv 2603.07685 Sec. 4.2.2 p.31.
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))


def base(s):
    # ---- left: local memory of GPU A0
    s.text(72, 22, "GPU A0 memory", 17, INK, weight="bold")
    s.box(4, 34, 136, 36, "routing map", NEU_F, NEU, size=17)
    s.rect(4, 86, 136, 146, "#FFFFFF", NEU, 2, 4)
    s.text(72, 108, "token chunks", 17, INK)
    for k in range(4):
        y = 120 + k * 27
        s.rect(16, y, 112, 21, NEU_F, NEU, 1.4, 2)
        s.text(72, y + 16, f"t{k + 1}", 15, INK)
    # ---- middle: one persistent block
    s.rect(196, 34, 390, 212, "#FFFFFF", ACC, 2.4, 6)
    s.text(391, 22, "one persistent block per SM (N in parallel)", 17, ACC, weight="bold")
    s.box(212, 64, 110, 60, ["G2S", "warps"], ACC_F, ACC, size=17, weight="bold")
    s.box(466, 64, 110, 60, ["S2G", "warps"], ACC_F, ACC, size=17, weight="bold")
    for k in range(4):
        s.rect(348 + k * 23, 76, 21, 36, "#FFFFFF", ACC, 1.6, 2)
    s.text(393, 140, "SMEM FIFO", 16, INK)
    s.line(324, 94, 346, 94, INK, 2.2, "arr")
    s.line(441, 94, 464, 94, INK, 2.2, "arr")
    # token chunks -> G2S (TMA load)
    s.line(142, 108, 210, 108, ACC, 2.4, "arrAcc")
    s.text(170, 100, "TMA", 16, ACC, weight="bold")
    # routing map -> S2G (decides destinations)
    s.path("M142,52 L521,52 L521,62", INK, 1.8, "arr")
    # ---- right: NVLink domain A
    s.text(748, 22, "NVLink domain A", 17, INK, weight="bold")
    s.rect(630, 34, 236, 140, "#FFFFFF", NEU, 2, 6)
    ys = [46, 77, 108, 139]
    for k, y in enumerate(ys):
        name = f"A{k}"
        if k == 2:
            s.box(646, y, 204, 26, name, "#FFFFFF", "#A6A6A6", color=MUTED, size=16, dash="4,3")
        else:
            s.box(646, y, 204, 26, name + " expert input", NEU_F, NEU, size=16)
    for k in (0, 1, 3):
        s.path(f"M578,94 L606,94 L606,{ys[k] + 13} L644,{ys[k] + 13}", ACC, 2.4, "arrAcc")
    s.text(611, 46, "TMA", 16, ACC, weight="bold")


# ---- build 1
s = SVG(870, 292, "fig_disp_1: HybridEP dispatch inside one NVLink domain")
base(s)
s.save(os.path.join(HERE, "fig_disp_1.svg"))

# ---- build 2: add the RDMA warp group and a second domain
s = SVG(870, 292, "fig_disp_2: HybridEP dispatch across NVLink domains")
base(s)
s.box(212, 172, 110, 56, ["RDMA", "warps"], ACC_F, ACC, size=17, weight="bold")
s.line(142, 200, 210, 200, ACC, 2.4, "arrAcc")
# peer tokens received by RDMA warps are forwarded through G2S/S2G
s.line(267, 170, 267, 126, ACC, 2.2, "arrAcc", dash="6,4")
s.text(276, 154, "peer tokens", 16, ACC, anchor="start")
# RDMA -> same-local-index GPU in domain B
s.text(748, 212, "NVLink domain B", 17, INK, weight="bold")
s.rect(630, 222, 236, 64, "#FFFFFF", NEU, 2, 6)
s.box(646, 236, 204, 36, "B0 (same local index)", NEU_F, NEU, size=16)
s.path("M324,200 L612,200 L612,254 L644,254", ACC, 2.6, "arrAcc")
s.text(440, 192, "IBGDA via NIC", 16, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_disp_2.svg"))
