"""fig_native: one KV split on GPU1 (node A) must reach 3 GPUs of node B.
(a) all-to-all-v prototype: one send buffer per destination -> 3 RDMA transfers of identical data.
(b) native GroupCast (DeepEP normal-mode design): one RDMA transfer to the peer with the same local rank (GPU5),
    which forwards over NVLink to GPU6 and GPU7 ([GRP] RDMA Transfer De-duplication). 4 GPUs per node drawn
    for space (the real intranode size is 8)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 244, "fig_native: RDMA de-duplication")

GW, GH, GG = 70, 42, 10


def panel(px, title, color):
    s.text(px, 20, title, 19, color, anchor="start", weight="bold")
    pos = {}
    for n, (ny, name) in enumerate(((36, "node A"), (172, "node B"))):
        s.rect(px, ny, 404, 60, "#FFFFFF", MUTED, 1.4, 5, dash="5,4")
        s.text(px + 10, ny + 36, name, 16, MUTED, anchor="start", weight="bold")
        for g in range(4):
            gid = n * 4 + g
            x = px + 84 + g * (GW + GG)
            src = gid == 1
            dst = gid in (5, 6, 7)
            fill, stroke = (ACC_F, ACC) if (src or dst) else (NEU_F, NEU)
            s.rect(x, ny + 9, GW, GH, fill, stroke, 2, 4)
            s.text(x + GW / 2, ny + 36, f"GPU{gid}", 16, INK, weight="bold" if (src or dst) else "normal")
            pos[gid] = (x + GW / 2, ny + 9, ny + 9 + GH)
    return pos


# (a) all-to-all-v
p = panel(10, "(a) All-to-all-v prototype", INK)
sx, _, sb = p[1]
for d in (5, 6, 7):
    dx, dt, _ = p[d]
    s.line(sx + (d - 6) * 8, sb + 3, dx, dt - 3, PRB, 2.4, "arrPrb")
s.text(192, 150, "3 RDMA copies", 17, PRB, anchor="end", weight="bold")
s.text(192, 130, "same data,", 17, PRB, anchor="end")

s.line(432, 10, 432, 236, "#BFBFBF", 1.2)

# (b) native
p = panel(452, "(b) Native GroupCast", ACC)
sx, _, sb = p[1]
dx, dt, db = p[5]
s.line(sx, sb + 3, dx, dt - 3, ACC, 2.6, "arrAcc")
s.text(sx - 12, 140, "1 RDMA send", 17, ACC, anchor="end", weight="bold")
for d in (6, 7):
    tx, _, tb = p[d]
    h = 26 if d == 6 else 46
    s.path(f"M{dx + 22},{dt - 3} C{dx + 22},{dt - h} {tx},{dt - h} {tx},{dt - 3}", ACC, 2.4, "arrAcc")
s.text(p[6][0] + 30, 134, "NVLink fan-out", 17, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_native.svg"))
