"""fig_symmetric: status quo. Every rank holds the same buffer layout in HBM; peers address a slot by
(rank, offset): NVLink stores inside a node, RDMA puts across nodes."""
import os
from svgkit import SVG, ACC, NEU, NEU_F, INK, MUTED, PRB

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 268
s = SVG(W, H, "fig_symmetric: symmetric NCCL window, same offset on every rank, all in HBM")

X0, BW = 190, 540
WS = 70                     # workspace width
SLOT_X, SLOT_W = X0 + BW - 92, 46
rows = [("rank 0", "node A", 36), ("rank 1", "node A", 110), ("rank 2", "node B", 184)]
for name, node, y in rows:
    s.text(X0 - 16, y + 22, name, 18, INK, anchor="end", weight="bold")
    s.text(X0 - 16, y + 44, node, 16, MUTED, anchor="end")
    s.rect(X0, y, WS, 50, NEU_F, NEU, 1.6, 0)
    s.text(X0 + WS / 2, y + 31, "ws", 16, INK)
    s.rect(X0 + WS, y, BW - WS, 50, NEU_F, NEU, 1.6, 0)
    s.text(X0 + WS + 120, y + 31, "EP / Engram buffer", 17, INK)
    s.rect(SLOT_X, y + 6, SLOT_W, 38, "#FFFFFF", ACC, 2, 2)
# same-offset guide
s.line(SLOT_X + SLOT_W / 2, 26, SLOT_X + SLOT_W / 2, 242, MUTED, 1.4, dash="5,4")
s.text(SLOT_X + SLOT_W / 2, 20, "same offset on every rank", 16, MUTED, style="italic")
# NVLink store rank1 -> rank0 (left of slot), RDMA put rank2 -> rank0 (right side curve)
s.line(SLOT_X - 14, 114, SLOT_X - 14, 88, ACC, 2.4, "arrAcc")
s.text(SLOT_X - 22, 106, "NVLink store", 16, ACC, anchor="end", weight="bold")
s.path(f"M{SLOT_X + SLOT_W + 4},209 L{X0 + BW + 30},209 L{X0 + BW + 30},61 L{SLOT_X + SLOT_W + 4},61",
       ACC, 2.4, "arrAcc")
s.text(X0 + BW + 40, 140, "RDMA put", 16, ACC, anchor="start", weight="bold")
s.text(X0, 262, "all pinned HBM in one registered NCCL window", 16, INK, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_symmetric.svg"))
