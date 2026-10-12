"""fig_coverage: which DeepEP V2.5 primitives move payload without SMs and which still need SM arithmetic."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 268
s = SVG(W, H, "fig_coverage: 0-SM payload path vs SM-bound primitives (DeepEP V2.5)")

groups = [
    (8, "payload moved by NIC or CE", ACC, ACC_F,
     [("PP send / recv", "RDMA put"), ("Engram fetch", "RDMA get"), ("all-gather", "CE or RDMA put")]),
    (448, "still needs SMs", NEU, NEU_F,
     [("EP dispatch", "routing + layout"), ("EP combine", "weighted sum"),
      ("reduce-scatter", "sum"), ("all-reduce", "sum")]),
]
for x0, title, stroke, fill, items in groups:
    s.rect(x0, 8, 414, 252, "#FFFFFF", stroke, 1.8, 8, dash=None if stroke == ACC else "6,4")
    s.text(x0 + 207, 36, title, 19, stroke if stroke == ACC else INK, weight="bold")
    for k, (name, how) in enumerate(items):
        y = 52 + k * 51
        s.rect(x0 + 16, y, 382, 42, fill, stroke, 1.5, 5)
        s.text(x0 + 30, y + 27, name, 18, INK, anchor="start", weight="bold")
        s.text(x0 + 384, y + 27, how, 17, ACC if stroke == ACC else INK, anchor="end")
s.save(os.path.join(HERE, "fig_coverage.svg"))
