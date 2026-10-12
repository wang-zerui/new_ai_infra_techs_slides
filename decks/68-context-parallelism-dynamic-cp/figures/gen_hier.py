"""fig_hier: hierarchical CP (a2a+p2p) on 2 nodes x 4 GPUs: a2a over heads inside the NVLink domain,
p2p ring over the sequence across nodes."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 340, "fig_hier: two nodes, a2a bus inside each, p2p arrows between nodes")

GW, GH = 78, 44
XS = [20, 112, 204, 296]


def node(y_box, y_gpu, y_bus, title_y, name, first):
    s.rect(6, y_box, 468, 132, "#FFFFFF", "#7F7F7F", 1.5, 6, dash="6,4")
    s.text(16, title_y, name, 18, INK, anchor="start", weight="bold")
    s.text(466, title_y, "a2a over heads", 17, ACC, anchor="end", weight="bold")
    s.line(XS[0] + GW / 2 - 1, y_bus, XS[-1] + GW / 2 + 1, y_bus, ACC, 3)
    for k, x in enumerate(XS):
        s.box(x, y_gpu, GW, GH, f"GPU{first + k}", NEU_F, NEU, size=17)
        stub_from = y_gpu if y_bus < y_gpu else y_gpu + GH
        s.line(x + GW / 2, y_bus, x + GW / 2, stub_from, ACC, 3)


node(6, 74, 50, 30, "node 0 · NVLink", 0)
node(202, 212, 290, 322, "node 1 · NVLink", 4)
for x in XS:
    s.line(x + GW / 2, 74 + GH + 5, x + GW / 2, 212 - 5, ACC, 2.4, "arrAcc", start_marker="arrAcc")
s.text(392, 156, "p2p ring", 17, ACC, anchor="start", weight="bold")
s.text(392, 174, "over the", 17, ACC, anchor="start", weight="bold")
s.text(392, 192, "sequence", 17, ACC, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_hier.svg"))
