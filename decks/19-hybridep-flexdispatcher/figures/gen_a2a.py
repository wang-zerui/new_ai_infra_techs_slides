"""fig_a2a: one token t (top-4: experts on A1, B0, B1, B3) under (a) the all-to-all dispatcher and
(b) token-based dispatch (DeepEP / HybridEP).

(a) permutation replicates t once per selected expert, so three copies cross InfiniBand
    (arXiv 2603.07685 Sec. 4.2.2 p.30-31: "replicates each token top-k times and creates redundant traffic").
(b) one copy crosses to the same-local-index GPU of node B, which forwards it over NVLink
    (Sec. 4.2.2 p.31; Hybrid-EP_Implementation.md Sec. 8 "peers").
"""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_a2a: replicated all-to-all vs token-based dispatch")

AY0, AY1 = 30, 250            # node box vertical extent
A_X0, A_X1 = 4, 140           # node A box (panel-local)
B_X0, B_X1 = 280, 416         # node B box
GA_X0, GA_X1 = 18, 126        # GPUs in node A
GB_X0, GB_X1 = 324, 404       # GPUs in node B (leave a forwarding lane on the left)
A_GPUS = {"A0": 62, "A1": 182}
B_GPUS = {"B0": 52, "B1": 100, "B2": 148, "B3": 196}
GH_A, GH_B = 46, 38
SEL = {"A1", "B0", "B1", "B3"}


def chip(x, y, color, fill="#FFFFFF"):
    s.rect(x - 10, y - 10, 20, 20, fill, color, 2, 3)
    s.text(x, y + 5.5, "t", 16, color, weight="bold")


def panel(ox, mode):
    # nodes
    s.rect(ox + A_X0, AY0, A_X1 - A_X0, AY1 - AY0, "#FFFFFF", NEU, 2, 6)
    s.text(ox + (A_X0 + A_X1) / 2, AY0 - 9, "node A", 17, INK, weight="bold")
    s.rect(ox + B_X0, AY0, B_X1 - B_X0, AY1 - AY0, "#FFFFFF", NEU, 2, 6)
    s.text(ox + (B_X0 + B_X1) / 2, AY0 - 9, "node B", 17, INK, weight="bold")
    s.text(ox + (A_X1 + B_X0) / 2, AY0 - 9, "InfiniBand", 17, MUTED, style="italic")
    for name, y in A_GPUS.items():
        fill = NEU_F if name in SEL or name == "A0" else "#FFFFFF"
        s.box(ox + GA_X0, y, GA_X1 - GA_X0, GH_A, "GPU " + name, fill, NEU, size=17)
    for name, y in B_GPUS.items():
        if name in SEL:
            s.box(ox + GB_X0, y, GB_X1 - GB_X0, GH_B, name, NEU_F, NEU, size=17)
        else:
            s.box(ox + GB_X0, y, GB_X1 - GB_X0, GH_B, name, "#FFFFFF", "#A6A6A6", color=MUTED, size=17, dash="4,3")
    a0_r = ox + GA_X1
    a0_cy = A_GPUS["A0"] + GH_A / 2
    # A0 -> A1 inside node A (NVLink), same in both panels
    s.line(ox + 72, A_GPUS["A0"] + GH_A + 2, ox + 72, A_GPUS["A1"] - 2, NEU, 2.2, "arr")
    chip(ox + 96, (A_GPUS["A0"] + GH_A + A_GPUS["A1"]) / 2, NEU)
    if mode == "a2a":
        for k, name in enumerate(["B0", "B1", "B3"]):
            ty = B_GPUS[name] + GH_B / 2
            sy = a0_cy - 12 + k * 12
            s.line(a0_r + 2, sy, ox + GB_X0 - 2, ty, PRB, 2.4, "arrPrb")
            mx = ox + 206
            my = sy + (ty - sy) * (mx - a0_r - 2) / (ox + GB_X0 - a0_r - 4)
            chip(mx, my, PRB, PRB_F)
    else:
        ty = B_GPUS["B0"] + GH_B / 2
        s.line(a0_r + 2, a0_cy, ox + GB_X0 - 2, ty, ACC, 2.6, "arrAcc")
        chip(ox + 206, a0_cy + (ty - a0_cy) * (206 - GA_X1) / (GB_X0 - GA_X1), ACC, ACC_F)
        # NVLink forwarding inside node B along the left lane
        lx = ox + 302
        for name in ["B1", "B3"]:
            y = B_GPUS[name] + GH_B / 2
            s.path(f"M{ox + GB_X0 - 1},{ty + 8} L{lx},{ty + 8} L{lx},{y} L{ox + GB_X0 - 2},{y}", ACC, 2.4, "arrAcc")


panel(0, "a2a")
s.line(433, 14, 433, 286, "#BFBFBF", 1.2)
panel(450, "token")
s.text(210, 282, "(a) all-to-all: one copy per expert", 18, PRB, weight="bold")
s.text(660, 282, "(b) token-based: one copy per node", 18, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_a2a.svg"))
