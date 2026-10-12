"""fig_layer: where the collectives sit inside one GLM-5 layer under TP + EP."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 336, "fig_layer: attention -> o_proj -> AllReduce -> norm/router -> all-to-all -> experts")

X, W, H, G = 60, 360, 34, 10
y = 8

def step(label, kind="neutral"):
    global y
    if kind == "prb":
        s.box(X, y, W, H, label, PRB_F, PRB, color=PRB, size=18, weight="bold")
    else:
        s.box(X, y, W, H, label, NEU_F, NEU, size=18)
    y += H
    s.line(X + W / 2, y, X + W / 2, y + G, INK, 2.2, "arr")
    y += G

step("Attention + o_proj")
step("AllReduce (TP)", "prb")
step("RMSNorm + router")
step("All-to-all dispatch", "prb")
step("Experts (EP)")
step("All-to-all combine", "prb")
s.box(X, y, W, H, "Next layer", "#FFFFFF", NEU, size=18, dash="6,4")

s.text(10, 326, "3 dense + 75 MoE layers", 17, MUTED, anchor="start", style="italic")
s.text(470, 326, "every token · every step", 17, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_layer.svg"))
