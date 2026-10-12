"""fig_mismatch: three ways a GPU-shaped deployment misses on a domestic NPU node."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 286, "fig_mismatch: (a) weights overflow one node (b) kernel swarm (c) exposed collective")

PANEL_Y, PANEL_H = 6, 212

# ---------------------------------------------------------------- (a) memory
s.rect(8, PANEL_Y, 250, PANEL_H, "#FFFFFF", MUTED, 1.2, 6, dash="5,4")
# capacity column
s.rect(40, 62, 80, 142, NEU_F, NEU, 2, 3)
s.text(80, 126, "node", 17, INK)
s.text(80, 148, "HBM", 17, INK)
# weights bar overflowing
s.rect(146, 28, 80, 176, PRB_F, PRB, 2.4, 3)
s.text(186, 126, "BF16", 18, PRB, weight="bold")
s.text(186, 148, "weights", 18, PRB, weight="bold")
s.line(32, 62, 240, 62, PRB, 2.2, dash="7,4")
s.text(236, 54, "capacity", 17, PRB, anchor="end", weight="bold")
s.text(133, PANEL_Y + PANEL_H + 24, "(a) Weights outgrow the node", 18, INK, weight="bold")

# ---------------------------------------------------------------- (b) kernel swarm
s.rect(272, PANEL_Y, 290, PANEL_H, "#FFFFFF", MUTED, 1.2, 6, dash="5,4")
s.text(417, PANEL_Y + 22, "MLA pre-processing", 18, INK, weight="bold")
x = 288
for i in range(8):
    s.rect(x, 54, 22, 40, NEU_F, NEU, 1.8, 2)
    x += 28
s.text(530, 80, "…", 20, INK)
s.text(417, 118, "many small kernels", 18, INK)
# idle units
s.box(292, 140, 120, 36, "Cube unit", "#FFFFFF", NEU, size=17)
s.box(424, 140, 120, 36, "Vector unit", "#FFFFFF", NEU, size=17)
s.text(417, 196, "Units wait on each other", 17, PRB, weight="bold")
s.text(417, PANEL_Y + PANEL_H + 24, "(b) Kernel swarm, idle units", 18, INK, weight="bold")

# ---------------------------------------------------------------- (c) exposed collective
s.rect(576, PANEL_Y, 286, PANEL_H, "#FFFFFF", MUTED, 1.2, 6, dash="5,4")
s.text(719, PANEL_Y + 22, "One layer, TP ranks", 18, INK, weight="bold")
s.box(592, 54, 100, 42, "Attention", NEU_F, NEU, size=17)
s.rect(700, 54, 110, 42, PRB_F, PRB, 2.4, 3)
s.text(755, 80, "AllReduce", 18, PRB, weight="bold")
s.box(592, 112, 100, 42, "MoE", NEU_F, NEU, size=17)
s.rect(700, 112, 110, 42, PRB_F, PRB, 2.4, 3)
s.text(755, 138, "AllReduce", 18, PRB, weight="bold")
s.text(719, 196, "Exposed on a weaker link", 17, PRB, weight="bold")
s.text(719, PANEL_Y + PANEL_H + 24, "(c) Collectives on the critical path", 18, INK, weight="bold")

s.save(os.path.join(HERE, "fig_mismatch.svg"))
