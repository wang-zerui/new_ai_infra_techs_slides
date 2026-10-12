"""fig_k2: Kimi K2 layout. One fixed model-parallel unit (16-way PP x 16-way EP = 256 GPUs = 32 nodes);
ZeRO-1 DP replicates the unit and shards optimizer states across replicas; at 32 nodes some optimizer
state goes to CPU (K2 report §2.4.2)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_k2")

# ---- left: one model-parallel unit, 16 x 16 cells
s.text(150, 22, "one model-parallel unit", 19, INK, weight="bold")
C, G, X0, Y0 = 10, 2, 60, 56
for r in range(16):
    for c in range(16):
        s.rect(X0 + c * (C + G), Y0 + r * (C + G), C, C, ACC_F, ACC, 0.9, 1)
W = 16 * (C + G) - G
s.text(X0 + W / 2, Y0 - 10, "EP 16 →", 17, ACC, weight="bold")
s.add(f'<text x="{X0 - 14}" y="{Y0 + W / 2}" text-anchor="middle" font-size="17" font-weight="bold" fill="{ACC}" '
      f'transform="rotate(-90 {X0 - 14} {Y0 + W / 2})">PP 16 (virtual stages) →</text>')
s.text(X0 + W / 2, Y0 + W + 24, "256 GPUs = 32 nodes", 17, INK)
s.text(X0 + W / 2, Y0 + W + 45, "same at every scale", 17, MUTED, style="italic")

s.line(300, 14, 300, 290, "#BFBFBF", 1.5)

# ---- right: DP replicas of the unit
rows = [("32 nodes", 1, "opt", True), ("64 nodes", 2, "opt ½", False), ("32k nodes", 4, "opt 1/k", False)]
UY = [40, 128, 216]
for (lab, k, opt, cpu), y in zip(rows, UY):
    s.text(316, y + 32, lab, 18, INK, anchor="start", weight="bold")
    for i in range(k):
        x = 430 + i * 92
        if k == 4 and i == 2:
            s.text(x + 30, y + 32, "…", 22, INK)
            continue
        s.box(x, y, 62, 46, "PP×EP", ACC_F, ACC, ACC, 16, "bold")
        s.box(x, y + 50, 62, 22, opt, "#FFFFFF", ACC, ACC, 15, "normal", 1.6, 3)
    if cpu:
        s.box(560, y + 6, 130, 60, ["part of opt", "on CPU"], NEU_F, NEU, INK, 16)
        s.line(494, y + 61, 558, y + 50, MUTED, 1.8, "arrGray", dash="4,3")
s.text(735, 172, "ZeRO-1 DP", 18, ACC, weight="bold")
s.text(735, 196, "grows or shrinks", 17, ACC)
s.save(os.path.join(HERE, "fig_k2.svg"))
