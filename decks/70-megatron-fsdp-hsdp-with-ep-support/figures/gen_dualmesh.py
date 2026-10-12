"""fig_dualmesh: 8 GPUs, TP=1. (a) dense params: FSDP over all 8 DP ranks. (b) experts: EP=4 places expert
pairs, then FSDP shards each pair over its EDP=2 group (GPU g and g+4)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_dualmesh: dense mesh (DP=8) vs expert mesh (EP=4 x EDP=2)")

# ---------------- (a) dense mesh
s.text(10, 24, "(a) Dense mesh: attention and norms", 19, INK, anchor="start", weight="bold")
X0, CW, Y = 20, 46, 96
for g in range(8):
    x = X0 + g * (CW + 4)
    s.rect(x, Y, CW, 44, ACC_F, ACC, 1.8, 3)
    s.text(x + CW / 2, Y + 28, "⅛", 18, ACC, weight="bold")
    s.text(x + CW / 2, Y + 66, f"GPU{g}", 15, MUTED)
xe = X0 + 8 * (CW + 4) - 4
s.line(X0, Y + 92, xe, Y + 92, ACC, 3)
for g in range(8):
    s.line(X0 + g * (CW + 4) + CW / 2, Y + 74, X0 + g * (CW + 4) + CW / 2, Y + 92, ACC, 2)
s.text((X0 + xe) / 2, Y + 118, "AG and RS over all 8 ranks", 17, ACC, weight="bold")

s.line(432, 12, 432, 290, "#BFBFBF", 1.2)

# ---------------- (b) expert mesh
s.text(448, 24, "(b) Expert mesh: EP 4 × EDP 2", 19, INK, anchor="start", weight="bold")
EX0, EW, EH = 520, 80, 50
rows = [(66, "node 0"), (184, "node 1")]
pairs = ["E0 E1", "E2 E3", "E4 E5", "E6 E7"]
for r, (y, lab) in enumerate(rows):
    s.text(EX0 - 10, y + 31, f"copy {r}", 16, INK, anchor="end", weight="bold")
    for c in range(4):
        x = EX0 + c * (EW + 6)
        s.box(x, y, EW, EH, [pairs[c], "half"], ACC_F, ACC, ACC, 16, "bold")
        s.text(x + EW / 2, y + EH + 18 if r == 1 else y - 8, f"GPU{c + 4 * r}", 15, MUTED)
for c in range(4):
    x = EX0 + c * (EW + 6) + EW / 2
    s.line(x, 66 + EH + 4, x, 184 - 4, ACC, 2.4, "arrAcc", start_marker="arrAcc")
s.text(EX0 + 2 * (EW + 6) - 3, 284, "AG and RS within each column: 2 ranks", 17, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_dualmesh.svg"))
