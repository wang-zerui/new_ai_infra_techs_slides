"""fig_indexer: what the Lightning Indexer super-operator absorbs.

score (q_idx x K_idx) -> ReLU -> per-head weighted sum -> TopK, in one kernel; the 32 x S score
matrix is the intermediate that never reaches HBM. Shapes: 32 index heads of dim 128
(GLM-5 Table 10), k = 2048 (GLM-5 DSA section).
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_indexer: Lightning Indexer fuses score, ReLU, aggregation and TopK")

CX0, CX1, CY0, CY1 = 150, 762, 26, 212
YM = 127                                   # mid-line of the data path

# input query
s.box(10, YM - 29, 126, 58, ["q index", "32 × 128"], "#FFFFFF", NEU, size=17)
s.line(136, YM, 172, YM, INK, 2.2, "arr")

# container
s.rect(CX0, CY0, CX1 - CX0, CY1 - CY0, ACC_F, ACC, 3, 6)
s.text(CX0 + 14, CY0 + 22, "❷ Lightning Indexer  ·  one kernel", 19, ACC, anchor="start",
       weight="bold")

# matmul operator node
s.add(f'<circle cx="194" cy="{YM}" r="17" fill="#FFFFFF" stroke="{ACC}" stroke-width="2.5"/>')
s.text(194, YM + 7, "×", 22, ACC, weight="bold")
s.line(211, YM, 230, YM, INK, 2.2, "arr")

# score matrix as a cell grid
GX, GY, CW, CHH = 232, 80, 18, 19
for r in range(5):
    for c in range(7):
        s.rect(GX + c * CW, GY + r * CHH, CW - 2, CHH - 2, "#FFFFFF", "#9DC3E6", 1.2, 2)
s.text(GX + 3.5 * CW, GY - 8, "never written to HBM", 16, ACC, weight="bold")
s.text(GX + 3.5 * CW, GY + 5 * CHH + 20, "scores  32 × S", 16, INK)
s.line(GX + 7 * CW - 2, YM, GX + 7 * CW + 20, YM, INK, 2.2, "arr")

# ReLU, weighted sum, TopK
s.box(380, YM - 25, 78, 50, ["ReLU"], "#FFFFFF", ACC, color=INK, size=17)
s.line(458, YM, 478, YM, INK, 2.2, "arr")
s.box(480, YM - 25, 140, 50, ["sum over", "32 heads"], "#FFFFFF", ACC, color=INK, size=17)
s.line(620, YM, 640, YM, INK, 2.2, "arr")
s.box(642, YM - 25, 102, 50, ["TopK", "k = 2048"], "#FFFFFF", ACC, color=INK, size=17)

# output
s.line(CX1, YM, CX1 + 26, YM, ACC, 2.6, "arrAcc")
s.text(812, YM - 6, "top-2048", 16, ACC, weight="bold")
s.text(812, YM + 16, "token ids", 16, ACC, weight="bold")

# key cache
s.box(150, 238, 268, 46, ["index key cache   S × 128"], NEU_F, NEU, size=16)
s.line(194, 238, 194, YM + 20, MUTED, 2.2, "arrGray")
s.text(204, 230, "streamed once", 16, MUTED, anchor="start")
s.save(os.path.join(HERE, "fig_indexer.svg"))
