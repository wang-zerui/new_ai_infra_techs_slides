"""fig_perm: fused permutation with a worked example (4 tokens, 3 experts; t1->E0,E2; t2->E1; t3->E0,E1;
t4->E2). The row-id map (token x expert -> destination row) is built once (3 Triton passes) and reused:
the permute kernel copies each token (and its prob / FP8 scales) to its expert-major rows; after the expert
GEMMs the unpermute kernel sums each token's copies weighted by probs in FP32.
Sources: TE common/triton/permutation.py; arXiv 2603.07685 Sec. 4.3.3 (Fig. 20)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 270, "fig_perm: row-id map, permute, unpermute")

rows_map = {"t1": [0, None, 4], "t2": [None, 2, None], "t3": [1, 3, None], "t4": [None, None, 5]}
perm_rows = ["t1", "t3", "t2", "t3", "t1", "t4"]          # expert-major: E0 E0 E1 E1 E2 E2
TY = {t: 64 + i * 44 for i, t in enumerate(rows_map)}   # token row tops (h 30)
RY = [40 + i * 36 for i in range(6)]                     # permuted row tops (h 28)
CELL_X = [64, 100, 136]

# ---- row-id map table
s.text(4, 20, "row-id map (once)", 16, ACC, anchor="start", weight="bold")
for j, e in enumerate(["E0", "E1", "E2"]):
    s.text(CELL_X[j] + 16, 54, e, 15, MUTED)
for t, cells in rows_map.items():
    y = TY[t]
    s.box(4, y, 52, 30, t, NEU_F, NEU, size=16, weight="bold")
    for j, v in enumerate(cells):
        s.box(CELL_X[j], y, 32, 30, "–" if v is None else str(v), ACC_F if v is not None else "#FFFFFF",
              ACC if v is not None else "#BFBFBF", color=INK if v is not None else MUTED, size=15, sw=1.6)

# ---- permuted (expert-major) rows
MX, MW = 300, 92
s.text(MX + MW / 2, 264, "expert-major rows", 15, MUTED, style="italic")
for i, t in enumerate(perm_rows):
    s.box(MX, RY[i], MW, 28, f"row {i}: {t}", "#FFFFFF", NEU, size=15)
for k, e in enumerate(["E0", "E1", "E2"]):
    y0, y1 = RY[2 * k], RY[2 * k + 1] + 28
    s.line(MX + MW + 6, y0 + 2, MX + MW + 6, y1 - 2, INK, 1.6)
    s.text(MX + MW + 12, (y0 + y1) / 2 + 5, e, 15, INK, anchor="start", weight="bold")

# ---- permute arrows: one per non-empty row-id map cell (token copied to its destination row)
for t, cells in rows_map.items():
    for j, v in enumerate(cells):
        if v is not None:
            s.line(CELL_X[2] + 36, TY[t] + 15, MX - 2, RY[v] + 14, ACC, 1.8, "arrAcc")
s.text(236, 20, "permute kernel", 16, ACC, weight="bold")

# ---- experts
s.box(452, 40, 84, RY[5] + 28 - 40, ["expert", "GEMMs"], NEU_F, NEU, size=16)
for i in range(6):
    s.line(MX + MW + 30, RY[i] + 14, 450, RY[i] + 14, MUTED, 1.2)
    s.line(538, RY[i] + 14, 578, RY[i] + 14, MUTED, 1.2)

# ---- unpermute: weighted sum into original token order
OX = 780
for t in rows_map:
    s.box(OX, TY[t], 86, 30, f"{t} out", ACC_F, ACC, size=16, sw=1.8)
for i, t in enumerate(perm_rows):
    s.line(580, RY[i] + 14, OX - 2, TY[t] + 15, ACC, 1.8, "arrAcc")
s.text(680, 20, "unpermute kernel", 16, ACC, weight="bold")
s.text(680, 252, "Σ prob · y, FP32 accumulate", 15, ACC)
s.save(os.path.join(HERE, "fig_perm.svg"))
