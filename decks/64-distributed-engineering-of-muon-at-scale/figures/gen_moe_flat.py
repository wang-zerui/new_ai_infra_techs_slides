"""fig_moe_flat: V4 MoE-parameter buckets (§3.4.1). All experts' down projections (all layers), then up, then gate,
flattened; padded so the vector splits evenly over ranks without cutting a matrix; consecutive same-shape matrices
merge into one batched Newton-Schulz call. Cell counts are illustrative (31 matrices + 1 pad over 4 ranks)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 226
s = SVG(W, H, "fig_moe_flat")
X0, CW, Y, CH = 54, 24, 62, 50
kinds = ["d"] * 11 + ["u"] * 10 + ["g"] * 10 + ["p"]
fill = {"d": ACC_F, "u": "#FFFFFF", "g": "#D9D9D9"}
s.text(4, 20, "expert matrices of all layers, flattened (illustrative count)", 18, MUTED, anchor="start", style="italic")
runs = [("down · 7168×3072", 0, 11), ("up · 3072×7168", 11, 21), ("gate · 3072×7168", 21, 31)]
for lab, a, b in runs:
    s.text(X0 + (a + b) / 2 * CW, Y - 14, lab, 17, INK, weight="bold")
for i, k in enumerate(kinds):
    x = X0 + i * CW
    if k == "p":
        s.add(f'<rect x="{x}" y="{Y}" width="{CW}" height="{CH}" fill="url(#hatchPrb)" stroke="{PRB}" stroke-width="1.4"/>')
    else:
        s.rect(x, Y, CW, CH, fill[k], NEU, 1.2, 0)
s.rect(X0, Y, 32 * CW, CH, "none", NEU, 1.8, 0)
# rank cuts (only between matrices)
for c in (8, 16, 24):
    x = X0 + c * CW
    s.line(x, Y, x, Y + CH + 14, ACC, 2.6, dash="7,4")
notes = [("1 NS batch", "down ×8"), ("2 NS batches", "down ×3 · up ×5"), ("1 NS batch", "up ×5 · gate ×3"),
         ("1 NS batch", "gate ×7 · pad")]
for r in range(4):
    a, b = X0 + r * 8 * CW + 4, X0 + (r + 1) * 8 * CW - 4
    yb = Y + CH + 14
    s.path(f"M{a},{yb - 6} V{yb} H{b} V{yb - 6}", INK, 1.8)
    xc = (a + b) / 2
    s.text(xc, yb + 26, f"R{r}", 19, INK, weight="bold")
    s.text(xc, yb + 52, notes[r][0], 18, ACC, weight="bold")
    s.text(xc, yb + 76, notes[r][1], 17, MUTED)
s.save(os.path.join(HERE, "fig_moe_flat.svg"))
