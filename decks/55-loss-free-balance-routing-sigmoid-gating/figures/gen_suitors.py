"""fig_suitors: (a) aux balance loss pushes an interfering gradient into the router;
(b) Expert Choice balances by letting experts pick tokens across the chunk, so future tokens steer earlier routes."""
import os
from svgkit import SVG, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_suitors: aux loss (gradient interference) vs expert choice (future-token leakage)")

# ---------------- panel (a): auxiliary balance loss
TOP, BH = 40, 50
s.box(20, TOP, 140, BH, "Router", NEU_F, NEU)
s.box(200, TOP, 100, BH, "Top-K", NEU_F, NEU)
s.box(330, TOP, 88, BH, "Experts", NEU_F, NEU)
s.line(162, TOP + BH / 2, 198, TOP + BH / 2, INK, 2.2, "arr")
s.line(302, TOP + BH / 2, 328, TOP + BH / 2, INK, 2.2, "arr")
LY = 196
s.box(10, LY, 140, BH, "LM loss", "#FFFFFF", NEU)
s.box(186, LY, 230, BH, "α · balance loss", PRB_F, PRB, color=PRB, weight="bold")
# LM gradient into the router
s.line(60, LY - 2, 60, TOP + BH + 2, MUTED, 2.2, "arrGray")
s.text(70, 166, "LM gradient", 16, MUTED, anchor="start", style="italic")
# balance-loss gradient into the router (the interference)
s.path(f"M 300 {LY - 2} L 300 126 L 130 126 L 130 {TOP + BH + 2}", PRB, 2.5, "arrPrb")
s.text(310, 154, "interference", 17, PRB, anchor="start", weight="bold")
s.text(310, 174, "gradient", 17, PRB, anchor="start", weight="bold")
s.text(214, 282, "(a) Auxiliary balance loss", 19, INK, weight="bold")

# divider
s.line(438, 18, 438, 290, "#BFBFBF", 1.2)

# ---------------- panel (b): Expert Choice
GX, CW, CH, GAP = 548, 46, 38, 4
rows = ["E1", "E2", "E3"]
picks = {0: (1, 4), 1: (0, 3), 2: (2, 5)}          # each expert picks its top-2 tokens of the chunk
GY = 66
for c in range(6):
    s.text(GX + c * CW + CW / 2, 58, f"t{c + 1}", 17, INK)
for r, name in enumerate(rows):
    y = GY + r * (CH + GAP)
    s.text(GX - 12, y + CH / 2 + 6.3, name, 18, INK, anchor="end")
    for c in range(6):
        x = GX + c * CW
        sel = c in picks[r]
        hot = (r == 0 and c == 1)
        s.rect(x + 2, y, CW - 4, CH, "#BFBFBF" if sel else "#FFFFFF", PRB if hot else "#7F7F7F",
               3 if hot else 1.4, 3)
# future token t5 decides whether E1 still has room for t2
x2, x5 = GX + 1 * CW + CW / 2, GX + 4 * CW + CW / 2
s.path(f"M {x5} 40 Q {(x2 + x5) / 2} 2 {x2 + 2} 40", PRB, 2.5, "arrPrb")
s.text(478, 104, "experts", 16, MUTED, style="italic")
s.text(478, 124, "pick", 16, MUTED, style="italic")
s.text(478, 144, "tokens", 16, MUTED, style="italic")
s.line(GX, 210, GX + 6 * CW, 210, INK, 2, "arr")
s.text(GX + 6 * CW + 8, 216, "time", 16, MUTED, anchor="start", style="italic")
s.text(GX + 3 * CW, 246, "t2's route depends on future t5", 17, PRB, weight="bold")
s.text(654, 282, "(b) Expert Choice routing", 19, INK, weight="bold")
s.save(os.path.join(HERE, "fig_suitors.svg"))
