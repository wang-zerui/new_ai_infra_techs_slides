"""fig_coarse: (a) dense FFN, every weight on every token; (b) coarse MoE, few big experts mix topics and
duplicate common knowledge. Glyphs: {} code, Σ math, Aa prose; red hatch = common knowledge copy."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_coarse: dense FFN vs coarse top-2 MoE")

GLYPHS = ["{ }", "Σ", "Aa"]

def expert(x, y, w, h, lit, common_h=34, gsize=18):
    """tall expert box: topic glyphs on top, hatched common-knowledge band at the bottom"""
    fill = PRB_F if lit else "#FFFFFF"
    stroke = PRB if lit else NEU
    s.rect(x, y, w, h, fill, stroke, 2.4 if lit else 1.6, 3)
    s.add(f'<rect x="{x + 1.5:.1f}" y="{y + h - common_h:.1f}" width="{w - 3:.1f}" height="{common_h - 1.5:.1f}" '
          f'fill="url(#hatchPrb)" stroke="none"/>')
    s.line(x, y + h - common_h, x + w, y + h - common_h, stroke, 1.2)
    step = (h - common_h) / 3
    for i, g in enumerate(GLYPHS):
        s.text(x + w / 2, y + step * i + step / 2 + 0.35 * gsize, g, gsize, INK if lit else MUTED,
               weight="bold" if lit else "normal")

# ---------------- panel (a) dense FFN
s.text(160, 22, "(a) Dense FFN", 20, INK, weight="bold")
s.box(20, 116, 70, 44, "Token", NEU_F, NEU)
s.line(92, 138, 128, 138, INK, 2.2, "arr")
expert(130, 52, 170, 172, True, common_h=40, gsize=22)
s.text(215, 254, "All weights active", 19, PRB, weight="bold")
s.text(215, 278, "on every token", 19, PRB, weight="bold")

# divider
s.line(330, 10, 330, H - 10, "#BFBFBF", 1.2)

# ---------------- panel (b) coarse MoE, top-2 of 16 (8 shown)
s.text(600, 22, "(b) Coarse MoE: top-2 of 16", 20, INK, weight="bold")
s.box(350, 116, 70, 44, "Token", NEU_F, NEU)
s.line(422, 138, 446, 138, INK, 2.2, "arr")
s.box(448, 108, 76, 60, "Router", "#FFFFFF", NEU)
EX0, EW, EG, EY, EH = 548, 30, 8, 60, 168
lit = {1, 5}
for i in range(7):
    x = EX0 + i * (EW + EG)
    expert(x, EY, EW, EH, i in lit, common_h=34, gsize=15)
s.text(EX0 + 7 * (EW + EG) + 6, EY + EH / 2 + 6, "…", 22, MUTED, anchor="start")
# routing arrows to the two lit experts: one trunk above the experts, a branch down to each lit one
xs = [EX0 + i * (EW + EG) + EW / 2 for i in sorted(lit)]
s.path(f"M486,106 V42 H{xs[-1]} V{EY - 3}", PRB, 2.2, "arrPrb")
for xc in xs[:-1]:
    s.line(xc, 42, xc, EY - 3, PRB, 2.2, "arrPrb")
# labels under panel (b)
s.text(548, 254, "Each expert mixes topics", 19, PRB, anchor="start", weight="bold")
# legend for hatch
s.add(f'<rect x="548" y="263" width="26" height="18" fill="url(#hatchPrb)" stroke="{PRB}" stroke-width="1.2"/>')
s.text(582, 278, "common knowledge copy", 18, INK, anchor="start")
s.save(os.path.join(HERE, "fig_coarse.svg"))
