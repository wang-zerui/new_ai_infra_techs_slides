"""fig_shared: shared expert isolation (DeepSeekMoE §3.2). (a) 4 routed experts each keep a copy of common
knowledge (red hatch); (b) 1 shared expert (always on) holds it once, routed top-k drops from 4 to 3 so the
number of active experts (FLOPs) is unchanged."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_shared: routed-only top-4 vs shared + top-3")
s.add(f'''<defs><pattern id="hatchAcc" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
  <rect width="7" height="7" fill="{ACC_F}"/><line x1="0" y1="0" x2="0" y2="7" stroke="{ACC}" stroke-width="2.2"/></pattern></defs>''')

EY, EH, EW, GAP, CB = 108, 124, 64, 26, 42   # expert top, height, width, gap, common-band height

def routed(x, glyph, band):
    s.rect(x, EY, EW, EH, "#FFFFFF", NEU, 2, 3)
    top_h = EH - (CB if band else 0)
    s.text(x + EW / 2, EY + top_h / 2 + 7, glyph, 21, INK, weight="bold")
    if band:
        s.add(f'<rect x="{x + 1.5}" y="{EY + EH - CB}" width="{EW - 3}" height="{CB - 1.5}" fill="url(#hatchPrb)"/>')
        s.line(x, EY + EH - CB, x + EW, EY + EH - CB, NEU, 1.2)
    s.text(x + EW / 2, EY + EH + 22, "routed", 16, MUTED)

def fan(tx, xs, color=INK, marker="arr"):
    for xc in xs:
        s.line(tx, 82, xc, EY - 3, color, 2.0, marker)

# ---------------- (a) routed only, top-4
s.text(205, 22, "(a) Routed only: top-4", 20, INK, weight="bold")
s.box(160, 38, 90, 42, "Token", NEU_F, NEU)
xa = [38 + i * (EW + GAP) for i in range(4)]
fan(205, [x + EW / 2 for x in xa])
for x, g in zip(xa, ["{ }", "Σ", "Aa", "文"]):
    routed(x, g, True)
s.text(205, 282, "common knowledge × 4", 19, PRB, weight="bold")

s.line(430, 10, 430, H - 10, "#BFBFBF", 1.2)

# ---------------- (b) shared + top-3 routed
s.text(655, 22, "(b) 1 shared + top-3 routed", 20, ACC, weight="bold")
s.box(610, 38, 90, 42, "Token", NEU_F, NEU)
xb = [488 + i * (EW + GAP) for i in range(4)]
# shared expert
s.add(f'<rect x="{xb[0]}" y="{EY}" width="{EW}" height="{EH}" rx="3" fill="url(#hatchAcc)" stroke="{ACC}" stroke-width="2.6"/>')
s.text(xb[0] + EW / 2, EY + EH + 22, "shared", 16, ACC, weight="bold")
s.line(655, 82, xb[0] + EW / 2, EY - 3, ACC, 2.4, "arrAcc")
fan(655, [x + EW / 2 for x in xb[1:]])
for x, g in zip(xb[1:], ["{ }", "Σ", "Aa"]):
    routed(x, g, False)
s.text(655, 282, "common knowledge × 1", 19, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_shared.svg"))
