"""Hook figure: where CSA sits in the V4 layer stacks (layer types from config.json compress_ratios)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
PRO = ["HCA", "HCA"] + ["CSA", "HCA"] * 29 + ["CSA"]          # 61 layers: 30 CSA + 31 HCA
FLASH = ["SWA", "SWA"] + ["CSA", "HCA"] * 20 + ["CSA"]        # 43 layers: 21 CSA + 20 HCA + 2 SWA
assert len(PRO) == 61 and PRO.count("CSA") == 30 and PRO.count("HCA") == 31
assert len(FLASH) == 43 and FLASH.count("CSA") == 21 and FLASH.count("HCA") == 20

W, H = 870, 290
s = SVG(W, H)
X0, CW, GAP, CH = 196, 64, 6, 50
STYLE = {"CSA": (ACC_FILL, ACC, ACC, "bold"), "HCA": (GRAY_FILL, GRAY, INK, "normal"),
         "SWA": ("#fff", GRAY, INK, "normal")}

# legend strip
lx = 196
for kind, label in [("CSA", "CSA: this deck"), ("HCA", "HCA: deck 03"), ("SWA", "window only")]:
    f, st, c, wt = STYLE[kind]
    s.rect(lx, 6, 26, 22, f, st, 2, 4, "6,4" if kind == "SWA" else None)
    s.text(lx + 34, 24, label, 18, "normal", INK, "start")
    lx += 34 + len(label) * 9.6 + 34


def row(y, name, sub, kinds, total, count_text):
    s.text(0, y + 22, name, 20, "bold", INK, "start")
    s.text(0, y + 46, sub, 18, "normal", GRAY, "start")
    show = list(range(8)) + [total - 1]
    x = X0
    for j, li in enumerate(show):
        if j == 8:
            s.text(x + 18, y + 32, "…", 22, "bold", GRAY)
            x += 42
        k = kinds[li]
        f, st, c, wt = STYLE[k]
        s.rect(x, y, CW, CH, f, st, 2.5 if k == "CSA" else 2, 5, "6,4" if k == "SWA" else None)
        s.text(x + CW / 2, y + 32, k, 18, wt, c)
        s.text(x + CW / 2, y - 8, f"L{li}", 15, "normal", GRAY)
        x += CW + GAP
    s.text(X0, y + CH + 30, count_text, 18, "normal", INK, "start")


row(66, "V4-Pro", "61 layers", PRO, 61, "30 CSA layers  +  31 HCA layers  (layers 0-1 are HCA)")
row(186, "V4-Flash", "43 layers", FLASH, 43, "21 CSA layers  +  20 HCA layers  +  2 window-only layers")
s.save(os.path.join(HERE, "fig_stack.svg"))
