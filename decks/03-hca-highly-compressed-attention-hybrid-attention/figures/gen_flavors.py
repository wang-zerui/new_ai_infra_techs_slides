"""Hook: what one query reads in each of V4's three attention flavours (schematic, not to scale)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282
s = SVG(W, H)
TX0, TX1 = 236, 828          # context track; query at the right end
WIN_W = 96                   # raw window segment (schematic)
DARK = "#7F7F7F"


def label(y, name, desc, color):
    s.text(0, y + 14, name, 20, "bold", color, "start")
    s.text(0, y + 40, desc, 18, "normal", color, "start")


def window(y, color, fill="#fff"):
    n = 8
    w = WIN_W / n
    for j in range(n):
        s.rect(TX1 - WIN_W + j * w + 1, y, w - 3, 40, fill, color, 1.8, 1)


# row 1: sliding window only
y1 = 14
label(y1, "SWA", "last 128 raw", INK)
s.rect(TX0, y1, TX1 - WIN_W - TX0 - 6, 40, "#fff", "#BFBFBF", 1.5, 3, "6,5")
window(y1, GRAY)

# row 2: CSA (deck 01): small blocks, a few selected
y2 = 112
label(y2, "CSA (deck 01)", "top-k of n/4 + 128", INK)
bw, gap = 14, 3
nb = int((TX1 - WIN_W - 6 - TX0) // (bw + gap))
for i in range(nb):
    sel = i % 7 in (1, 4)
    s.rect(TX0 + i * (bw + gap), y2, bw, 40, DARK if sel else "#fff", GRAY, 1.5, 2)
window(y2, GRAY)

# row 3: HCA (this deck): large blocks, all read
y3 = 210
label(y3, "HCA (this deck)", "all n/128 + 128", ACC)
nb3, g3 = 7, 6
bw3 = (TX1 - WIN_W - 6 - TX0 - (nb3 - 1) * g3) / nb3
for i in range(nb3):
    s.rect(TX0 + i * (bw3 + g3), y3, bw3, 40, ACC_FILL, ACC, 2.5, 4)
window(y3, ACC, ACC_FILL)

# query marker
s.arrow(TX1 + 14, 4, TX1 + 14, 262, INK, 2, head=False)
s.text(TX1 + 10, 278, "query", 16, "bold", INK, "end")
s.save(os.path.join(HERE, "fig_flavors.svg"))
