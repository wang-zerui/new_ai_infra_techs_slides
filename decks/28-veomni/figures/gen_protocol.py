"""fig_protocol: omni protocol in training; encoder/decoder lm_encode outputs are masked-scattered into the
foundation's input embeddings, losses come from the LM head and the decoder's lm_head (VeOmni §3.1, Listing 3)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 296, "fig_protocol")
MONO = "Menlo, monospace"

X0, SW, SG, SY, SH = 176, 48, 6, 124, 46
kinds = ["txt", "txt", "img", "img", "img", "img", "txt", "txt", "gen", "gen", "gen", "txt"]
def sx(i):
    return X0 + i * (SW + SG)
for i, k in enumerate(kinds):
    if k == "txt":
        s.box(sx(i), SY, SW, SH, k, "#FFFFFF", NEU, INK, 16)
    else:
        s.box(sx(i), SY, SW, SH, k, ACC_F, ACC, ACC, 16, "bold")
END = sx(11) + SW

# source boxes (encoder / decoder lm_encode)
def source(i0, i1, label):
    xa, xb = sx(i0), sx(i1) + SW
    s.rect(xa, 8, xb - xa, 54, ACC_F, ACC, 2, 4)
    s.text((xa + xb) / 2, 30, label, 18, ACC, weight="bold")
    s.text((xa + xb) / 2, 53, "lm_encode()", 16, ACC, family=MONO)
    for i in range(i0, i1 + 1):
        cx = sx(i) + SW / 2
        s.line(cx, 62, cx, SY - 2, ACC, 2.2, "arrAcc")
source(2, 5, "ViT encoder")
source(8, 10, "VQ decoder")

# left labels
s.text(164, 41, "modules", 18, INK, anchor="end", weight="bold")
s.text(164, 97, "masked_scatter", 16, ACC, anchor="end", family=MONO)
s.text(164, SY + SH / 2 + 6, "input embeds", 18, INK, anchor="end", weight="bold")

# foundation
FY, FH = 194, 40
s.box(X0, FY, END - X0, FH, "Foundation LLM forward", NEU_F, NEU, INK, 18, "bold")
s.line((X0 + END) / 2, SY + SH + 2, (X0 + END) / 2, FY - 2, INK, 2.2, "arr")

# losses
OY, OH = 252, 36
s.box(sx(0), OY, sx(6) + SW - sx(0), OH, "LM head: text loss", "#FFFFFF", NEU, INK, 17)
s.box(sx(8), OY, END - sx(8), OH, "VQ lm_head: image loss", ACC_F, ACC, ACC, 17, "bold")
s.line((sx(0) + sx(6) + SW) / 2, FY + FH + 2, (sx(0) + sx(6) + SW) / 2, OY - 2, INK, 2.2, "arr")
s.line((sx(8) + END) / 2, FY + FH + 2, (sx(8) + END) / 2, OY - 2, ACC, 2.2, "arrAcc")
s.text(164, OY + OH / 2 + 6, "losses", 18, INK, anchor="end", weight="bold")
s.save(os.path.join(HERE, "fig_protocol.svg"))
