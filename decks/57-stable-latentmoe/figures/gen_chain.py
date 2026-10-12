"""fig_chain: the vanilla LatentMoE routed path at K3 shapes. Four consecutive matmuls, an unbounded SwiGLU
product and an aggregation whose scale depends on routing, with nothing normalizing it before W_up
(K3 §2.3 p.6-7; LatentMoE §3)."""
import os
from svgkit import SVG, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 248, "fig_chain: vanilla latent routed path, problems in red")
I = "italic"
CY = 96          # centre line
H = 64


def gbox(x, w, l1, l2, n):
    s.rect(x, CY - H / 2, w, H, NEU_F, NEU, 2, 5)
    s.rich(x + w / 2, CY - 4, l1, 20)
    s.text(x + w / 2, CY + 20, l2, 17, MUTED)
    # matmul index
    s.circle(x + w / 2, CY - H / 2 - 18, 12, "#FFFFFF", MUTED, 1.6)
    s.text(x + w / 2, CY - H / 2 - 12, str(n), 16, MUTED, weight="bold")


# input
s.rich(30, CY + 7, [("x", INK, "bold", I)], 22)
s.line(46, CY, 74, CY, INK, 2.2, "arr")
gbox(76, 116, [("W", INK, "bold"), ("↓", INK, "bold")], "7168 → 3584", 1)
# expert container (one of 16), with stack effect
EX, EW = 254, 316
for off in (8, 4):
    s.rect(EX + off, CY - H / 2 - 42 - off, EW, H + 60, "#FFFFFF", "#A6A6A6", 1.4, 6)
s.rect(EX, CY - H / 2 - 42, EW, H + 60, "#FFFFFF", NEU, 1.8, 6, dash="6,4")
s.text(EX + EW - 8, CY + H / 2 + 34, "16 of 896 experts", 17, MUTED, anchor="end", style="italic")
s.line(192 + 2, CY, EX - 1, CY, INK, 2.2, "arr")
s.text((192 + EX) / 2 - 3, CY - 12, "dispatch", 16, MUTED, style="italic")
gbox(EX + 12, 112, [("W", INK, "bold"), ("g", INK, "bold", "sub"), (", W", INK, "bold"), ("u", INK, "bold", "sub")],
     "3584 → 3072", 2)
# SwiGLU product node (problem)
NX = EX + 12 + 112 + 34
s.circle(NX, CY, 17, PRB_F, PRB, 2.2)
s.text(NX, CY + 7, "×", 22, PRB, weight="bold")
s.line(EX + 12 + 112 + 2, CY, NX - 18, CY, INK, 2.2, "arr")
gbox(NX + 30, 112, [("W", INK, "bold"), ("d", INK, "bold", "sub")], "3072 → 3584", 3)
s.line(NX + 18, CY, NX + 29, CY, INK, 2.2, "arr")
# aggregation (problem: scale depends on routing)
SX = EX + EW + 70
s.line(EX + EW + 10, CY, SX - 22, CY, INK, 2.2, "arr")
s.circle(SX, CY, 21, PRB_F, PRB, 2.2)
s.text(SX, CY + 8, "Σ", 22, PRB, weight="bold")
s.text((EX + EW + 8 + SX - 21) / 2 + 7, CY - 12, "combine", 16, MUTED, style="italic")
# gap with no normalization, then W_up
UX = SX + 60
s.line(SX + 22, CY, UX - 1, CY, PRB, 2.4, "arrPrb", dash="5,4")
gbox(UX, 118, [("W", INK, "bold"), ("↑", INK, "bold")], "3584 → 7168", 4)
s.line(UX + 118 + 2, CY, 866, CY, INK, 2.2, "arr")

# problem callouts (red, short)
s.line(NX, CY + 18, NX, 188, PRB, 1.8)
s.text(NX, 210, "SwiGLU: unbounded", 18, PRB, weight="bold")
s.text(NX, 232, "× unbounded", 18, PRB, weight="bold")
s.line(SX, CY + 22, SX, 188, PRB, 1.8)
s.text(SX + 4, 210, "scale depends on", 18, PRB, weight="bold")
s.text(SX + 4, 232, "which experts fire", 18, PRB, weight="bold")
s.text((SX + UX) / 2 + 2, CY - 42, "no norm", 17, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_chain.svg"))
