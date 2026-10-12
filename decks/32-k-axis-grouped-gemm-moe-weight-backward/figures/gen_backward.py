"""fig_backward: the three GEMMs of one expert linear layer in training, which axis is ragged in each, and which
DeepGEMM grouped kernel serves it. Wgrad is the one whose ragged axis is the reduction axis."""
import os
from tilekit import SVG, seg, outline, mono, EXP, INK, MUTED, ACC, ACC_F, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 262, "fig_backward: Fprop / Dgrad / Wgrad of one expert")

rows = [("Fprop", "Y = X Wᵀ", "M rows", "m_grouped nt", False),
        ("Dgrad", "dX = dY W", "M rows", "m_grouped nn", False),
        ("Wgrad", "dW = dYᵀ X", "K (summed)", "k_grouped", True)]
XC = [8, 92, 236, 340]          # column x
s.text(XC[0] + 4, 26, "GEMM", 16, MUTED, anchor="start", weight="bold")
s.text(XC[1] + 4, 26, "per expert e", 16, MUTED, anchor="start", weight="bold")
s.text(XC[2] + 4, 26, "ragged axis", 16, MUTED, anchor="start", weight="bold")
s.text(XC[3] + 4, 26, "DeepGEMM", 16, MUTED, anchor="start", weight="bold")
for i, (name, eq, rag, api, hot) in enumerate(rows):
    y = 40 + i * 74
    if hot:
        s.rect(4, y - 4, 472, 66, ACC_F, ACC, 2, rx=6)
    s.text(XC[0] + 4, y + 36, name, 18, ACC if hot else INK, anchor="start", weight="bold")
    mono(s, XC[1] + 4, y + 36, eq, 16, INK, anchor="start")
    # ragged-axis glyph: three expert segments stacked (rows) or laid along the reduction
    gx = XC[2] + 4
    if hot:
        for e, w in enumerate([30, 10, 20]):
            seg(s, gx, y + 12, w, 18, e, None, sw=1)
            gx += w
        s.text(XC[2] + 4, y + 50, rag, 15, ACC, anchor="start", weight="bold")
    else:
        yy = y + 14
        for e, h in enumerate([16, 6, 11]):
            seg(s, gx, yy, 30, h, e, None, sw=1)
            yy += h
        s.text(XC[2] + 40, y + 36, rag, 15, INK, anchor="start")
    mono(s, XC[3] + 4, y + 36, api, 15, ACC if hot else INK, anchor="start", weight="bold" if hot else "normal")
s.save(os.path.join(HERE, "fig_backward.svg"))
