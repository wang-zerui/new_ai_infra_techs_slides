"""fig_flow: attention output -> reshape (no communication) -> token bags -> A2A-V over EP -> experts.

Same 8 GPUs as fig_ranks: attention TP2 CP2 DP2 (with sequence parallelism every GPU holds a
different slice of tokens), MoE ETP1 EP8. [PF] Sec. 3.3 and App. 6.2.
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 330, "fig_flow: reshape at the attention/MoE boundary, then all-to-all over the EP group")
X0, W = 184, 74
def xi(i):
    return X0 + i * 80 + (i // 2) * 6 + (i // 4) * 14
Y1, Y2, Y3, H = 30, 134, 266, 50

s.text(170, Y1 + 22, "attention out", 18, INK, anchor="end", weight="bold")
s.text(170, Y1 + 44, "sample · slice", 16, MUTED, anchor="end")
s.text(170, Y2 + 22, "token bag", 18, INK, anchor="end", weight="bold")
s.text(170, Y2 + 44, "T local tokens", 16, MUTED, anchor="end")
s.text(170, Y3 + 22, "experts", 18, ACC, anchor="end", weight="bold")
s.text(170, Y3 + 44, "EP8", 16, MUTED, anchor="end")
slices = "abcd"
for i in range(8):
    x = xi(i)
    smp, part = i // 4, slices[i % 4]
    s.box(x, Y1, W, H, f"S{smp}·{part}", NEU_F, NEU, size=17)
    s.line(x + W / 2, Y1 + H + 3, x + W / 2, Y2 - 2, ACC, 2.2, "arrAcc")
    s.box(x, Y2, W, H, f"T{i}", "#FFFFFF", NEU, size=17)
    s.box(x, Y3, W, H, f"E{i}", ACC_F, ACC, color=ACC, weight="bold", size=18)
s.text(170, Y1 + H + 22, "reshape", 18, ACC, anchor="end", weight="bold")
s.text(170, Y1 + H + 44, "no communication", 16, ACC, anchor="end")
# all-to-all-V over the EP group: every bag sends to every expert (all 64 links drawn)
for i in range(8):
    for j in range(8):
        x1 = xi(i) + 12 + 50 * j / 7
        x2 = xi(j) + 12 + 50 * i / 7
        s.line(x1, Y2 + H + 2, x2, Y3 - 2, "#7F7F7F", 1.0)
s.text(X0 - 14, (Y2 + H + Y3) / 2 + 6, "A2A-V", 17, INK, anchor="end", weight="bold")
s.save(os.path.join(HERE, "fig_flow.svg"))
