"""fig_dynbsz: fixed sample count vs token budget (VeOmni dyn_bsz). Samples S1..S4 = 4, 2, 3, 1 tokens.
(a) 2 samples per rank -> 6 vs 4 tokens (docs example); (b) budget 5 tokens, first-fit from the buffer ->
rank 0 gets S1 + S4, rank 1 gets S2 + S3."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 268, "fig_dynbsz")
S = {"S1": "ABCD", "S2": "EF", "S3": "GHI", "S4": "J"}
FILL = {"S1": NEU_F, "S2": "#FFFFFF", "S3": NEU_F, "S4": "#FFFFFF"}
CW, CG, SG = 30, 3, 10

def group(x, y, name, show_name=False):
    for i, t in enumerate(S[name]):
        s.box(x + i * (CW + CG), y, CW, 30, t, FILL[name], NEU, INK, 16, "bold", 1.5, 3)
    w = len(S[name]) * (CW + CG) - CG
    if show_name:
        s.text(x + w / 2, y + 50, name, 16, MUTED)
    return x + w + SG

# buffer
s.text(130, 41, "samples", 18, INK, anchor="end", weight="bold")
x = 145
for n in ["S1", "S2", "S3", "S4"]:
    x = group(x, 20, n, show_name=True)


def panel(x0, title, color, ranks, idle=None):
    s.text(x0, 112, title, 19, color, anchor="start", weight="bold")
    for r, names in enumerate(ranks):
        y = 132 + r * 54
        s.text(x0 + 70, y + 21, f"rank {r}", 17, INK, anchor="end", weight="bold")
        x = x0 + 82
        for n in names:
            x = group(x, y, n)
        if idle and idle[r]:
            w = idle[r] * (CW + CG) - CG
            s.rect(x - SG + 6, y, w, 30, "url(#hatchPrb)", PRB, 1.6, 3)
            s.text(x - SG + 6 + w + 8, y + 21, "idle", 16, PRB, anchor="start", weight="bold")

panel(10, "(a) 2 samples per rank", PRB, [["S1", "S2"], ["S3", "S4"]], idle=[0, 2])
s.text(220, 252, "6 vs 4 tokens: rank 1 waits", 16, PRB, style="italic")
s.line(445, 96, 445, 262, "#BFBFBF", 1.5)
panel(460, "(b) token budget = 5", ACC, [["S1", "S4"], ["S2", "S3"]])
s.text(665, 252, "5 and 5 tokens: no wait", 16, ACC, style="italic")
s.save(os.path.join(HERE, "fig_dynbsz.svg"))
