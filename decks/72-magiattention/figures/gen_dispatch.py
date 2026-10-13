"""fig_dispatch (analytic): varlen causal mask, 8 equal chunks (doc A = C0-C5, doc B = C6-C7), 4 ranks x 2 chunks.
Row area in chunk-block units (diagonal block = 1/2): A 0.5..5.5, B 0.5, 1.5; total 20, even share 5.
(a) contiguous split: 2, 6, 10, 2.  (b) greedy min-heap ([BLOG] Fig. 10): {C5,C6}=6, {C4,C0}=5, {C3,C7}=5, {C2,C1}=4."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 270, "fig_dispatch: contiguous vs min-heap dispatch")

doc = [0, 0, 0, 0, 0, 0, 1, 1]
start = [0, 0, 0, 0, 0, 0, 6, 6]
area = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.5, 1.5]
C = 26
X0, Y0 = 52, 54
s.text(12, 22, "Varlen causal mask, 8 chunks", 18, INK, anchor="start", weight="bold")
for r in range(8):
    for c in range(8):
        x, y = X0 + c * C, Y0 + r * C
        if start[r] <= c < r:
            s.rect(x, y, C, C, "#BFBFBF", "#FFFFFF", 0.8, 0)
        elif c == r:
            s.rect(x, y, C, C, "#FFFFFF", "#E6E6E6", 0.8, 0)
            s.add(f'<path d="M{x},{y} L{x},{y + C} L{x + C},{y + C} Z" fill="#BFBFBF"/>')
        else:
            s.rect(x, y, C, C, "#FFFFFF", "#E6E6E6", 0.8, 0)
    s.text(X0 - 8, Y0 + r * C + C / 2 + 5, f"C{r}", 15, INK, anchor="end")
    s.text(X0 + 8 * C + 26, Y0 + r * C + C / 2 + 5, f"{area[r]:g}", 15, MUTED)
s.rect(X0, Y0, 8 * C, 8 * C, "none", NEU, 1.4, 0)
s.text(X0 + 8 * C + 26, Y0 - 8, "area", 15, MUTED, style="italic")

U = 16                       # px per area unit


def panel(x0, title, loads, chunks, ours):
    s.text(x0, 22, title, 18, ACC if ours else INK, anchor="start", weight="bold")
    mx = max(loads)
    xe = x0 + 60 + 5 * U
    s.line(xe, 44, xe, 248, MUTED, 1.4, dash="5,4")
    s.text(xe, 264, "even share", 15, MUTED)
    for r, (w, ch) in enumerate(zip(loads, chunks)):
        y = 50 + r * 50
        s.text(x0 + 52, y + 21, f"rank{r}", 16, INK, anchor="end", weight="bold")
        hot = (not ours) and w == mx
        fill, stroke = (PRB_F, PRB) if hot else ((ACC_F, ACC) if ours else (NEU_F, NEU))
        s.rect(x0 + 60, y + 2, w * U, 28, fill, stroke, 2, 2)
        s.text(x0 + 60 + w * U + 8, y + 22, f"{w:g}", 17, PRB if hot else (ACC if ours else INK),
               anchor="start", weight="bold", halo=True)
        s.text(x0 + 60, y + 46, ch, 15, MUTED, anchor="start")


panel(330, "(a) Contiguous split", [2, 6, 10, 2], ["C0 C1", "C2 C3", "C4 C5", "C6 C7"], False)
panel(612, "(b) Min-heap dispatch", [6, 5, 5, 4], ["C5 C6", "C4 C0", "C3 C7", "C2 C1"], True)
s.save(os.path.join(HERE, "fig_dispatch.svg"))
