"""fig_statusquo: which KV chunks each rank receives under ring P2P CP, 4 ranks x 8 chunks (blog Fig. 11).
(a) causal mask, zigzag: rank r holds chunks {r, 7-r}; rank j needs chunk c iff c <= 7-j  -> 6/24 unused.
(b) varlen block-causal mask with a last global block: holdings {0,2},{1,3},{4,6},{5,7};
    ranks 0-1 need only chunks 0-3, ranks 2-3 need all  -> 8/24 unused.
Ring P2P delivers every remote chunk to every rank."""
import os
from svgkit import SVG, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 286, "fig_statusquo: ring P2P KV deliveries, needed vs unused")

USED, OWN = "#BFBFBF", "#FFFFFF"
CW, CH = 40, 36

def panel(x0, title, own, need):
    s.text(x0, 24, title, 19, INK, anchor="start", weight="bold")
    gx = x0 + 74
    for c in range(8):
        s.text(gx + c * CW + CW / 2, 58, f"KV{c}", 15, INK)
    for r in range(4):
        y = 68 + r * CH
        s.text(gx - 10, y + CH / 2 + 6, f"rank{r}", 17, INK, anchor="end", weight="bold")
        for c in range(8):
            x = gx + c * CW
            if c in own[r]:
                s.rect(x, y, CW, CH, OWN, NEU, 1.2, 0)
                s.text(x + CW / 2, y + CH / 2 + 5, "own", 14, MUTED)
            elif need(r, c):
                s.rect(x, y, CW, CH, USED, NEU, 1.2, 0)
            else:
                s.rect(x, y, CW, CH, "url(#hatchPrb)", PRB, 1.6, 0)
    # ring arrow hint on the right
    return gx + 8 * CW

zig = {r: {r, 7 - r} for r in range(4)}
xe = panel(14, "(a) Causal mask", zig, lambda r, c: c <= 7 - r)
s.line(438, 14, 438, 222, "#BFBFBF", 1.2)
vb_own = {0: {0, 2}, 1: {1, 3}, 2: {4, 6}, 3: {5, 7}}
panel(452, "(b) Packed block-causal mask", vb_own, lambda r, c: r >= 2 or c <= 3)

# legend
ly = 244
s.rect(150, ly, 30, 24, USED, NEU, 1.2, 0)
s.text(188, ly + 18, "received and used", 17, INK, anchor="start")
s.rect(390, ly, 30, 24, "url(#hatchPrb)", PRB, 1.6, 0)
s.text(428, ly + 18, "received, never used", 17, PRB, anchor="start", weight="bold")
s.rect(650, ly, 30, 24, OWN, NEU, 1.2, 0)
s.text(688, ly + 18, "own chunk", 17, INK, anchor="start")
s.save(os.path.join(HERE, "fig_statusquo.svg"))
