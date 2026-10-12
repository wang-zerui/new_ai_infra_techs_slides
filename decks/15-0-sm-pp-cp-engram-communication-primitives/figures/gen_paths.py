"""fig_paths: the four traffic patterns DeepEP covers (EP all-to-all, PP point-to-point, CP all-gather, Engram gather)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_paths: EP / PP / CP / Engram traffic patterns")

PW, GAP = 204, 18
X0 = (W - (4 * PW + 3 * GAP)) / 2
TOP = 8
titles = ["EP", "PP", "CP", "Engram"]
caps = [["tokens to experts", "and back"], ["activations to", "the next stage"],
        ["every KV shard", "to every rank"], ["table rows", "to each token"]]


def gpu(x, y, label, w=44, h=32, fill="#FFFFFF", stroke=NEU):
    s.box(x, y, w, h, label, fill, stroke, size=15)


for i in range(4):
    px = X0 + i * (PW + GAP)
    new = i > 0
    s.rect(px, TOP, PW, H - TOP - 4, "#FFFFFF", ACC if new else NEU, 1.6, 8, dash=None if new else "6,4")
    s.text(px + PW / 2, TOP + 30, titles[i], 21, ACC if new else INK, weight="bold")
    cx = px + PW / 2
    if i == 0:      # EP: all-to-all between 2 left and 2 right GPUs (full bipartite + same side)
        L = [(px + 30, 72), (px + 30, 150)]
        R = [(px + PW - 74, 72), (px + PW - 74, 150)]
        for (lx, ly) in L:
            for (rx, ry) in R:
                s.line(lx + 44, ly + 16, rx, ry + 16, INK, 1.8)
        s.line(L[0][0] + 22, L[0][1] + 32, L[1][0] + 22, L[1][1], INK, 1.8)
        s.line(R[0][0] + 22, R[0][1] + 32, R[1][0] + 22, R[1][1], INK, 1.8)
        for k, (x, y) in enumerate(L + R):
            gpu(x, y, f"G{k}")
    elif i == 1:    # PP: stage chain
        ys = 110
        xs = [px + 14, px + 80, px + 146]
        for k, x in enumerate(xs):
            gpu(x, ys, f"S{k}")
            if k < 2:
                s.line(x + 46, ys + 16, xs[k + 1] - 2, ys + 16, ACC, 2.4, "arrAcc")
    elif i == 2:    # CP all-gather: G0 sends its shard to all others (each does the same)
        cpos = [(cx - 22, 66), (px + 14, 150), (cx - 22, 150), (px + PW - 58, 150)]
        for k, (x, y) in enumerate(cpos):
            gpu(x, y, f"G{k}", fill=ACC_F if k == 0 else "#FFFFFF", stroke=ACC if k == 0 else NEU)
        for k in (1, 2, 3):
            x, y = cpos[k]
            s.line(cx, 66 + 34, x + 22, y - 3, ACC, 2.2, "arrAcc")
    else:           # Engram: rows of a sharded table gathered into one token's slot
        tx, ty, rw, rh = px + PW - 92, 50, 58, 13
        for k in range(8):
            hot = k in (1, 6)
            s.rect(tx, ty + k * rh, rw, rh, ACC_F if hot else NEU_F, ACC if hot else "#A6A6A6", 1.2, 0)
        s.line(tx - 4, ty + 4 * rh, tx + rw + 4, ty + 4 * rh, MUTED, 1.6, dash="4,3")
        s.text(tx + rw + 6, ty + 2 * rh + 6, "s0", 15, MUTED, anchor="start")
        s.text(tx + rw + 6, ty + 6 * rh + 6, "s1", 15, MUTED, anchor="start")
        gx, gy = px + 16, 176
        gpu(gx, gy, "token", w=60)
        for k, ex in ((1, gx + 16), (6, gx + 44)):
            yy = ty + k * rh + rh / 2
            s.path(f"M{tx - 2},{yy} L{ex + 14},{yy} Q{ex},{yy} {ex},{yy + 14} L{ex},{gy - 3}", ACC, 2.2, "arrAcc")
    # caption
    for j, line in enumerate(caps[i]):
        s.text(cx, 232 + j * 22, line, 18, INK)
s.save(os.path.join(HERE, "fig_paths.svg"))
