"""fig_deps: per-tile dependencies inside the kernel for one M-block (rows of one expert)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "tile-level dependencies via counters")

LX = 150
# row labels
s.text(LX, 40, "pulled rows", 18, INK, anchor="end", weight="bold")
s.text(LX, 108, "Linear-1 tiles", 18, INK, anchor="end", weight="bold")
s.text(LX, 128, "(N blocks)", 15, MUTED, anchor="end", style="italic")
s.text(LX, 178, "ready bits", 18, INK, anchor="end", weight="bold")
s.text(LX, 244, "Linear-2 steps", 18, INK, anchor="end", weight="bold")
s.text(LX, 264, "(K blocks)", 15, MUTED, anchor="end", style="italic")

# row A: pulled rows of one M-block + arrival counter
x0 = 172
for i in range(8):
    s.rect(x0 + i * 22, 22, 18, 22, ACC, ACC, 1, 2)
s.line(x0 + 8 * 22 + 4, 32, 382, 32, ACC, 2.2, "arrAcc")
s.box(384, 12, 196, 40, "all rows landed", fill=ACC_F, stroke=ACC, size=17)
s.text(592, 39, "block counter = rows in block", 16, MUTED, anchor="start", style="italic")

# row B: L1 N-block tiles
TW, TG = 70, 10
tx = [172 + i * (TW + TG) for i in range(8)]
done = 5
for i, x in enumerate(tx):
    s.box(x, 86, TW, 44, f"N{i}", fill=ACC_F if i < done else "#FFFFFF", stroke=ACC, size=18,
          dash=None if i < done else "5,3")
s.text(tx[-1] + TW + 8, 114, "…", 20, INK, anchor="start")
# gate: the counter releases every L1 tile of this block (one drop per tile)
s.line(482, 52, 482, 70, ACC, 2.2)
s.line(tx[0] + TW / 2, 70, tx[-1] + TW / 2, 70, ACC, 2.2)
for x in tx:
    s.line(x + TW / 2, 70, x + TW / 2, 85, ACC, 2, "arrAcc")

# row C: ready bits (one per L1 tile)
for i, x in enumerate(tx):
    cx = x + TW / 2
    s.rect(cx - 10, 166, 20, 20, ACC if i < done else "#FFFFFF", ACC, 1.6, 2)
    s.line(cx, 130, cx, 165, ACC if i < done else MUTED, 1.8, "arrAcc" if i < done else "arrGray",
           dash=None if i < done else "4,3")

# row D: L2 K-steps, each fed by two L1 tiles
for k in range(4):
    xa, xb = tx[2 * k], tx[2 * k + 1] + TW
    ready = (2 * k + 1) < done
    s.box(xa, 222, xb - xa, 44, f"K{k}" if ready else f"K{k}: waits", fill=ACC_F if ready else "#FFFFFF",
          stroke=ACC if ready else MUTED, color=INK if ready else MUTED, size=18, dash=None if ready else "5,3")
    for j in (2 * k, 2 * k + 1):
        cx = tx[j] + TW / 2
        s.line(cx, 186, cx + (0.45 if j % 2 == 0 else -0.45) * 18, 221, ACC if j < done and ready else MUTED, 1.8,
               "arrAcc" if (j < done and ready) else "arrGray", dash=None if (j < done and ready) else "4,3")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_deps.svg"))
