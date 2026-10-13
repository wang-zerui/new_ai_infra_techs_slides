"""Pixel-measure bar heights in MagiAttention benchmark PNGs.
usage: measure_bars.py <png> <tick_step> <n_series> [names]
Calibration: y0 = x-axis frame row; d = gridline spacing measured only in the background columns between bar
groups (where gridlines are not hidden by bars) = tick_step units. Bar top = median over 4 columns of the
highest row still in the bar colour (gaps up to 0.8% of height bridged for crossing lines)."""
import sys, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p, step, nser = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
names = sys.argv[4].split(",") if len(sys.argv) > 4 else None
a = np.asarray(Image.open(p).convert("RGB")).astype(int)
H, W, _ = a.shape
r, g, b = a[..., 0], a[..., 1], a[..., 2]
def isbg(px):
    return (px[0] > 225 and px[1] > 225 and px[2] > 225) or (abs(px[0]-px[1]) < 8 and abs(px[1]-px[2]) < 8 and px[0] > 180)
gray = (abs(r - g) < 6) & (abs(g - b) < 6) & (r > 150) & (r < 248)
# frame bottom: row with longest gray run in lower half
frac_all = gray.mean(axis=1)
y0 = int(np.argmax(np.where(np.arange(H) > H * 0.6, frac_all, 0)))
ys = y0 - int(0.012 * H)
row = a[ys]
segs = []
x = 0
while x < W:
    if isbg(row[x]):
        x += 1; continue
    x1 = x; c0 = row[x]
    while x < W and not isbg(row[x]) and np.abs(row[x] - c0).sum() < 40:
        x += 1
    if x - x1 > 0.006 * W:
        segs.append((x1, x, tuple(int(v) for v in np.median(a[ys - 5:ys + 5, x1 + 3:x - 3].reshape(-1, 3), axis=0))))
    if x == x1: x += 1
groups = [[segs[0]]]
for s in segs[1:]:
    if s[0] - groups[-1][-1][1] > 0.02 * W: groups.append([s])
    else: groups[-1].append(s)
# background columns between groups
cols = []
for g1, g2 in zip(groups[:-1], groups[1:]):
    xa, xb = g1[-1][1] + 10, g2[0][0] - 10
    cols += list(range(xa, xb))
cols = np.array(cols)
fr = gray[:, cols].mean(axis=1)
rows = np.where((fr > 0.6) & (np.arange(H) < y0 - 5) & (np.arange(H) > 0.1 * H))[0]
cl = []
for y in rows:
    if cl and y - cl[-1][-1] <= 3: cl[-1].append(y)
    else: cl.append([y])
cents = sorted([float(np.mean(c)) for c in cl])
lines = cents + [y0]
diffs = np.diff(lines)
d = float(np.median(diffs))
ppu = d / step
def top_of(x1, x2, col):
    tops = []
    for f in (0.15, 0.3, 0.7, 0.85):
        x = int(x1 + f * (x2 - x1))
        y = ys; last = ys; gap = 0
        while y > 0:
            px = a[y, x]
            if np.abs(px - np.array(col)).sum() < 60:
                last = y; gap = 0
            else:
                gap += 1
                if gap > int(0.008 * H): break
            y -= 1
        tops.append(last)
    tops = sorted(tops)
    return (tops[1] + tops[2]) / 2, tops[0], tops[-1]
print(p.split("/")[-1], f"y0={y0} gridlines={[round(c) for c in cents]} d={d:.1f}px/{step:g} (diff spread {diffs.min():.0f}-{diffs.max():.0f})")
for gi, gr in enumerate(groups):
    out = []
    for si, (x1, x2, col) in enumerate(gr):
        med, hi, lo = top_of(x1, x2, col)
        v = (y0 - med) / ppu; vmax = (y0 - hi) / ppu; vmin = (y0 - lo) / ppu
        nm = names[si] if names and len(gr) == len(names) else f"s{si}"
        out.append(f"{nm}={v:.0f}" + (f"({vmin:.0f}-{vmax:.0f})" if vmax - vmin > 3 else ""))
    print(f"  g{gi} n={len(gr)}: " + "  ".join(out))
