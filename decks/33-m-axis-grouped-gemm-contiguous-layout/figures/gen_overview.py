"""fig_overview: contiguous layout. A = all experts' rows stacked along M, each segment padded to BLOCK_M;
m_indices names the expert of every row (-1 = padding); each M tile reads its expert's weight slab B[e];
D has the same row layout as A."""
import os
from tilekit import SVG, seg, pad, hlines, vlines, outline, mono, EXP, INK, MUTED, ACC

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_overview: contiguous layout + per-tile expert lookup")

BM, TH, Y0 = 128, 34, 56            # rows per tile, pt per tile, top of the matrices
experts = [(0, 300), (1, 60), (2, 200)]   # running example: rows routed to each local expert
px = TH / BM

# compute segments (rows) -> y ranges
segs, tiles, y = [], [], Y0
for e, m in experts:
    ntile = -(-m // BM)
    segs.append((e, y, m * px, (ntile * BM - m) * px))
    for t in range(ntile):
        tiles.append((e, y + t * TH))
    y += ntile * TH
YEND = y

XI, WI = 34, 50          # m_indices column
XA, WA = 104, 150        # A [sumM, K]
XB, WB = 344, 112        # B [G, N, K]
XD, WD = 552, 216        # D [sumM, N]

# headers
mono(s, XI + WI / 2, 44, "m_indices", 15, ACC, weight="bold")
s.text(XA + WA / 2, 44, "A (ΣM × K)", 17, INK, weight="bold")
s.text(XB + WB / 2, 44, "B (G × N × K)", 17, INK, weight="bold")
s.text(XD + WD / 2, 44, "D (ΣM × N)", 17, INK, weight="bold")

for e, ys, hv, hp in segs:
    seg(s, XA, ys, WA, hv, e, f"E{e}", size=18)
    seg(s, XI, ys, WI, hv, e, None)
    mono(s, XI + WI / 2, ys + hv / 2 + 5.5, str(e), 16, INK, weight="bold")
    pad(s, XA, ys + hv, WA, hp)
    pad(s, XI, ys + hv, WI, hp)
    mono(s, XI + WI / 2, ys + hv + hp / 2 + 5, "-1", 15, INK, weight="bold", halo=True)
    # D rows: same layout, tinted per expert, with N-block grid
    f, st = EXP[e]
    s.rect(XD, ys, WD, hv, f, st, 1.4, rx=0)
    pad(s, XD, ys + hv, WD, hp)

# tile boundaries (accent dashed) across m_indices, A and D; N blocks on D
tile_ys = [ty for _, ty in tiles][1:]
hlines(s, XI, 0, WI, tile_ys)
hlines(s, XA, 0, WA, tile_ys)
hlines(s, XD, 0, WD, tile_ys)
vlines(s, [XD + WD * k / 4 for k in (1, 2, 3)], Y0, YEND - Y0, color=ACC)
outline(s, XI, Y0, WI, YEND - Y0, ACC, 2.2)
outline(s, XA, Y0, WA, YEND - Y0)
outline(s, XD, Y0, WD, YEND - Y0)

# weight slabs placed next to the tiles that use them
slab_y = {}
for e in range(3):
    ty = [t for ee, t in tiles if ee == e]
    cy = (ty[0] + ty[-1] + TH) / 2
    h = 40
    slab_y[e] = (cy - h / 2, h)
    seg(s, XB, cy - h / 2, WB, h, e, f"B[{e}]", size=17)

# lookup arrows: every M tile -> its expert's slab (all six drawn)
for e, ty in tiles:
    sy, sh = slab_y[e]
    tgt = min(max(ty + TH / 2, sy + 8), sy + sh - 8)
    s.line(XA + WA + 4, ty + TH / 2, XB - 3, tgt, ACC, 2, "arrAcc")
    s.add(f'<circle cx="{XA + WA + 4:.1f}" cy="{ty + TH / 2:.1f}" r="3" fill="{ACC}"/>')

s.text((XB + WB + XD) / 2, (Y0 + YEND) / 2 + 9, "=", 28, MUTED)
# tile-size bracket on the right of D
s.line(XD + WD + 10, Y0, XD + WD + 10, Y0 + TH, INK, 2)
s.line(XD + WD + 6, Y0, XD + WD + 14, Y0, INK, 2)
s.line(XD + WD + 6, Y0 + TH, XD + WD + 14, Y0 + TH, INK, 2)
s.text(XD + WD + 18, Y0 + TH / 2 + 6, "BLOCK_M", 15, INK, anchor="start")
s.text(XD + WD + 18, Y0 + TH / 2 + 24, "= 128 rows", 15, INK, anchor="start")
pad(s, XD + WD + 10, YEND - 70, 22, 16)
s.text(XD + WD + 38, YEND - 57, "padding", 15, INK, anchor="start")
s.text(XD + WD + 10, YEND - 36, "zero rows", 15, INK, anchor="start")
s.text(XD + WD + 10, YEND - 18, "index −1", 15, INK, anchor="start")
mono(s, (XA + XB + WB) / 2, YEND + 24, "tile m uses B[ m_indices[128·m] ]", 16, ACC)
s.save(os.path.join(HERE, "fig_overview.svg"))
