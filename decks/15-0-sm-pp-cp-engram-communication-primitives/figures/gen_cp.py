"""fig_cp_1/2: CP all-gather. (1) NVLink domain: G0's copy engine writes shard s0 into slot 0 of every peer's
symmetric buffer, no kernel. (2) V2.5 hybrid: RDMA put to the rail peer, then the copy engine forwards over NVLink."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282


def gpu_with_slots(s, x, y, idx, have_s0):
    s.rect(x, y, 176, 80, "#FFFFFF", NEU, 1.6, 5)
    s.text(x + 166, y + 24, f"G{idx}", 18, INK, anchor="end", weight="bold")
    for k in range(4):
        cx = x + 10 + k * 40
        if k == idx:          # own shard
            fill, stroke, dash, lab, col = (ACC_F, ACC, None, f"s{k}", ACC) if k == 0 else (NEU_F, NEU, None, f"s{k}", INK)
        elif k == 0 and have_s0:
            fill, stroke, dash, lab, col = ACC_F, ACC, None, "s0", ACC
        else:
            fill, stroke, dash, lab, col = "#FFFFFF", "#A6A6A6", "4,3", "", INK
        s.rect(cx, y + 40, 34, 28, fill, stroke, 1.5, 2, dash=dash)
        if lab:
            s.text(cx + 17, y + 60, lab, 15, col)


def draw(step):
    s = SVG(W, H, f"fig_cp_{step}: CP all-gather on copy engines (build step {step})")
    # left panel: NVLink domain
    s.rect(8, 8, 412, 268, "#FFFFFF", NEU, 1.6, 8)
    s.text(214, 32, "one NVLink domain", 19, INK, weight="bold")
    pos = [(24, 46), (228, 46), (24, 160), (228, 160)]
    for i, (x, y) in enumerate(pos):
        gpu_with_slots(s, x, y, i, have_s0=True)
    # CE writes from G0's s0 to slot 0 of each peer
    sx, sy = 51, 46 + 40 + 28          # bottom-center of G0's s0 cell
    s.line(202, 100, 236, 100, ACC, 2.4, "arrAcc")                 # to G1 (same row)
    s.line(sx, sy + 2, sx, 160 + 38, ACC, 2.4, "arrAcc")            # to G2 (below)
    s.line(190, 128, 250, 196, ACC, 2.4, "arrAcc")                 # to G3 (diagonal through the gap)
    s.text(124, 150, "CE copy", 17, ACC, weight="bold", halo=True)
    s.text(214, 266, "flags by stream write/wait ops", 16, MUTED, style="italic")
    if step >= 2:
        # right panel: hybrid domain (two nodes, one rail shown)
        s.rect(436, 8, 426, 268, "#FFFFFF", ACC, 1.6, 8)
        s.text(649, 32, "two nodes (V2.5 hybrid)", 19, ACC, weight="bold")
        BW, BH, GX = 84, 42, [532, 636, 740]
        ya, yb = 100, 178
        s.text(450, ya + 27, "node A", 17, INK, anchor="start", weight="bold")
        s.text(450, yb + 27, "node B", 17, INK, anchor="start", weight="bold")
        for j, x in enumerate(GX):
            s.box(x, ya, BW, BH, f"G{j}", ACC_F if j == 0 else "#FFFFFF", ACC if j == 0 else NEU, size=17)
            s.box(x, yb, BW, BH, f"G{j + 3}", "#FFFFFF", NEU, size=17)
        # CE push in node A (arcs above)
        for j in (1, 2):
            x0, x1 = GX[0] + BW / 2, GX[j] + BW / 2
            off = 14 if j == 1 else -14
            s.path(f"M{x0 + off},{ya - 2} Q{(x0 + off + x1) / 2},{ya - 6 - 26 * j} {x1},{ya - 3}", ACC, 2.2, "arrAcc")
        s.text(450, 62, "CE push", 16, ACC, anchor="start", weight="bold")
        # RDMA put along the rail G0 -> G3
        s.line(GX[0] + BW / 2, ya + BH + 2, GX[0] + BW / 2, yb - 3, INK, 2.6, "arr")
        s.text(GX[0] + BW / 2 + 10, (ya + BH + yb) / 2 + 6, "RDMA put", 16, INK, anchor="start", weight="bold")
        # CE forward in node B (arcs below)
        for j in (1, 2):
            x0, x1 = GX[0] + BW / 2, GX[j] + BW / 2
            off = 14 if j == 1 else -14
            s.path(f"M{x0 + off},{yb + BH + 2} Q{(x0 + off + x1) / 2},{yb + BH + 6 + 26 * j} {x1},{yb + BH + 3}", ACC, 2.2, "arrAcc")
        s.text(450, 262, "CE forward", 16, ACC, anchor="start", weight="bold")
    s.save(os.path.join(HERE, f"fig_cp_{step}.svg"))


draw(1)
draw(2)
