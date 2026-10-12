"""fig_overview.svg: FlashKDA v1 forward = K1 prepare (token-parallel) -> HBM workspace -> K2 recurrence (head-parallel,
warp-specialized). Sources: FlashKDA deep dive Sec. 2; csrc/smxx/fwd_launch.cu (grid_k1 = tiles x H, grid_k2 = N x H,
3 input / 2 output stages), fwd_kernel2.cuh (load / 4 MMA / store warps)."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 320)
# inputs
s.box(8, 104, 96, 104, GRAY_F, GRAY, 1.5)
s.t(56, 132, "HBM", 18, "bold")
s.t(56, 160, "q · k · g", 17)
s.t(56, 188, "v · β", 17)
s.line(104, 150, 138, 150)
# K1
s.box(140, 30, 236, 248, ACC_F, ACC, 2.5, 6)
s.t(258, 58, "K1 prepare", 21, "bold", ACC)
s.t(258, 82, "token-parallel", 17)
s.t(258, 104, "grid: tiles × heads", 16, "normal", GRAY)
ops = ["QK L2 norm", "gate cumsum + decay", "L and Mqk matrices", "16×16 inverse"]
for i, op in enumerate(ops):
    y = 118 + i * 38
    s.box(152, y, 212, 30, "#FFFFFF", ACC, 1.2, 4)
    s.t(258 if i < 3 else 248, y + 21, op, 16)
s.badge(348, 118 + 3 * 38 + 15, 3, 11)
s.line(376, 150, 402, 150)
# workspace
s.box(404, 30, 106, 248, "#FFFFFF", GRAY, 1.5, 6, "5,4")
s.t(457, 56, "workspace", 16, "bold")
s.t(457, 76, "per tile", 15, "normal", GRAY)
items = ["k decayed", "q decayed", "k restored", "g total", "INV", "Mqk"]
for i, it in enumerate(items):
    y = 88 + i * 30
    s.box(414, y, 86, 24, GRAY_F, GRAY, 1, 3)
    s.t(457, y + 17, it, 15)
s.badge(457, 14 + 2, 1, 11)
s.line(510, 150, 538, 150)
# K2
s.box(540, 30, 240, 248, ACC_F, ACC, 2.5, 6)
s.t(660, 58, "K2 recurrence", 21, "bold", ACC)
s.t(660, 82, "head-parallel", 17)
s.t(660, 104, "grid: seqs × heads", 16, "normal", GRAY)
rows = [("load warp · 3 stages", "#FFFFFF", ACC, 1.2), ("4 MMA warps + state S", ACC_F, ACC, 2.5), ("store warp · 2 stages", "#FFFFFF", ACC, 1.2)]
for i, (lab, f, st, sw) in enumerate(rows):
    y = 120 + i * 48
    s.box(552, y, 216, 38, f, st, sw, 4)
    s.t(660 if i != 1 else 650, y + 25, lab, 16, "bold" if i == 1 else "normal", ACC if i == 1 else INK)
s.badge(754, 120 + 48 + 19, 2, 11)
s.line(780, 150, 798, 150)
s.box(800, 104, 64, 104, GRAY_F, GRAY, 1.5)
s.t(832, 146, "o", 18, "bold")
s.t(832, 174, "final S", 15)
# v, beta path
s.path("M56,208 V304 H660 V280", INK, 1.8)
s.t(258, 294, "v · β read directly by K2", 15, "normal", GRAY, "middle", "italic")
s.save(pathlib.Path(__file__).with_name("fig_overview.svg"))
