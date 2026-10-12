"""fig_k2.svg: schematic timeline of one K2 CTA (one sequence x head). Warp roles and pipeline depths from FlashKDA
csrc/smxx/fwd_kernel2.cuh (1 LOAD warp, 4 MMA warps = 128 threads, 1 STORE warp) and fwd_launch.cu
(kInputStages = 3, kOutputStages = 2). Block widths are illustrative, not measured."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
lanes = [("load warp", 40), ("4 MMA warps", 120), ("store warp", 200)]
for name, y in lanes:
    s.t(150, y + 26, name, 18, "bold" if "MMA" in name else "normal", ACC if "MMA" in name else INK, "end")
s.t(150, 120 + 48, "state S on chip", 15, "normal", GRAY, "end", "italic")
H = 40
# loads: fill 3 stages, then refill as stages free
loads = [(160, "chunk 1"), (244, "chunk 2"), (328, "chunk 3"), (414, "chunk 4"), (538, "chunk 5"), (662, "chunk 6")]
for x, lab in loads:
    s.box(x, 40, 78, H, "#FFFFFF", ACC, 1.5, 3)
    s.t(x + 39, 66, lab, 16)
s.t(280, 32, "3 stages in flight", 15, "normal", GRAY, "middle", "italic")
# MMA chain
xs = [250, 374, 498, 622, 746]
for i, x in enumerate(xs):
    w = 100 if i < 4 else 96
    s.box(x, 120, w, H + 4, ACC_F, ACC, 2.5, 3)
    s.t(x + w / 2, 148, f"chunk {i+1}" if i < 4 else "⋯", 17, "bold", ACC)
    if i < 4:
        s.line(x + w + 1, 142, xs[i + 1] - 1, 142, ACC, 2.5, "b")
for i in range(3):
    s.t(xs[i] + 112, 134, "S", 15, "bold", ACC)
# stores
for i, x in enumerate(xs[:4]):
    s.box(x + 108, 200, 76, H, "#FFFFFF", ACC, 1.5, 3)
    s.t(x + 146, 226, f"out {i+1}", 16)
# dependency hints
for i, x in enumerate(xs[:3]):
    s.line(x + 50, 166, x + 128, 198, GRAY, 1.2, "a", "3,3")
s.line(160, 262, 860, 262, GRAY, 1.5)
s.t(860, 280, "time", 15, "normal", GRAY, "end")
s.save(pathlib.Path(__file__).with_name("fig_k2.svg"))
