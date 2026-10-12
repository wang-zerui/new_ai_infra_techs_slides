"""fig_merge.svg: one all-gather of [S~_j, M_j] then an in-order merge S <- M_j S + S~_j over preceding ranks
(K3 Sec. 5.1.2; FLA fla/ops/cp README 'Merge'; chunk_delta_h.py: hm = [H, K, V + K] fp32 all-gathered).
Backward: same structure reversed with transposed M (FLA README 'Pre-Process Backward')."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
for j in range(4):
    y = 34 + j * 44
    s.t(60, y + 23, f"GPU {j}", 16, "bold", INK, "end")
    s.box(68, y, 60, 34, ACC_F, ACC, 2, 3); s.t(98, y + 23, sub("S̃", j), 16, "bold", ACC)
    s.box(128, y, 60, 34, "#FFFFFF", ACC, 2, 3); s.t(158, y + 23, sub("M", j), 16, "bold", ACC)
    s.line(190, y + 17, 222, y + 17, INK, 1.8)
s.box(224, 34, 46, 166, ACC, ACC, 1.5, 4)
s.raw('<text x="247" y="117" font-size="17" font-weight="bold" fill="#FFFFFF" text-anchor="middle" transform="rotate(-90 247 117)">all-gather</text>')
s.line(272, 117, 318, 117, INK, 2.5)
s.t(594, 34, "merge on GPU 3 (needs ranks 0-2)", 18, "bold", ACC)
s.raw(f'<circle cx="340" cy="117" r="16" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>'); s.t(340, 123, "0", 17, "bold")
xs = [380, 510, 640]
s.line(357, 117, xs[0] - 2, 117, ACC, 2.5, "b")
for j, x in enumerate(xs):
    s.box(x, 93, 110, 48, ACC_F, ACC, 2, 4)
    s.t(x + 55, 123, f'{sub("M", j)}·S + {sub("S̃", j)}', 17, "bold", ACC)
    nxt = xs[j + 1] if j < 2 else 772
    s.line(x + 112, 117, nxt - 2, 117, ACC, 2.5, "b")
s.box(772, 93, 90, 48, ACC_F, ACC, 2.5, 4); s.t(817, 123, sub("S", "in,3"), 18, "bold", ACC)
s.t(600, 180, "small matmuls · no more communication", 16, "normal", INK)
s.t(600, 206, "backward: same chain in reverse with Mᵀ", 16, "normal", GRAY, "middle", "italic")
s.t(435, 262, "payload per GPU: H × d<tspan font-size=\"13\" dy=\"4\">k</tspan><tspan dy=\"-4\"> × (d</tspan><tspan font-size=\"13\" dy=\"4\">k</tspan><tspan dy=\"-4\"> + d</tspan><tspan font-size=\"13\" dy=\"4\">v</tspan><tspan dy=\"-4\">) values · no dependence on length</tspan>", 17, "bold")
s.save(pathlib.Path(__file__).with_name("fig_merge.svg"))
