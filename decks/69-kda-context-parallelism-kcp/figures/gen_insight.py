"""fig_insight.svg: a segment of KDA tokens acts on its entry state as an affine map S_out = M S_in + S~,
with M (d_k x d_k) and S~ (d_k x d_v) computable from the segment's tokens alone (K3 Eq. 17)."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(470, 290)
s.box(110, 8, 250, 40, GRAY_F, GRAY, 1.5, 4)
s.t(235, 34, "rank i: local tokens only", 18, "bold")
for cx, a, b in ((170, sub("M", "i"), "d<tspan font-size=\"12\" dy=\"4\">k</tspan><tspan dy=\"-4\"> × d</tspan><tspan font-size=\"12\" dy=\"4\">k</tspan>"),
                 (310, sub("S̃", "i"), "d<tspan font-size=\"12\" dy=\"4\">k</tspan><tspan dy=\"-4\"> × d</tspan><tspan font-size=\"12\" dy=\"4\">v</tspan>")):
    s.line(cx, 50, cx, 76)
    s.box(cx - 54, 78, 108, 50, ACC_F, ACC, 2.5, 4)
    s.t(cx, 101, a, 20, "bold", ACC)
    s.t(cx, 121, b, 15)
    s.line(cx, 130, cx, 152)
y = 172
s.t(58, y + 7, sub("S", "in"), 20, "bold", INK, "end")
s.line(64, y, 150, y, INK, 2.5)
s.raw(f'<circle cx="170" cy="{y}" r="18" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>'); s.t(170, y + 7, "×", 22, "bold")
s.line(188, y, 290, y, INK, 2.5)
s.raw(f'<circle cx="310" cy="{y}" r="18" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>'); s.t(310, y + 7, "+", 22, "bold")
s.line(328, y, 398, y, INK, 2.5)
s.t(404, y + 7, sub("S", "out"), 20, "bold", INK, "start")
s.t(235, 234, f'{sub("S", "out")} = {sub("M", "i")} · {sub("S", "in")} + {sub("S̃", "i")}', 20, "bold", ACC)
s.t(235, 266, "affine maps compose: prefix scan", 17, "normal", GRAY, "middle", "italic")
s.save(pathlib.Path(__file__).with_name("fig_insight.svg"))
