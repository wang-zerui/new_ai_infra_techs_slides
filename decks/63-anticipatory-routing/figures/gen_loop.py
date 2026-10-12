"""fig_loop: the vicious cycle DeepSeek-V4 §4.2.3 attributes to routing, drawn as a hypothesis (the paper says the
mechanism is not yet understood): MoE outliers -> update -> routing shifts -> larger outliers."""
import math, os
from svgkit import SVG, ACC, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 300
s = SVG(W, H, "fig_loop")

BW, BH = 200, 64
nodes = {
    "a": (140, 8, ["outliers in", "MoE layers"]),
    "b": (272, 210, ["update moves", "θ everywhere"]),
    "c": (8, 210, ["routing shifts", "at the next step"]),
}
for k, (x, y, lab) in nodes.items():
    s.box(x, y, BW, BH, lab, PRB_F if k == "a" else "#FFFFFF", PRB if k == "a" else NEU, INK, 18, sw=2)
# a -> b (right side), b -> c (bottom), c -> a (left side)
s.path("M342,74 Q392,120 382,206", PRB, 2.4, "arrPrb")
s.line(270, 242, 210, 242, PRB, 2.4, "arrPrb")
s.path("M98,208 Q86,120 136,64", PRB, 2.4, "arrPrb")
s.text(240, 140, "one θ for", 18, PRB, weight="bold")
s.text(240, 162, "features and routing", 18, PRB, weight="bold")
s.text(240, 296, "hypothesized loop", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_loop.svg"))
