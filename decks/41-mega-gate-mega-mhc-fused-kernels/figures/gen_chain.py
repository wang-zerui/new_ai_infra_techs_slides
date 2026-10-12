"""fig_chain: where Mega mHC and Mega Gate sit around one MoE sub-layer (tensor formats from the APIs)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

W, H = 870, 250
s = SVG(W, H, "Mega mHC and Mega Gate around an MoE sub-layer")

Y0, Y1 = 98, 158
cy = (Y0 + Y1) / 2
boxes = {
    "attn": (6, 112), "mhc": (168, 300), "gate": (384, 516), "moe": (600, 724), "mhc2": (756, 866)}
s.box(*[boxes["attn"][0], Y0, boxes["attn"][1] - boxes["attn"][0], Y1 - Y0], ["Attention", "sub-layer"],
      fill=NEU_F, stroke=NEU, sizes=[18, 15], weights=["bold", "normal"])
s.box(boxes["mhc"][0], Y0, boxes["mhc"][1] - boxes["mhc"][0], Y1 - Y0, ["Mega mHC", "1 kernel"],
      fill=ACC_F, stroke=ACC, sizes=[19, 15], weights=["bold", "normal"], colors=[ACC, INK], sw=2.4)
s.box(boxes["gate"][0], Y0, boxes["gate"][1] - boxes["gate"][0], Y1 - Y0, ["Mega Gate", "1 kernel"],
      fill=ACC_F, stroke=ACC, sizes=[19, 15], weights=["bold", "normal"], colors=[ACC, INK], sw=2.4)
s.box(boxes["moe"][0], Y0, boxes["moe"][1] - boxes["moe"][0], Y1 - Y0, ["Mega MoE", "deck 35"],
      fill=NEU_F, stroke=NEU, sizes=[18, 15], weights=["bold", "normal"])
s.box(boxes["mhc2"][0], Y0, boxes["mhc2"][1] - boxes["mhc2"][0], Y1 - Y0, ["Mega mHC", "next"],
      fill=ACC_F, stroke=ACC, sizes=[18, 15], weights=["bold", "normal"], colors=[ACC, INK], sw=2.4)

# main flow
s.line(112, cy, 167, cy, INK, 2.2, "arr");  s.text(140, cy - 8, "x", 16, MUTED, style="italic")
s.line(300, cy, 383, cy, INK, 2.2, "arr");  s.text(342, cy - 8, "y BF16", 16, INK)
s.line(516, cy, 599, cy, INK, 2.2, "arr");  s.text(558, cy - 26, "top-6", 16, INK); s.text(558, cy - 8, "idx, w", 16, INK)
s.line(724, cy, 755, cy, INK, 2.2, "arr");  s.text(740, cy - 8, "x", 16, MUTED, style="italic")

# FP8 copy of y straight to the experts
s.path(f"M 264 {Y0} L 264 56 L 662 56 L 662 {Y0 - 1}", ACC, 2.2, "arrAcc")
s.text(463, 48, "y FP8 + UE8M0 scales, in Mega MoE's layout", 16, ACC, weight="bold")

# residual stream bypass to the next boundary
s.path(f"M 234 {Y1} L 234 206 L 811 206 L 811 {Y1 + 1}", INK, 2.2, "arr")
s.text(526, 228, "new residual 4 × d + next H_post, H_res", 16, INK)

# badges
s.badge(178, 90, 1, fill=ACC)
s.badge(204, 90, 2, fill=ACC)
s.badge(394, 90, 3, fill=ACC)

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_chain.svg"))
