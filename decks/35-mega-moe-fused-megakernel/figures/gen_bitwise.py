"""fig_bitwise: what tests/test_mega_moe.py checks (same inputs, two paths, bitwise comparison)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED, GOOD

W, H = 450, 300
s = SVG(W, H, "bitwise check in the Mega MoE test")

s.box(60, 6, 330, 52, ["same inputs", "x, top-k, weights"], fill="#FFFFFF", stroke=NEU, sizes=[18, 16],
      weights=["bold", "normal"])
s.box(6, 112, 200, 72, ["five kernels", "DeepEP + DeepGEMM", "+ TileLang"], fill=NEU_F, stroke=NEU,
      sizes=[18, 15, 15], weights=["bold", "normal", "normal"])
s.box(244, 112, 200, 72, ["Mega MoE", "one kernel"], fill=ACC_F, stroke=ACC, sizes=[18, 16],
      weights=["bold", "normal"], colors=[ACC, INK])
s.line(150, 58, 106, 111, INK, 2.2, "arr")
s.line(300, 58, 344, 111, INK, 2.2, "arr")
s.line(106, 184, 170, 233, INK, 2.2, "arr")
s.line(344, 184, 280, 233, INK, 2.2, "arr")
s.rect(60, 234, 330, 60, "#FFFFFF", GOOD, 2.4, 4)
s.text(225, 258, "torch.equal(y, y_ref)", 16, INK, family="Menlo, Courier New, monospace")
s.text(225, 283, "bitwise identical", 18, GOOD, weight="bold")
s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_bitwise.svg"))
