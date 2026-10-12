"""fig_overview: HPC-Ops fused sampler = K1 (one vocabulary pass, N blocks per row) + K2 (one block per row over
N x K candidates) + GPU-resident penalty bitmask read by K1 and updated by K2; temperature-only fast path = 1 kernel."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, badge

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 300
s = SVG(W, H, "fig_overview")

MY, MH = 104, 92          # main row
# penalty bitmask (top)
s.box(300, 16, 300, 46, "penalty bitmask · 1 bit / token", ACC_F, ACC, color=ACC, weight="bold", size=18)
badge(s, 300, 16, "3")
# main row
s.box(8, MY + 16, 128, 60, ["logits", "B × V"], "#FFFFFF", NEU, size=18)
s.box(176, MY, 230, MH, ["K1 · one vocab pass", "penalty · ÷T", "max/sum · top-K"], ACC_F, ACC, size=18)
badge(s, 176, MY, "1")
s.box(440, MY + 16, 112, 60, ["N × K", "candidates"], "#FFFFFF", NEU, size=18)
s.box(586, MY, 200, MH, ["K2 · 1 block/row", "softmax · top-p", "Gumbel-max"], ACC_F, ACC, size=18)
badge(s, 586, MY, "2")
s.box(808, MY + 16, 56, 60, "token", "#FFFFFF", NEU, size=17)
cy = MY + MH / 2
s.line(136, cy, 174, cy, INK, 2.2, "arr")
s.line(406, cy, 438, cy, INK, 2.2, "arr")
s.line(552, cy, 584, cy, INK, 2.2, "arr")
s.line(786, cy, 806, cy, INK, 2.2, "arr")
# bitmask read by K1, written by K2
s.path(f"M330,64 L330,{MY - 2}", ACC, 2.2, "arrAcc")
s.text(320, 86, "read bits", 16, ACC, anchor="end", style="italic")
s.path(f"M686,{MY - 2} L686,39 L602,39", ACC, 2.2, "arrAcc")
s.text(696, 86, "atomicOr sampled bit", 16, ACC, anchor="start", style="italic")
# temperature-only fast path
FY = 226
s.rect(176, FY, 610, 62, "#FFFFFF", ACC, 1.6, 6, dash="6,4")
s.text(192, FY + 26, "temperature only: 1 kernel", 18, ACC, anchor="start", weight="bold")
s.text(192, FY + 50, "token = argmax(logit / T + Gumbel noise) · last block reduces", 17, INK, anchor="start")
s.save(os.path.join(HERE, "fig_overview.svg"))
