"""fig_exact: tests/test_sampler.py pattern: inject the same Gumbel noise into hpc.fused_sampler and a pure-PyTorch
reference; sampled tokens and penalty-mask writeback must be torch.equal."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, GOOD

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 480, 300
s = SVG(W, H, "fig_exact")
s.box(90, 8, 300, 56, ["logits + settings", "+ injected Gumbel noise g"], "#FFFFFF", NEU, size=18)
s.box(10, 114, 210, 56, ["hpc.fused_sampler", "2 kernels"], ACC_F, ACC, size=18, color=ACC, weight="bold")
s.box(260, 114, 210, 56, ["PyTorch reference", "stage by stage"], NEU_F, NEU, size=18)
s.path("M180,66 L180,90 L115,90 L115,112", INK, 2.2, "arr")
s.path("M300,66 L300,90 L365,90 L365,112", INK, 2.2, "arr")
for cx in (115, 365):
    s.line(cx, 172, cx, 180, INK, 2.2, "arr")
    s.box(cx - 90, 182, 180, 34, "token + mask bits", "#FFFFFF", NEU, size=17, sw=1.5)
s.path("M115,216 L115,232 L365,232 L365,216", INK, 2.2)
s.line(240, 232, 240, 246, INK, 2.2, "arr")
s.box(150, 248, 180, 46, "torch.equal ✓", "#FFFFFF", GOOD, color=GOOD, weight="bold", size=19, sw=2.2)
s.save(os.path.join(HERE, "fig_exact.svg"))
