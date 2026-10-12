"""fig_fragments.svg: how one rank builds its fragments before any communication (FLA fla/ops/cp README,
'Pre-Process Forward'; K3 Eq. 17): S~ by the ordinary chunk recurrence from S = 0, M by chaining per-chunk
transitions M_[t] = Diag(gamma^C) - (Gamma (.) K)^T W from the identity."""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgkit import *
s = SVG(870, 285)
s.t(112, 54, "local chunks", 16, "normal", GRAY, "end")
s.t(112, 128, "S̃ chain", 18, "bold", ACC, "end")
s.t(112, 200, "M chain", 18, "bold", ACC, "end")
xs = [150 + j * 140 for j in range(4)]
for j, x in enumerate(xs):
    s.box(x, 32, 110, 36, GRAY_F, GRAY, 1.5, 3)
    s.t(x + 55, 56, f"chunk {j+1}", 16)
    s.line(x + 55, 70, x + 55, 102, GRAY, 1.5)
    s.line(x + 55, 70, x + 55, 174, GRAY, 1.2, None, "3,3")
    s.box(x + 20, 104, 70, 40, ACC_F, ACC, 2, 4)
    s.t(x + 55, 130, "step", 15, "normal", ACC)
    s.box(x + 20, 176, 70, 40, ACC_F, ACC, 2, 4)
    s.t(x + 55, 202, f"× {sub('M', '[t]')}", 16, "bold", ACC)
# start nodes
for y, lab in ((124, "0"), (196, "I")):
    s.raw(f'<circle cx="138" cy="{y}" r="15" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
    s.t(138, y + 6, lab, 17, "bold")
    s.line(154, y, xs[0] + 18, y, INK, 2)
    for j in range(3):
        s.line(xs[j] + 92, y, xs[j + 1] + 18, y, ACC, 2.5, "b")
    s.line(xs[3] + 92, y, 732, y, ACC, 2.5, "b")
s.box(734, 104, 120, 40, ACC_F, ACC, 2.5, 4); s.t(794, 131, sub("S̃", "i"), 19, "bold", ACC)
s.box(734, 176, 120, 40, ACC_F, ACC, 2.5, 4); s.t(794, 203, sub("M", "i"), 19, "bold", ACC)
s.t(435, 252, "step: S̃ ← Diag(γ)·S̃ + (Γ⊙K)ᵀ·(U − W·S̃)", 17)
s.t(435, 277, f"{sub('M', '[t]')} = Diag(γ) − (Γ⊙K)ᵀ·W", 17)
s.save(pathlib.Path(__file__).with_name("fig_fragments.svg"))
