"""fig_graph: one decode MoE layer, (a) eager: the CPU launches every kernel, GPU idles between them;
(b) CUDA Graph replay: one launch, kernels back to back, with pointers/shapes/grid frozen at capture."""
import os
from tilekit import SVG, INK, MUTED, ACC, ACC_F, PRB, NEU, NEU_F

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 290, "fig_graph: eager launches vs CUDA Graph replay")
names = ["attn", "disp", "GEMM", "GEMM", "comb"]
X0, KW = 64, 56

# (a) eager
s.text(10, 22, "(a) eager", 18, INK, anchor="start", weight="bold")
s.text(470, 22, "L = kernel launch", 15, MUTED, anchor="end")
s.text(X0 - 8, 54, "CPU", 16, INK, anchor="end", weight="bold")
s.text(X0 - 8, 92, "GPU", 16, INK, anchor="end", weight="bold")
for i, nm in enumerate(names):
    x = X0 + i * 82
    s.box(x, 38, 24, 24, "L", NEU_F, NEU, INK, 15)
    s.box(x + 24, 76, KW, 26, nm, NEU_F, NEU, INK, 15)
    if i < len(names) - 1:
        s.rect(x + 24 + KW, 76, 26, 26, "url(#hatchPrb)", PRB, 1, rx=0)
s.text(X0 + 2 * 82 + 52, 126, "GPU idle between launches", 15, PRB, weight="bold")

s.line(6, 146, 474, 146, "#999999", 1)

# (b) CUDA Graph replay
s.text(10, 172, "(b) CUDA Graph replay", 18, ACC, anchor="start", weight="bold")
s.text(X0 - 8, 204, "CPU", 16, INK, anchor="end", weight="bold")
s.text(X0 - 8, 242, "GPU", 16, INK, anchor="end", weight="bold")
s.box(X0, 188, 120, 24, "graph launch", ACC_F, ACC, ACC, 15, "bold")
for i, nm in enumerate(names):
    s.box(X0 + 24 + i * KW, 226, KW, 26, nm, NEU_F, NEU, INK, 15)
s.text(X0 + 24 + 2.5 * KW, 278, "frozen pointers · shapes · grid", 16, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_graph.svg"))
