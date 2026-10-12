"""fig_rules: what a captured CUDA graph requires vs what a training iteration does.
Sources: arXiv 2603.07685 Sec. 4.3.6-4.3.7 (static shapes, D2H token counts, PP pool corruption);
docs cuda_graph.html (NaN check needs --no-check-for-nan-in-loss-and-grad); dl-cuda-graph TE/Megatron doc
(FP8 global buffers, amax reduction, PP replay order); full_cuda_graph.py (static input buffers)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [
    ("fixed tensor shapes", "dropless MoE: tokens per expert vary"),
    ("no host sync inside", "D2H token counts and NaN checks"),
    ("fixed input addresses", "fresh dataloader tensors every step"),
    ("replay in capture order", "PP interleaves microbatches in one pool"),
    ("same work on every replay", "FP8 amax reduce once per step, not per layer"),
]
H, G, Y0 = 40, 10, 34
s = SVG(870, Y0 + len(rows) * (H + G), "fig_rules: graph requirements vs training behavior")
s.text(130, 20, "a captured graph needs", 18, INK, weight="bold")
s.text(600, 20, "a training iteration has", 18, PRB, weight="bold")
for i, (need, brk) in enumerate(rows):
    y = Y0 + i * (H + G)
    s.box(2, y, 256, H, need, NEU_F, NEU, size=18)
    s.box(330, y, 538, H, brk, PRB_F, PRB, color=PRB, size=18)
    s.text(294, y + H / 2 + 8, "✗", 24, PRB, weight="bold")
s.save(os.path.join(HERE, "fig_rules.svg"))
