"""fig_baseline: the vLLM-style PyTorch full-sampling path of benchmark/sampler/benchmark_sampler.py, one box per
vocabulary-wide tensor op (21 counted; see notes/sources.md), grouped by stage; every op round-trips a B x V
tensor through HBM. FlashInfer's path fuses the top-k/top-p/sample tail but keeps the penalty/temperature ops."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 304
s = SVG(W, H, "fig_baseline")

cols = [
    ("penalty", 8, 188, ["to FP32", "zeros ×2", "scatter ×2", "copy ×2", "apply penalty"]),
    ("temp.", 1, 92, ["÷ T"]),
    ("top-k", 3, 150, ["sort", "compare", "mask"]),
    ("top-p", 5, 184, ["softmax", "cumsum", "compare", "mask", "scatter"]),
    ("sample", 4, 186, ["softmax", "exp noise", "÷ noise", "argmax"]),
]
GAPX = 13
x = 6
HY, BY0, BH, BG = 66, 76, 26, 5
HBM_Y = 270
xs = []
for name, n, w, ops in cols:
    xs.append((x, w))
    s.text(x + w / 2, HY, name, 18, INK, weight="bold")
    s.text(x + w / 2, HY - 20, f"{n} op" + ("s" if n > 1 else ""), 16, PRB)
    for i, op in enumerate(ops):
        s.box(x + 8, BY0 + i * (BH + BG), w - 16, BH, op, PRB_F, PRB, size=16, sw=1.5)
    yb = BY0 + len(ops) * (BH + BG) - BG
    s.line(x + w / 2, yb + 4, x + w / 2, HBM_Y - 3, PRB, 2.2, "arrPrb", start_marker="arrPrb")
    x += w + GAPX
# HBM bar
s.rect(6, HBM_Y, W - 12, 30, PRB_F, PRB, 1.8, 3)
s.text(W / 2, HBM_Y + 21, "HBM: each op reads a full B × V tensor, most write one", 18, PRB, weight="bold")
# FlashInfer bracket over top-k .. sample
xa = xs[2][0] + 4
xb = xs[4][0] + xs[4][1] - 4
s.path(f"M{xa},{30} L{xa},{22} L{xb},{22} L{xb},{30}", MUTED, 1.8)
s.text((xa + xb) / 2, 14, "FlashInfer: one fused kernel for this tail", 16, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_baseline.svg"))
