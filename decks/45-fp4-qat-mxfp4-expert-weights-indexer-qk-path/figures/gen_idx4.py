"""Mechanism 2: CSA indexer QK path after QAT ([V4] §2.3.4, §5.2.1; Hadamard step from [code])."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK, RED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 284
s = SVG(W, H)


def two(x, y, w, h, l1, l2, hot=True):
    s.rect(x, y, w, h, ACC_FILL if hot else GRAY_FILL, ACC if hot else GRAY, 2.5 if hot else 2, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 18, "bold", INK)
    s.text(x + w / 2, y + h / 2 + 18, l2, 18, "normal", INK)


two(0, 0, 200, 60, "Indexer queries", "64 × 128 in FP4")
two(0, 96, 200, 60, "Indexer key cache", "n/4 × 128 in FP4")
two(250, 48, 180, 60, "ReLU(q · k)", "FP4 multiply")
two(470, 48, 180, 60, "Weighted sum", "scores in BF16")
two(690, 48, 178, 60, "Top-k", "over n/4 blocks", hot=False)
s.path("M202,30 H340 V46")
s.path("M202,126 H340 V110")
s.arrow(432, 78, 468, 78)
s.arrow(652, 78, 688, 78)
s.text(0, 180, "Hadamard rotation before FP4 quantization (reference code)", 16, "normal", GRAY, "start")

# precision strip: before -> after
def chip(x, y, t, hot):
    s.rect(x, y, 170, 40, ACC_FILL if hot else "#fff", ACC if hot else GRAY, 2, 20)
    s.text(x + 85, y + 26, t, 18, "bold", ACC if hot else GRAY)

s.text(0, 222, "Q, K element", 18, "bold", INK, "start")
chip(170, 196, "FP8 · 1 B (V3.2)", False); s.arrow(344, 216, 376, 216); chip(380, 196, "FP4 · ½ B + scale", True)
s.text(0, 270, "Index score", 18, "bold", INK, "start")
chip(170, 244, "FP32 · 4 B", False); s.arrow(344, 264, 376, 264); chip(380, 244, "BF16 · 2 B", True)
s.save(os.path.join(HERE, "fig_idx4.svg"))
