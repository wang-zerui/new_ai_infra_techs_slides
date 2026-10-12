"""Overview: the three FP4 paths of V4's post-training QAT ([V4] §5.2.1)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 298
s = SVG(W, H)
X = [24, 250, 476, 702]
BW = 166


def two(i, y, h, l1, l2, hot=False, ghost=False):
    x = X[i]
    s.rect(x, y, BW, h, ACC_FILL if hot else ("#fff" if ghost else GRAY_FILL), ACC if hot else GRAY,
           2.5 if hot else 2, 6)
    s.text(x + BW / 2, y + h / 2 - 4, l1, 18, "bold", INK)
    s.text(x + BW / 2, y + h / 2 + 18, l2, 18, "normal", INK)


# lane 1: expert-weight QAT (training)
yA, h = 6, 56
two(0, yA, h, "FP32 master", "optimizer state", ghost=True)
two(1, yA, h, "MXFP4", "1×32 + E8M0", True)
two(2, yA, h, "FP8 E4M3", "128×128 block")
two(3, yA, h, "FP8 GEMM", "unchanged")
for i in range(3):
    s.arrow(X[i] + BW + 2, yA + h / 2, X[i + 1] - 2, yA + h / 2)
s.path(f"M{X[3] + BW / 2},{yA + h + 2} V{yA + h + 26} H{X[0] + BW / 2} V{yA + h + 4}", INK, 2, dash="6,4")
s.text(X[1] + BW + 30, yA + h + 46, "gradient to master (STE)", 16, "normal", INK, "middle")

# lane 2: indexer QK path
yB = 130
two(0, yB, h, "Indexer Q, K", "FP4 cache", True)
two(1, yB, h, "Q · K", "FP4 multiply", True)
two(2, yB, h, "Index scores", "BF16 (was FP32)", True)
two(3, yB, h, "Top-k", "k = 1,024")
for i in range(3):
    s.arrow(X[i] + BW + 2, yB + h / 2, X[i + 1] - 2, yB + h / 2)

# lane 3: native FP4 in rollout and serving
yC = 218
two(0, yC, h, "FP4 weights", "+ FP4 indexer QK", True)
two(1, yC, h, "RL rollouts", "native FP4")
two(3, yC, h, "Online serving", "native FP4")
s.text(X[2] + BW / 2, yC + 36, "=", 40, "bold", ACC)
s.text(X[2] + BW / 2, yC + 62, "same bits", 18, "bold", ACC)
s.path(f"M{X[0] + BW / 2},{yC + h + 2} V{yC + h + 18} H{X[3] + BW / 2}", ACC, 2.5, head=False)
for i in (1, 3):
    s.arrow(X[i] + BW / 2, yC + h + 18, X[i] + BW / 2, yC + h + 4, ACC, 2.5)

for y, n in ((yA, 1), (yB, 2), (yC, 3)):
    s.badge(12, y + h / 2, n, ACC)
s.save(os.path.join(HERE, "fig_overview.svg"))
