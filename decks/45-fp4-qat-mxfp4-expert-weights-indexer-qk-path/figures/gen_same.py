"""Mechanism 3: trainer simulates FP4; rollout engine and serving load the same native FP4 bits ([V4] §5.2.1)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 284
s = SVG(W, H)


def container(x, w, title):
    s.rect(x, 0, w, 284, "#fff", GRAY, 1.5, 6, "7,5")
    s.text(x + 14, 26, title, 18, "bold", GRAY, "start")


def chip(x, y, w, l1, l2, hot=True):
    s.rect(x, y, w, 58, ACC_FILL if hot else GRAY_FILL, ACC if hot else GRAY, 2.5 if hot else 2, 6)
    s.text(x + w / 2, y + 25, l1, 18, "bold", INK)
    s.text(x + w / 2, y + 47, l2, 18, "normal", INK)


container(0, 262, "Trainer (QAT)")
chip(14, 46, 234, "FP32 master", "→ FP4 → FP8 forward", hot=False)
chip(14, 126, 234, "Teacher models", "QAT'ed too", hot=False)
chip(14, 206, 234, "Reference model", "QAT'ed too", hot=False)

container(338, 246, "RL rollout engine")
chip(352, 46, 218, "FP4 weights", "native kernels")
chip(352, 126, 218, "FP4 indexer QK", "native kernels")

container(640, 230, "Online serving")
chip(654, 46, 202, "FP4 weights", "native kernels")
chip(654, 126, 202, "FP4 indexer QK", "native kernels")

s.text(612, 88, "=", 36, "bold", ACC)
s.text(612, 168, "=", 36, "bold", ACC)
s.text(755, 236, "same bits as", 18, "bold", ACC)
s.text(755, 258, "the rollouts", 18, "bold", ACC)
# trainer <-> rollout
s.arrow(250, 75, 350, 75, ACC, 2.5)
s.text(300, 66, "publish", 16, "normal", ACC)
s.arrow(350, 195, 264, 195, INK, 2)
s.text(304, 186, "samples", 16, "normal", INK)
s.save(os.path.join(HERE, "fig_same.svg"))
