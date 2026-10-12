"""Slide 4: unfused (GEMM writes per-head scores to HBM, a 2nd kernel reduces them) vs one fused kernel."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 268
s = SVG(W, H, "unfused vs fused indexer")


def two(x, y, w, h, l1, l2, fill, stroke, sw=2, c2=INK):
    s.rect(x, y, w, h, fill, stroke, sw, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 18, INK, weight="bold")
    s.text(x + w / 2, y + h / 2 + 18, l2, 16, c2)


# row (a)
s.text(0, 20, "(a) GEMM + elementwise kernel", 19, INK, anchor="start", weight="bold")
two(0, 52, 150, 64, "GEMM", "q · kᵀ per head", NEU_F, NEU)
for d in (12, 6, 0):
    s.rect(196 + d, 36 - d + 6, 250, 92, PRB_F if d == 0 else "#FFFFFF", PRB, 2.2, 4)
s.text(321, 74, "per-head scores in HBM", 18, INK, weight="bold")
s.text(321, 98, "n_q × 64 × n_kv · FP32", 17, PRB, weight="bold")
s.arrow(152, 84, 193, 84, PRB, 2.4)
two(500, 52, 170, 64, "ReLU · w · Σ", "2nd kernel", NEU_F, NEU)
s.arrow(460, 84, 497, 84, PRB, 2.4)
s.text(478, 72, "read", 15, PRB, weight="bold")
s.text(173, 72, "write", 15, PRB, weight="bold")
two(716, 60, 150, 48, "logits", "n_q × n_kv", NEU_F, NEU)
s.arrow(672, 84, 713, 84)
# row (b)
s.line(0, 152, 870, 152, "#BFBFBF", 1.2)
s.text(0, 180, "(b) one fused kernel (DeepGEMM)", 19, INK, anchor="start", weight="bold")
s.rect(0, 196, 670, 64, ACC_F, ACC, 2.6, 6)
s.text(335, 222, "MMA  →  ReLU · w  →  Σ heads  →  × key scale", 18, INK, weight="bold")
s.text(335, 246, "all in registers / shared memory", 16, INK)
two(716, 204, 150, 48, "logits", "n_q × span", ACC_F, ACC, 2.4)
s.arrow(672, 228, 713, 228, ACC, 2.4)
s.save(os.path.join(HERE, "fig_unfused.svg"))
