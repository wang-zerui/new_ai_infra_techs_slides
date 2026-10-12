"""Slide 8 (mechanism 2): the fused epilogue for one (token, key) pair.
SM90: fmaxf ReLU, x weight, in-thread sums, 2 shfl_xor, x key scale, masked store (sm90_fp8_mqa_logits.cuh).
SM100: TMEM load, cvt.rn.relu.bf16x2.f32, __hfma2 with BF16 weights, __hadd2 (sm100_mqa_logits.cuh, ptx/utils.cuh)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 244
s = SVG(W, H, "fused epilogue")
MONO = "Menlo, Courier New, monospace"


def node(x, y, w, h, l1, l2=None, hot=False, mono=False):
    s.rect(x, y, w, h, PRB_F if hot else ACC_F, PRB if hot else ACC, 2.2, 6)
    if l2:
        s.text(x + w / 2, y + h / 2 - 3, l1, 17, INK, weight="bold", family=MONO if mono else None)
        s.text(x + w / 2, y + h / 2 + 18, l2, 15, INK)
    else:
        s.text(x + w / 2, y + h / 2 + 6, l1, 17, INK, weight="bold", family=MONO if mono else None)


# ---- (a) SM90
s.text(0, 18, "(a) Hopper (SM90): FP32 registers", 18, INK, anchor="start", weight="bold")
ya, ha = 34, 56
node(0, ya, 120, ha, "64 scores", "per head")
node(150, ya, 96, ha, "ReLU", "fmaxf", hot=True)
node(272, ya, 96, ha, "× w_h", "per head", hot=True)
node(394, ya, 104, ha, "Σ heads", "in thread")
node(524, ya, 100, ha, "shuffle ×2", "4 lanes")
node(650, ya, 96, ha, "× s_k", "once")
node(772, ya, 96, ha, "store", "if in span")
for x1, x2 in ((120, 150), (246, 272), (368, 394), (498, 524), (624, 650), (746, 772)):
    s.arrow(x1 + 2, ya + ha / 2, x2 - 2, ya + ha / 2)
# hot bracket
s.line(150, ya + ha + 10, 368, ya + ha + 10, PRB, 2)
s.text(259, ya + ha + 30, "64 × per (token, key)", 16, PRB, weight="bold")
s.text(698, ya + ha + 30, "ReLU(s·x) = s·ReLU(x), s > 0", 16, MUTED, style="italic")
# ---- (b) SM100
yb = 168
s.line(0, yb - 22, 870, yb - 22, "#BFBFBF", 1.2)
s.text(0, yb - 2, "(b) Blackwell (SM100): TMEM, packed BF16", 18, INK, anchor="start", weight="bold")
hb = 58
node(0, yb + 12, 120, hb, "TMEM", "→ registers")
node(150, yb + 12, 232, hb, "cvt.rn.relu.bf16x2", "ReLU rides on the convert", hot=True, mono=True)
node(408, yb + 12, 150, hb, "hfma2 × w", "2 heads / instr", hot=True, mono=True)
node(584, yb + 12, 136, hb, "hadd2", "reduce pairs", mono=True)
node(746, yb + 12, 122, hb, "store BF16", "if in span")
for x1, x2 in ((120, 150), (382, 408), (558, 584), (720, 746)):
    s.arrow(x1 + 2, yb + 12 + hb / 2, x2 - 2, yb + 12 + hb / 2)
s.save(os.path.join(HERE, "fig_epilogue.svg"))
