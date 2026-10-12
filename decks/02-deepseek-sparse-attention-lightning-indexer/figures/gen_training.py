"""Slide 11 (mechanism 3): dense-to-sparse continued pre-training (V3.2 Sec. 2.1.1, Eq. 3-4)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "two-stage DSA training")


def two(x, y, w, h, l1, l2, new=False, fill=None):
    s.rect(x, y, w, h, fill or (ACC_F if new else NEU_F), ACC if new else NEU, 2.5 if new else 2, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 19, INK, weight="bold")
    s.text(x + w / 2, y + h / 2 + 19, l2, 17, INK)


# ---------- stage 1
s.rect(2, 2, 408, 256, "#FFFFFF", NEU, 1.6, 8, "6,4")
s.text(16, 28, "Stage 1: dense warm-up", 20, INK, anchor="start", weight="bold")
two(16, 52, 176, 60, "Main model", "frozen · dense")
two(16, 186, 176, 60, "Target pₜ", "head-summed attn")
two(226, 52, 168, 60, "Indexer", "trainable", new=True)
s.rect(226, 186, 168, 60, "#FFFFFF", ACC, 2.5, 6)
s.rich(310, 214, [("KL( p", ""), ("t", "sub"), (" ‖ softmax I", ""), ("t", "sub"), (" )", "")], 18)
s.text(310, 236, "only loss", 16, MUTED, style="italic")
s.arrow(104, 114, 104, 183)
s.arrow(194, 216, 223, 216)
s.arrow(286, 114, 286, 183)
s.path("M352,184 V116", ACC, 2.2, "arrAcc", dash="6,4")
s.text(360, 156, "grad", 16, ACC, anchor="start", weight="bold")

# ---------- hand-off block arrow
s.path("M418,128 H432 V114 L448,144 L432,174 V160 H418 Z", NEU, 1.6, None, None, "#FFFFFF")

# ---------- stage 2
s.rect(456, 2, 412, 256, "#FFFFFF", NEU, 1.6, 8, "6,4")
s.text(470, 28, "Stage 2: sparse training", 20, INK, anchor="start", weight="bold")
two(470, 52, 176, 60, "Main model", "sparse attention")
two(684, 52, 168, 60, "Indexer", "input detached", new=True)
s.rect(470, 186, 176, 60, "#FFFFFF", NEU, 2, 6)
s.text(558, 214, "LM loss", 19, INK, weight="bold")
s.text(558, 236, "trains main model", 16, MUTED, style="italic")
s.rect(684, 186, 168, 60, "#FFFFFF", ACC, 2.5, 6)
s.rich(768, 214, [("KL on top-k set S", ""), ("t", "sub")], 18)
s.text(768, 236, "trains indexer", 16, MUTED, style="italic")
# top-k ids feed the main model's sparse attention
s.arrow(682, 70, 649, 70, ACC, 2.4)
s.text(665, 46, "top-k", 16, ACC, weight="bold")
# detached input: hidden states into the indexer, gradient cut
s.line(648, 96, 681, 96, MUTED, 2, "arrGray", dash="4,3")
s.line(660, 88, 668, 104, ACC, 2.4)
s.line(666, 88, 674, 104, ACC, 2.4)
# gradients
s.path("M630,114 V150 H720 V183", INK, 2.2, "arr")
s.rich(650, 142, [("p", ""), ("t", "sub"), (" on S", ""), ("t", "sub")], 16, anchor="start")
s.path("M524,184 V116", NEU, 2.2, "arr", dash="6,4")
s.path("M810,184 V116", ACC, 2.2, "arrAcc", dash="6,4")
s.text(532, 156, "grad", 16, INK, anchor="start", weight="bold")
s.text(818, 156, "grad", 16, ACC, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_training.svg"))
