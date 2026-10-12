"""Slide 2: the kernel contract (DeepGEMM README 'V3.2 MQA kernels for the indexer'; SM90 dtypes)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 484, 338
s = SVG(W, H, "MQA logits kernel contract")


def two(x, y, w, h, l1, l2, fill=NEU_F, stroke=NEU, sw=2):
    s.rect(x, y, w, h, fill, stroke, sw, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, 18, INK, weight="bold")
    s.text(x + w / 2, y + h / 2 + 18, l2, 16, INK)


s.rect(2, 2, 480, 138, "#FFFFFF", NEU, 1.4, 8, "5,4")
two(10, 10, 226, 56, "q · FP8", "n_q × 64 heads × 128")
two(248, 10, 226, 56, "keys · FP8 + scale", "n_kv × (128 + 4 B)")
two(10, 76, 226, 56, "head weights", "n_q × 64")
two(248, 76, 226, 56, "[ks, ke) per query", "which keys to score")
s.arrow(242, 142, 242, 165)
s.rect(2, 168, 480, 64, ACC_F, ACC, 2.6, 6)
s.text(242, 194, "MQA-logits kernel", 20, INK, weight="bold")
s.rich(242, 220, [("Σ", ""), ("h", "sub"), (" w", ""), ("h", "sub"), (" · ReLU( q", ""), ("h", "sub"),
                  (" · k ) · s", ""), ("k", "sub")], 18)
s.arrow(122, 234, 122, 263)
two(2, 266, 240, 66, "logits", "n_q × max span", fill=ACC_F, stroke=ACC, sw=2.4)
s.rect(278, 266, 204, 66, "#FFFFFF", NEU, 2, 6, "6,4")
s.text(380, 293, "top-2,048", 18, INK, weight="bold")
s.text(380, 315, "separate kernel", 16, MUTED, style="italic")
s.arrow(244, 299, 275, 299)
s.save(os.path.join(HERE, "fig_contract.svg"))
