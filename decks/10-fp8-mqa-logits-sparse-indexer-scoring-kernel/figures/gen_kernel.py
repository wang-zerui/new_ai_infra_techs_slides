"""Slide 6 (overview): one persistent CTA of the SM90 non-paged MQA-logits kernel (sm90_fp8_mqa_logits.cuh)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "SM90 MQA logits CTA anatomy")


def two(x, y, w, h, l1, l2, fill=NEU_F, stroke=NEU, sw=2, s1=18, s2=16):
    s.rect(x, y, w, h, fill, stroke, sw, 6)
    s.text(x + w / 2, y + h / 2 - 4, l1, s1, INK, weight="bold")
    s.text(x + w / 2, y + h / 2 + 18, l2, s2, INK)


# HBM inputs
s.text(70, 18, "HBM", 18, MUTED, weight="bold")
two(0, 30, 140, 58, "Q tile", "2 tok × 64 heads", NEU_F, NEU)
two(0, 112, 140, 58, "keys + scales", "streamed", NEU_F, NEU)
# TMA warp
two(178, 70, 112, 60, "TMA", "warp", ACC_F, ACC, 2.4)
s.arrow(142, 59, 176, 88)
s.arrow(142, 141, 176, 114)
# SMEM rings
s.text(400, 18, "shared memory", 18, MUTED, weight="bold")
s.rect(328, 30, 150, 58, "#FFFFFF", ACC, 2, 6)
s.text(403, 52, "Q ring", 17, INK, weight="bold")
for i in range(3):
    s.rect(350 + i * 38, 60, 30, 20, ACC_F, ACC, 1.4, 2)
s.rect(328, 112, 150, 58, "#FFFFFF", ACC, 2, 6)
s.text(403, 134, "KV ring", 17, INK, weight="bold")
for i in range(3):
    s.rect(350 + i * 38, 142, 30, 20, ACC_F, ACC, 1.4, 2)
s.arrow(292, 92, 326, 66)
s.arrow(292, 108, 326, 136)
s.text(403, 190, "3 stages each", 16, MUTED, style="italic")
# math warpgroups
s.text(632, 18, "4 math warpgroups", 18, MUTED, weight="bold")
for i in range(4):
    y = 28 + i * 40
    s.rect(540, y, 186, 34, ACC_F, ACC, 2, 5)
    s.text(633, y + 23, f"WG{i}: keys {64*i}–{64*i+63}", 16, INK, weight="bold")
s.arrow(480, 59, 537, 59)
s.arrow(480, 141, 537, 141)
s.badge(509, 100, 1)
# epilogue -> logits
two(756, 62, 112, 68, "logits", "[ks, ke) only", "#FFFFFF", ACC, 2.4)
s.arrow(728, 96, 753, 96, ACC, 2.4)
s.badge(741, 74, 2)
s.text(812, 152, "fused", 16, ACC, weight="bold")
s.text(812, 172, "epilogue", 16, ACC, weight="bold")
# persistent loop
s.path("M633,190 V230 H70 V172", MUTED, 1.8, "arrGray", dash="6,4")
s.text(352, 252, "persistent CTA: grid = #SMs, next Q block = this + #SMs; Q loaded once per block", 16, MUTED,
       style="italic")
s.save(os.path.join(HERE, "fig_kernel.svg"))
