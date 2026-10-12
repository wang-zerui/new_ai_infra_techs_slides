"""fig_stack: the GLM-5 Ascend deployment stack, one highlighted layer per pillar."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 358, "fig_stack: model / quantized weights / engine / fused kernels / node, with deck refs")

L, W = 18, 622          # layer left edge and width
ROLE_X = 656            # right-hand role column

def role(y, text):
    s.text(ROLE_X, y, text, 18, MUTED, anchor="start", style="italic")

# ---- model (pre-existing)
s.box(L, 6, W, 44, "GLM-5 · 744B MoE · DSA · 200K context", "#FFFFFF", NEU, size=19)
role(34, "the model")

# ---- pillar 1: quantized weights
s.rect(L, 58, W, 48, ACC_F, ACC, 2.4, 5)
s.text(L + 14, 88, "Mixed W4A8 / W8A8 weights", 19, ACC, anchor="start", weight="bold")
s.text(L + W - 14, 88, "deck 44", 18, ACC, anchor="end")
role(88, "memory")

# ---- pillar 3: inference engine
s.rect(L, 114, W, 90, "#FFFFFF", NEU, 1.5, 6)
s.text(L + 12, 136, "Inference engine · vLLM-Ascend / SGLang", 19, INK, anchor="start", weight="bold")
s.text(L + W - 14, 136, "deck 12", 18, ACC, anchor="end")
labels = ["FlashComm", "DP attn + EP", "PD split", "Prefix cache", "MTP"]
bw, gap = 114, 9
for i, t in enumerate(labels):
    s.box(L + 12 + i * (bw + gap), 150, bw, 44, t, ACC_F, ACC, color=ACC, size=17, weight="bold")
role(165, "exposed time")

# ---- pillar 2: fused kernels
s.rect(L, 212, W, 90, "#FFFFFF", NEU, 1.5, 6)
s.text(L + 12, 234, "Fused kernels · CANN", 19, INK, anchor="start", weight="bold")
s.text(L + W - 14, 234, "deck 37", 18, ACC, anchor="end")
k = ["Lightning Indexer", "Sparse Flash Attn", "MLAPO"]
kw, kgap = 192, 11
for i, t in enumerate(k):
    s.box(L + 12 + i * (kw + kgap), 248, kw, 44, t, ACC_F, ACC, color=ACC, size=17, weight="bold")
role(263, "compute density")

# ---- hardware
s.box(L, 310, W, 42, "Atlas 800 A3 node · 8 × 128 GB NPU memory", NEU_F, NEU, size=18)
role(337, "hardware")

s.save(os.path.join(HERE, "fig_stack.svg"))
