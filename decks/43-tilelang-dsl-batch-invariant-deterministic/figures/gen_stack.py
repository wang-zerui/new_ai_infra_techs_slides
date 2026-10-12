"""fig_stack: overview of DeepSeek-V4's reproducible kernel stack (DeepSeek-V4 §3.2-3.3).
❶ TileLang compiler features, ❷ batch-invariant forward kernels, ❸ deterministic backward kernels."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED, badge

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 346
s = SVG(W, H, "fig_stack")

# consumers
for x, lab in ((10, "pre-training"), (300, "RL rollout + trainer"), (590, "inference serving")):
    s.box(x, 4, 270, 38, lab, NEU_F, NEU, INK, 18, sw=1.8)
    s.line(x + 135, 66, x + 135, 44, INK, 2.2, "arr")
s.rect(4, 68, 862, 182, "#FFFFFF", ACC, 1.5, 8, dash="6,4")

cols = [  # x, w, title, chips [(label, badges)]
    (10, 300, "Attention kernels", [("dual-kernel decoding", [2]),
                                    ("per-SM dKV buffers", [3])]),
    (322, 300, "DeepGEMM GEMMs", [("no cuBLAS", [2]), ("no split-K by default", [2]),
                                  ("mHC: fixed 16-way split", [2, 3])]),
    (634, 226, "MoE backward", [("fixed token order", [3]), ("isolated rank buffers", [3])]),
]
for x, w, title, chips in cols:
    s.rect(x, 78, w, 164, ACC_F, ACC, 2, 6)
    s.text(x + 12, 102, title, 18, ACC, anchor="start", weight="bold")
    for i, (lab, bs) in enumerate(chips):
        y = 114 + i * 42
        s.rect(x + 10, y, w - 20, 34, "#FFFFFF", ACC, 1.5, 4)
        s.text(x + 22, y + 23, lab, 16, INK, anchor="start")
        for j, b in enumerate(reversed(bs)):
            badge(s, x + w - 26 - j * 26, y + 17, b)

# compiler
s.rect(10, 276, 400, 66, "#FFFFFF", ACC, 2, 6)
s.text(22, 297, "TileLang compiler", 18, ACC, anchor="start", weight="bold")
badge(s, 390, 292, 1)
for x, w, lab in ((20, 122, "host codegen"), (150, 112, "Z3 analysis"), (270, 130, "exact numerics")):
    s.box(x, 306, w, 30, lab, ACC_F, ACC, INK, 16, sw=1.5)
s.line(160, 274, 160, 244, ACC, 2.4, "arrAcc")
s.text(172, 265, "builds most fused kernels", 16, ACC, anchor="start", style="italic")

# legend for the badges
for i, (n, lab) in enumerate(((1, "compiler and numerics"), (2, "batch-invariant forward"),
                              (3, "deterministic backward"))):
    y = 290 + i * 22
    badge(s, 450, y, n)
    s.text(468, y + 6, lab, 16, INK, anchor="start")
s.save(os.path.join(HERE, "fig_stack.svg"))
