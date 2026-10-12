"""fig_overview: one transformer layer under TP2 x CP2 on 4 GPUs (redrawn after Megatron CP guide Fig. 1)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 284, "fig_overview: two CP bands (GPU0+GPU1, GPU2+GPU3), token-local ops unchanged, attention swaps KV")

OPS = ["LayerNorm", "QKV Linear", "Attention", "Out Linear", "LayerNorm", "MLP"]
X0, BW, GAP = 196, 98, 13
BH = 50
BANDS = [(52, "CP rank 0", "GPU0 + GPU1 (TP2)", "half of the tokens"),
         (196, "CP rank 1", "GPU2 + GPU3 (TP2)", "other half")]

# legend
s.rect(470, 8, 26, 18, NEU_F, NEU, 2, 3)
s.text(504, 23, "token-local, unchanged", 17, INK, anchor="start")
s.rect(700, 8, 26, 18, ACC_F, ACC, 2, 3)
s.text(734, 23, "needs all KV", 17, INK, anchor="start")

for y, name, gpus, toks in BANDS:
    s.rect(6, y - 10, 858, BH + 34, "#FFFFFF", "#BFBFBF", 1.2, 6, dash="6,4")
    s.text(16, y + 14, name, 19, INK, anchor="start", weight="bold")
    s.text(16, y + 37, gpus, 16, MUTED, anchor="start")
    s.text(16, y + 58, toks, 16, MUTED, anchor="start", style="italic")
    for i, op in enumerate(OPS):
        x = X0 + i * (BW + GAP)
        attn = op == "Attention"
        s.box(x, y + 2, BW, BH, op, ACC_F if attn else NEU_F, ACC if attn else NEU,
              ACC if attn else INK, 16, "bold" if attn else "normal", sw=2.2 if attn else 2)
        if i:
            s.line(x - GAP + 2, y + 2 + BH / 2, x - 1, y + 2 + BH / 2, INK, 2, "arr")

# KV exchange between the two attention boxes
xa = X0 + 2 * (BW + GAP) + BW / 2
s.line(xa, 52 + 2 + BH + 4, xa, 196 + 2 - 4, ACC, 2.6, "arrAcc", start_marker="arrAcc")
s.text(xa + 12, 140, "KV exchange", 18, ACC, anchor="start", weight="bold", halo=True)
s.text(xa + 12, 160, "in a ring", 18, ACC, anchor="start", weight="bold", halo=True)
s.save(os.path.join(HERE, "fig_overview.svg"))
