"""fig_pipeline: data path with dynamic batching and packing (VeOmni names; Megatron THD equivalent)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_pipeline")
MONO = "Menlo, monospace"
BW, G, Y, H = 148, 24, 30, 92
items = [(["samples", "any length"], 0), (["❷ buffer +", "token budget"], 1), (["❶ packing", "collator"], 1),
         (["❸ SP / CP", "pad + slice"], 1), (["model", "varlen attention"], 0)]
x = 16
xs = []
for lines, hot in items:
    if hot:
        s.rect(x, Y, BW, H, ACC_F, ACC, 2.4, 6)
        s.text(x + BW / 2, Y + 40, lines[0], 19, ACC, weight="bold")
        s.text(x + BW / 2, Y + 66, lines[1], 17, ACC)
    else:
        s.rect(x, Y, BW, H, NEU_F, NEU, 2, 6)
        s.text(x + BW / 2, Y + 40, lines[0], 19, INK, weight="bold")
        s.text(x + BW / 2, Y + 66, lines[1], 17, MUTED)
    xs.append(x)
    x += BW + G
for a, b in zip(xs, xs[1:]):
    s.line(a + BW + 2, Y + H / 2, b - 2, Y + H / 2, INK, 2.2, "arr")
# outputs of the collator
cx = xs[2] + BW / 2
chips = ["input_ids", "position_ids", "labels", "cu_seqlens"]
CW, CH, CG = 150, 28, 14
tot = 4 * CW + 3 * CG
x0 = cx - tot / 2
s.line(cx, Y + H + 2, cx, 140, ACC, 2)
s.line(x0 + CW / 2, 140, x0 + 3 * (CW + CG) + CW / 2, 140, ACC, 2)
for i, c in enumerate(chips):
    xx = x0 + i * (CW + CG)
    s.line(xx + CW / 2, 140, xx + CW / 2, 158, ACC, 2, "arrAcc")
    s.rect(xx, 160, CW, CH, "#FFFFFF", ACC, 1.8, 4)
    s.text(xx + CW / 2, 160 + 20, c, 16, ACC, family=MONO)
s.text(435, 222, "one micro-batch = one token stream", 18, INK, weight="bold")
s.text(435, 248, "Megatron-Core: THD tensors [T, heads, dim] + cu_seqlens", 17, MUTED, style="italic")
s.save(os.path.join(HERE, "fig_pipeline.svg"))
