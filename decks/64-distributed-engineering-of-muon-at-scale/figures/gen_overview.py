"""fig_overview: V4's hybrid ZeRO for Muon (§3.4.1, §2.4). Dense matrices: knapsack whole-matrix buckets on a capped
ZeRO group -> reduce-scatter -> NS per owned matrix. MoE matrices: flatten down|up|gate, cut at matrix edges ->
BF16 stochastic-rounding all-to-all + FP32 local sum -> batched NS. AdamW params keep element-wise ZeRO."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_overview")
cols = [(0, 176), (204, 236), (468, 184), (680, 188)]
heads = ["Weights", "Bucket assignment", "Gradient sync", "Update on owner"]
for (x, w), t in zip(cols, heads):
    s.text(x + w / 2, 22, t, 19, MUTED, weight="bold")

def row(y, h, labels, styles):
    for i, ((x, w), lab, st) in enumerate(zip(cols, labels, styles)):
        if st == "acc":
            s.box(x, y, w, h, lab, ACC_F, ACC, color=ACC, weight="bold", sw=2.4)
        elif st == "gray":
            s.box(x, y, w, h, lab, NEU_F, "#A6A6A6", color=MUTED)
        else:
            s.box(x, y, w, h, lab, "#FFFFFF", NEU)
        if i < 3:
            nx = cols[i + 1][0]
            s.line(x + w + 2, y + h / 2, nx - 3, y + h / 2, INK, 2.2, "arr")

row(40, 74, [["Dense matrices", "attention / mHC / …"], ["❶ Knapsack buckets", "capped ZeRO group"],
             ["Reduce-scatter"], ["Newton-Schulz", "per owned matrix"]], ["n", "acc", "n", "n"])
row(140, 74, [["Expert matrices", "down / up / gate"], ["❷ Flatten, cut at", "matrix edges"],
              ["❸ BF16 all-to-all", "+ FP32 sum"], ["Batched NS", "per shape run"]], ["n", "acc", "acc", "n"])
# AdamW row: one gray band
s.box(0, 240, 176, 46, "Kept on AdamW", NEU_F, "#A6A6A6", color=MUTED)
s.line(178, 263, 201, 263, MUTED, 2, "arrGray")
s.box(204, 240, 664, 46, "embedding / output head / RMSNorm / mHC static biases + gates", NEU_F, "#A6A6A6",
      color=MUTED, size=18)
s.save(os.path.join(HERE, "fig_overview.svg"))
