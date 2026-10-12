"""fig_hbm: the eager router path (Megatron moe_utils.topk_routing_with_score_function, unfused branch,
sigmoid + expert bias + group-limited top-k + top-8 + normalization), its 22 ops grouped into 10 steps.
Every step reads its input from HBM and writes its output back, and each op is a separate launch."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
steps = [["sigmoid"], ["+ expert", "bias"], ["group", "scores"], ["keep 4", "groups"], ["mask", "fill"],
         ["top-8"], ["gather"], ["normal-", "ize"], ["scatter", "probs"], ["scatter", "map"]]
W, G, X0 = 78, 9, 2
s = SVG(870, 222, "fig_hbm: eager router ops and their HBM round trips")
s.box(X0, 6, 10 * (W + G) - G, 34, "HBM: [tokens × 256] score tensors", NEU_F, NEU, size=17)
YB, HB = 100, 52
for i, lab in enumerate(steps):
    x = X0 + i * (W + G)
    s.box(x, YB, W, HB, lab, "#FFFFFF", NEU, size=16)
    s.line(x + W / 2 - 12, 42, x + W / 2 - 12, YB - 2, PRB, 2.2, "arrPrb")
    s.line(x + W / 2 + 12, YB - 2, x + W / 2 + 12, 42, PRB, 2.2, "arrPrb")
    s.rect(x + W / 2 - 14, 176, 28, 14, "#595959", "#595959", 1, 2)
s.text(X0 + 2, 72, "read", 15, PRB, anchor="start", halo=True)
s.text(X0 + W / 2 + 18, 72, "write", 15, PRB, anchor="start", halo=True)
s.text(X0, 214, "CPU: one launch per op (22 ops in total)", 16, INK, anchor="start")
s.line(X0, 198, X0 + 10 * (W + G) - G, 198, MUTED, 1.2)
s.save(os.path.join(HERE, "fig_hbm.svg"))
