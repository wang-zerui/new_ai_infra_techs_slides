"""fig_te: TE make_graphed_callables with the PP schedule passed as _order (2 model chunks, 3 microbatches,
1-indexed chunk ids, positive = forward, negative = backward; example from the dl-cuda-graph TE/Megatron doc).
Graphs are captured ahead of training in exactly this order. FP8: weight casts run only for microbatch 0
(is_first_microbatch / fp8_weight_caching), later microbatches reuse the cached FP8 weights; the amax
reduction runs once per step instead of once per graph (_graph=True during capture).
Sources: transformer_engine/pytorch/graph.py; cuda_graphs.py TECudaGraphHelper; te-megatron-cuda-graphs.html."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED, MONO

HERE = os.path.dirname(os.path.abspath(__file__))
order = [1, 2, 1, 2, 1, 2, -2, -1, -2, -1, -2, -1]
X0, W, G = 132, 56, 5.5
s = SVG(870, 214, "fig_te: _order capture sequence and FP8 handling")

s.text(4, 44, "_order", 18, ACC, anchor="start", weight="bold", family=MONO)
s.text(4, 84, "pass", 17, INK, anchor="start", weight="bold")
s.text(4, 124, "FP8 weights", 17, INK, anchor="start", weight="bold")
fwd_mb = {1: 0, 2: 0}
bwd_mb = {1: 0, 2: 0}
for i, c in enumerate(order):
    x = X0 + i * (W + G)
    fwd = c > 0
    s.box(x, 22, W, 34, str(c).replace("-", "\u2212"), ACC_F if fwd else "#FFFFFF", ACC, color=INK, size=18, weight="bold", sw=2.2)
    if fwd:
        mb = fwd_mb[c]; fwd_mb[c] += 1
        s.text(x + W / 2, 84, f"F{c} m{mb}", 15.5, INK)
        if mb == 0:
            s.box(x + 2, 104, W - 4, 30, "cast", ACC_F, ACC, size=15.5, weight="bold", sw=2)
        else:
            s.box(x + 2, 104, W - 4, 30, "reuse", "#FFFFFF", "#A6A6A6", color=MUTED, size=15.5, dash="4,3")
    else:
        mb = bwd_mb[-c]; bwd_mb[-c] += 1
        s.text(x + W / 2, 84, f"B{-c} m{mb}", 15.5, INK)
s.line(X0, 150, 866, 150, INK, 1.8, "arr")
s.text(X0 + 12 * (W + G) - 4, 168, "capture and replay order", 15, MUTED, anchor="end", style="italic")
s.box(X0, 176, 12 * (W + G) - G, 34, "FP8 amax all-reduce: once per step, not once per graph", "#FFFFFF", ACC,
      color=ACC, size=16, weight="bold", sw=2.2)
s.save(os.path.join(HERE, "fig_te.svg"))
