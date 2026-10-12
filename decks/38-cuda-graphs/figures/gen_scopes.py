"""fig_scopes: one training iteration without PP (3 layers x 2 microbatches, F = forward, B = backward,
then the optimizer) and the capture scope of each Megatron-Core mode.
Concept from arXiv 2603.07685 Fig. 23-24 and the cuda_graph.html overview table (redrawn)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, MONO

HERE = os.path.dirname(os.path.abspath(__file__))
seq = ["F11", "F12", "F13", "B13", "B12", "B11", "F21", "F22", "F23", "B23", "B22", "B21"]
X0, W, G, H = 196, 46, 4, 40
s = SVG(870, 292, "fig_scopes: capture scope per cuda_graph_impl")

# legend
s.rect(196, 6, 22, 14, "#FFFFFF", ACC, 2.6, 2)
s.text(224, 18, "one CUDA graph", 15, INK, anchor="start")
s.rect(354, 6, 22, 14, "url(#hatchPrb)", PRB, 1.4, 2)
s.text(382, 18, "eager: dynamic shapes", 15, INK, anchor="start")
s.rect(566, 6, 22, 14, NEU_F, NEU, 1.4, 2)
s.text(594, 18, "eager", 15, INK, anchor="start")


def label(y, flag, sub):
    s.text(4, y + 17, flag, 16, ACC, anchor="start", weight="bold", family=MONO)
    s.text(4, y + 37, sub, 15, MUTED, anchor="start")


def opt(y):
    x = X0 + 12 * (W + G) + 8
    s.box(x, y, 52, H, "opt", NEU_F, NEU, size=15)


rows = [44, 124, 204]
# row 1: full iteration
y = rows[0]
label(y, "full_iteration", "one graph per step")
for i, t in enumerate(seq):
    s.box(X0 + i * (W + G), y, W, H, t, "#FFFFFF", NEU, size=15, sw=1.4)
s.rect(X0 - 5, y - 5, 12 * (W + G) - G + 10, H + 10, "none", ACC, 2.8, 5)
opt(y)
# row 2: per layer
y = rows[1]
label(y, "local / TE", "one graph per layer pass")
for i, t in enumerate(seq):
    s.box(X0 + i * (W + G), y, W, H, t, "#FFFFFF", ACC, size=15, sw=2.6)
opt(y)
# row 3: per module (MoE)
y = rows[2]
label(y, "partial (MoE)", "graph up to dispatch")
for i, t in enumerate(seq):
    x = X0 + i * (W + G)
    if t.startswith("F"):
        s.rect(x + 26, y, W - 26, H, "url(#hatchPrb)", PRB, 1.4, 2)
        s.rect(x, y, 26, H, "#FFFFFF", ACC, 2.6, 2)
    else:
        s.rect(x, y, W - 26, H, "url(#hatchPrb)", PRB, 1.4, 2)
        s.rect(x + W - 26, y, 26, H, "#FFFFFF", ACC, 2.6, 2)
opt(y)
s.line(X0, 258, X0 + 12 * (W + G) + 60, 258, INK, 1.8, "arr")
s.text(X0, 284, "Fij / Bij: forward / backward of microbatch i, layer j", 15, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_scopes.svg"))
