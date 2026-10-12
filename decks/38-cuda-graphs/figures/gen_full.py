"""fig_full: FullCudaGraphWrapper. Iterations 1-3 run eager (default --cuda-graph-warmup-steps 3), the next
one is captured, later ones replay. Zoom: per step the dataloader output is copied into static buffers
(StaticBufferLoader), the graph runs forward+backward of all microbatches including PP/DP communication,
the optimizer stays eager. Sources: full_cuda_graph.py; cuda_graph.html; dl-cuda-graph Llama 3.1 405B example."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 236, "fig_full: warmup, capture, replay")

# ---- top: iteration sequence
Y, H = 28, 46
xs = 4
items = [("step 1", "eager"), ("step 2", "eager"), ("step 3", "eager"), ("step 4", "capture"),
         ("step 5", "replay"), ("step 6", "replay")]
widths = {"eager": 142, "capture": 150, "replay": 100}
x = xs
pos = {}
for name, kind in items:
    w = widths[kind]
    if kind == "eager":
        s.box(x, Y, w, H, [name, "eager"], NEU_F, NEU, size=16)
    elif kind == "capture":
        s.box(x, Y, w, H, [name, "capture"], "#FFFFFF", ACC, color=ACC, size=16, weight="bold", sw=2.4, dash="6,3")
    else:
        s.box(x, Y, w, H, [name, "replay"], ACC_F, ACC, size=16, weight="bold", sw=2.4)
    pos[name] = (x, w)
    x += w + 8
s.text(4 + (3 * 142 + 16) / 2, Y - 10, "warmup", 16, MUTED, style="italic")

# ---- zoom of one replay step
ZY, ZH = 150, 70
zx, zw = pos["step 5"]
s.path(f"M{zx},{Y + H + 2} L18,{ZY - 22}", MUTED, 1.4, dash="4,3")
s.path(f"M{zx + zw},{Y + H + 2} L866,{ZY - 22}", MUTED, 1.4, dash="4,3")
s.text(436, ZY - 30, "one replay step", 16, MUTED, style="italic")
s.box(4, ZY, 120, ZH, ["dataloader", "batches"], NEU_F, NEU, size=16)
s.line(126, ZY + ZH / 2, 176, ZY + ZH / 2, INK, 2.2, "arr")
s.text(150, ZY + ZH / 2 - 9, "copy", 15, INK)
# static buffers: one per microbatch (stack effect, offset up-right)
for d in (10, 5):
    s.rect(178 + d, ZY + 4 - d, 108, ZH - 8, "#FFFFFF", ACC, 1.4, 3)
s.box(178, ZY + 4, 108, ZH - 8, ["static input", "buffers"], "#FFFFFF", ACC, size=16, sw=1.8)
s.line(296, ZY + ZH / 2, 322, ZY + ZH / 2, ACC, 2.4, "arrAcc")
s.box(324, ZY - 6, 380, ZH + 12, ["one graph: F + B of all microbatches", "with PP send/recv and DP grad reduce"],
      ACC_F, ACC, size=16, sw=2.6)
s.line(706, ZY + ZH / 2, 730, ZY + ZH / 2, INK, 2.2, "arr")
s.box(732, ZY, 134, ZH, ["optimizer", "eager"], NEU_F, NEU, size=16)
s.save(os.path.join(HERE, "fig_full.svg"))
