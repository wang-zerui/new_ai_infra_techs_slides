"""fig_sync: (a) a host-shaped grouped GEMM: the CPU waits for each layer's expert counts before it can size and
launch the GEMM, the GPU idles, and the step cannot be captured; (b) a worst-case batched GEMM is capturable but
computes every capacity row."""
import os
from tilekit import SVG, seg, waste, outline, INK, MUTED, PRB, PRB_F, NEU, NEU_F, ACC

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 292, "fig_sync: host-shaped grouped GEMM vs worst-case batched GEMM")

# ---------- (a) shapes on the host
s.text(10, 24, "(a) GEMM shaped on the host", 19, INK, anchor="start", weight="bold")
s.text(56, 86, "CPU", 17, INK, anchor="end", weight="bold")
s.text(56, 168, "GPU", 17, INK, anchor="end", weight="bold")
s.box(66, 148, 100, 32, "dispatch", NEU_F, NEU, INK, 16)
s.rect(166, 148, 196, 32, "url(#hatchPrb)", PRB, 1.2, rx=0)
s.text(264, 169, "GPU idle", 16, PRB, weight="bold", halo=True)
s.box(362, 148, 100, 32, "GEMM", NEU_F, NEU, INK, 16)
s.box(176, 66, 112, 32, "wait counts", PRB_F, PRB, PRB, 16, "bold")
s.box(296, 66, 86, 32, "launch", NEU_F, NEU, INK, 16)
s.line(160, 146, 196, 100, PRB, 2, "arrPrb")
s.text(140, 124, "D2H copy", 15, PRB, anchor="end", weight="bold")
s.line(372, 100, 396, 146, INK, 2, "arr")
s.line(66, 206, 470, 206, INK, 1.6, "arr")
s.text(466, 224, "time", 15, MUTED, anchor="end")
s.text(240, 252, "once per MoE layer", 17, PRB, weight="bold")
s.text(240, 274, "shape depends on GPU data: no capture", 16, PRB)

s.line(488, 14, 488, 286, "#999999", 1)

# ---------- (b) worst-case batched GEMM
s.text(508, 24, "(b) Batched GEMM at worst case", 19, INK, anchor="start", weight="bold")
CAP, BASE = 170, 216
xs = [534, 640, 746]
valid = [96, 22, 64]
for e, (x, v) in enumerate(zip(xs, valid)):
    seg(s, x, BASE - CAP, 80, v, e, f"E{e}", size=17)
    waste(s, x, BASE - CAP + v, 80, CAP - v, None)
    outline(s, x, BASE - CAP, 80, CAP, INK, 1.8)
s.text(680, 252, "every capacity row computed", 17, PRB, weight="bold")
s.text(680, 274, "graph-safe but wasteful", 16, PRB)
s.save(os.path.join(HERE, "fig_sync.svg"))
