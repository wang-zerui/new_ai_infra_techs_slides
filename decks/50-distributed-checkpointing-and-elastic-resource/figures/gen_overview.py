"""fig_overview: save once from mesh A, load into mesh B; mechanisms 1-3."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_overview")
MONO = "Menlo, monospace"

def mesh(x, y, w, h, cols, title, sub):
    s.rect(x, y, w, h, "#FFFFFF", NEU, 2, 6)
    s.text(x + w / 2, y + 26, title, 19, INK, weight="bold")
    s.text(x + w / 2, y + 50, sub, 16, MUTED)
    c, g = 20, 4
    gw = cols * (c + g) - g
    x0 = x + (w - gw) / 2
    for r in range(2):
        for k in range(cols):
            s.rect(x0 + k * (c + g), y + 66 + r * (c + g), c, c, NEU_F, NEU, 1.4, 2)

mesh(10, 30, 210, 126, 8, "job A: 16 GPUs", "EP 8 × FSDP 2")
mesh(650, 30, 210, 126, 4, "job B: 8 GPUs", "EP 4 × FSDP 2")

# checkpoint (global view)
CX, CW = 336, 198
s.rect(CX, 22, CW, 142, ACC_F, ACC, 2.2, 6)
s.text(CX + CW / 2, 48, "checkpoint", 19, ACC, weight="bold")
for i, t in enumerate(["per-rank .distcp", ".metadata", "manifest.json"]):
    s.text(CX + CW / 2, 80 + i * 28, t, 16, INK, family=MONO)

# arrows
YM = 93
s.line(224, YM, CX - 4, YM, ACC, 2.6, "arrAcc")
s.text(278, YM - 14, "❶ save", 19, ACC, weight="bold")
s.text(278, YM + 28, "own shard", 16, MUTED)
s.line(CX + CW + 4, YM, 646, YM, ACC, 2.6, "arrAcc")
s.text(592, YM - 14, "❷ load", 19, ACC, weight="bold")
s.text(592, YM + 28, "overlap only", 16, MUTED)

# mechanism 3 band
s.rect(10, 196, 850, 76, "#FFFFFF", ACC, 2, 6, dash="7,4")
s.text(435, 226, "❸ Elastic layout (Kimi K2)", 19, ACC, weight="bold")
s.text(435, 256, "fixed 256-GPU model-parallel unit + ZeRO-1 DP that grows or shrinks", 17, INK)
s.save(os.path.join(HERE, "fig_overview.svg"))
