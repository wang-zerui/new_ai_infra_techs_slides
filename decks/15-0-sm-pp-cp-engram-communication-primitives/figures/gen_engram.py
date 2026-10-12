"""fig_engram: Engram fetch. Deterministic hashed indices -> issue warps post one RDMA GET per row (batched
doorbell) -> NIC reads rows from remote storage shards (GPU HBM or host DRAM) -> local fetch area, one wait
hook per Engram layer (layers 2 and 15 as in Engram-27B)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 286
s = SVG(W, H, "fig_engram: Engram fetch over RDMA GET")

# column 1: indices and issue warps
s.box(16, 16, 172, 44, "token IDs", NEU_F, NEU, size=18)
s.line(102, 62, 102, 98, INK, 2.2, "arr")
s.badge(102, 80, 1)
s.text(122, 86, "hash", 17, INK, anchor="start", style="italic")
s.box(16, 100, 172, 62, ["row indices", "[layer, token, entry]"], "#FFFFFF", NEU, size=17)
s.line(102, 164, 102, 196, INK, 2.2, "arr")
s.box(16, 198, 172, 62, ["issue warps", "on SMs"], "#FFFFFF", NEU, size=18)

# column 2: local fetch area
s.rect(222, 16, 210, 196, "#FFFFFF", ACC, 1.8, 8)
s.text(327, 40, "local fetch area", 18, ACC, weight="bold")
for b, (lab, filled) in enumerate((("layer 2", (0, 2, 3)), ("layer 15", (1, 4)))):
    y = 58 + b * 58
    s.text(236, y + 16, lab, 16, INK, anchor="start")
    for c in range(6):
        on = c in filled
        s.rect(236 + c * 30, y + 24, 26, 22, ACC_F if on else "#FFFFFF", ACC if on else "#A6A6A6", 1.3, 2,
               dash=None if on else "3,3")
s.badge(250, 192, 4)
s.text(268, 198, "wait hook per layer", 16, INK, anchor="start", style="italic")

# NIC
s.box(482, 92, 84, 92, "NIC", ACC_F, ACC, ACC, 20, "bold")
s.line(480, 126, 434, 126, ACC, 2.6, "arrAcc")
s.text(458, 116, "write", 15, ACC)

# doorbell from issue warps
s.path("M190,229 L524,229 L524,187", INK, 2, "arr", dash="6,4")
s.badge(300, 229, 2)
s.text(318, 254, "post 1 GET per row", 17, INK, anchor="start")

# remote storage
s.rect(604, 8, 258, 270, "#FFFFFF", NEU, 1.6, 8)
s.text(733, 32, "remote storage shards", 18, INK, weight="bold")
s.text(733, 54, "GPU HBM or host DRAM", 16, MUTED, style="italic")
for k, hot in ((0, 2), (1, 1)):
    y0 = 70 + k * 104
    s.rect(620, y0, 226, 92, "#FFFFFF", NEU, 1.4, 5)
    s.text(836, y0 + 20, f"rank {k + 1}", 16, INK, anchor="end", weight="bold")
    for r in range(4):
        on = r == hot
        s.rect(632, y0 + 28 + r * 14, 150, 13, ACC_F if on else NEU_F, ACC if on else "#A6A6A6", 1.2, 0)
    hy = y0 + 28 + hot * 14 + 6.5
    s.path(f"M630,{hy} C600,{hy} 600,{140 + 14 * k} 568,{140 + 14 * k}", ACC, 2.4, "arrAcc")
s.badge(482, 72, 3)
s.text(500, 78, "RDMA read", 17, ACC, anchor="start", weight="bold")
s.save(os.path.join(HERE, "fig_engram.svg"))
