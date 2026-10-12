"""fig_windows: registration split and the resulting data path. Owner node A: GPU segment is a normal NCCL
window (NVLink + RDMA), the host-backed storage shard is a Gin-only window. Reader node B posts a GET; NIC A
reads host DRAM over PCIe, NIC B writes the row into B's GPU fetch area. No kernel runs on A."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
s = SVG(W, H, "fig_windows: GPU window vs Gin-only storage window, RDMA read path")

# node A (owner)
s.rect(8, 8, 520, 274, "#FFFFFF", NEU, 1.6, 8)
s.text(20, 32, "node A · owner", 18, INK, anchor="start", weight="bold")
VY = 92
s.rect(24, VY, 140, 44, NEU_F, NEU, 1.6, 0)
s.text(94, VY + 28, "GPU segment", 17, INK)
s.rect(164, VY, 240, 44, ACC_F, ACC, 1.8, 0)
s.text(284, VY + 28, "storage shard", 17, ACC, weight="bold")
# window brackets
for (x0, x1, lab, col) in ((26, 162, "NCCL window", INK), (166, 402, "Gin-only window", ACC)):
    s.path(f"M{x0},{VY - 6} L{x0},{VY - 14} L{x1},{VY - 14} L{x1},{VY - 6}", col, 1.6)
    s.text((x0 + x1) / 2, VY - 22, lab, 16, col, weight="bold")
s.text(94, VY + 64, "NVLink + RDMA", 15, MUTED, style="italic")
# physical
s.box(24, 196, 140, 50, "HBM", NEU_F, NEU, size=17)
s.box(184, 196, 200, 50, "host DRAM", ACC_F, ACC, ACC, 17, "bold")
s.line(284, VY + 46, 284, 194, ACC, 1.6, dash="5,4")
s.line(94, VY + 70, 94, 194, NEU, 1.6, dash="5,4")
s.box(426, 186, 88, 64, "NIC", ACC_F, ACC, ACC, 19, "bold")
s.line(386, 221, 424, 221, ACC, 2.6, "arrAcc")
s.text(400, 178, "PCIe read", 15, ACC)
s.text(268, 276, "no kernel runs on A", 15, MUTED, style="italic")

# node B (reader)
s.rect(600, 8, 262, 274, "#FFFFFF", NEU, 1.6, 8)
s.text(612, 32, "node B · reader", 18, INK, anchor="start", weight="bold")
s.box(616, 52, 230, 52, ["issue warp", "posts GET"], "#FFFFFF", NEU, size=17)
s.box(616, 186, 88, 64, "NIC", ACC_F, ACC, ACC, 19, "bold")
s.box(730, 186, 116, 64, ["GPU fetch", "area"], NEU_F, NEU, size=17)
s.line(660, 106, 660, 184, INK, 2, "arr", dash="6,4")
s.text(670, 150, "doorbell", 15, INK, anchor="start", style="italic")
s.line(706, 218, 728, 218, ACC, 2.6, "arrAcc")
# IB between NICs
s.line(516, 218, 614, 218, ACC, 2.8, "arrAcc")
s.text(565, 208, "IB", 16, ACC, weight="bold")
s.save(os.path.join(HERE, "fig_windows.svg"))
