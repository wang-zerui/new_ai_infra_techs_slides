"""fig_overview: who moves the bytes. PP put and Engram get ride the RDMA NIC, CP all-gather rides the copy
engines over NVLink; SMs only post descriptors (doorbell)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 330
s = SVG(W, H, "fig_overview: local GPU (HBM, copy engines, SMs), RDMA NIC, NVLink peer, remote node")

# local GPU container
s.rect(8, 8, 396, 314, "#FFFFFF", NEU, 1.6, 8)
s.text(20, 34, "local GPU", 19, INK, anchor="start", weight="bold")
s.box(24, 50, 120, 258, ["HBM", "symmetric", "buffer"], NEU_F, NEU, size=18)
s.box(176, 50, 212, 70, "", ACC_F, ACC, ACC, 19, "bold")
s.text(282, 80, "Copy Engines", 19, ACC, weight="bold")
s.text(282, 106, "queued by host stream", 16, INK, style="italic")
s.box(176, 142, 212, 62, ["SMs", "post descriptors"], "#FFFFFF", NEU, INK, 18)

# outside boxes
s.box(452, 222, 128, 86, ["RDMA", "NIC"], ACC_F, ACC, ACC, 19, "bold")
s.box(652, 44, 210, 82, ["NVLink peer", "GPU HBM"], "#FFFFFF", NEU, INK, 18)
s.rect(652, 196, 210, 126, "#FFFFFF", NEU, 1.6, 8)
s.text(757, 222, "remote node", 18, INK, weight="bold")
s.box(668, 236, 178, 70, ["GPU HBM or", "host DRAM"], NEU_F, NEU, INK, 18)

# 2: CP all-gather, HBM -> CE -> NVLink -> peer
s.line(146, 85, 174, 85, ACC, 2.6, "arrAcc")
s.line(390, 85, 650, 85, ACC, 2.6, "arrAcc")
s.badge(470, 85, 2, r=13)
s.text(492, 76, "CP all-gather", 18, ACC, anchor="start", weight="bold")
s.text(492, 109, "NVLink", 17, MUTED, anchor="start")

# 1: PP put, HBM -> NIC -> IB -> remote
s.line(146, 252, 450, 252, ACC, 2.6, "arrAcc")
s.line(582, 252, 666, 252, ACC, 2.6, "arrAcc")
s.badge(196, 252, 1, r=13)
s.text(216, 244, "PP put", 18, ACC, anchor="start", weight="bold")
# 3: Engram get, remote -> NIC -> HBM
s.line(666, 286, 584, 286, ACC, 2.6, "arrAcc")
s.line(450, 286, 148, 286, ACC, 2.6, "arrAcc")
s.badge(196, 286, 3, r=13)
s.text(216, 306, "Engram get", 18, ACC, anchor="start", weight="bold")
s.text(624, 240, "IB", 17, MUTED)

# SM doorbell (dashed): SMs -> NIC
s.path("M390,173 L516,173 L516,220", INK, 2, "arr", dash="6,4")
s.text(524, 166, "doorbell", 17, INK, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_overview.svg"))
