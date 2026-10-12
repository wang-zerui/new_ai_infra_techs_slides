"""fig_dcp_sched: Dynamic-CP scheduler (cost model -> solver -> simulator) runs asynchronously in the data
sampler; the training loop picks a pre-built CP group per micro-batch (sizes 1, 2, 4, ... up to dp x cp)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 284, "fig_dcp_sched: async scheduler row + training-loop row")

# ---------------- scheduler row (async)
s.rect(6, 8, 858, 128, "#FFFFFF", ACC, 1.5, 6, dash="6,4")
s.text(16, 30, "data sampler (asynchronous)", 18, ACC, anchor="start", weight="bold")
Y, H = 50, 64
s.box(16, Y, 128, H, ["sequence", "lengths"], "#FFF2CC", "#B8860B", size=17)
boxes = [(170, ["Cost model", "work ≈ S² / CP"]), (390, ["Solver", "pack and pick CP"]),
         (610, ["Simulator", "fastest plan in HBM"])]
prev = 144
for x, lab in boxes:
    s.line(prev + 2, Y + H / 2, x - 2, Y + H / 2, INK, 2.2, "arr")
    s.rect(x, Y, 196, H, ACC_F, ACC, 2.2, 5)
    s.text(x + 98, Y + 26, lab[0], 18, ACC, weight="bold")
    s.text(x + 98, Y + 50, lab[1], 17, INK)
    prev = x + 196

# ---------------- training loop
s.rect(6, 150, 858, 128, "#FFFFFF", "#7F7F7F", 1.2, 6, dash="6,4")
s.text(16, 172, "training loop", 18, INK, anchor="start", weight="bold")
# ---------------- hand-off
xa = 708
s.line(xa, Y + H + 3, xa, 192, ACC, 2.4, "arrAcc")
s.text(xa - 12, 182, "cp_size + cp_group per micro-batch", 17, ACC, anchor="end", weight="bold", halo=True)

s.box(16, 196, 170, H, ["pre-built CP groups", "1 · 2 · 4 · 8"], ACC_F, ACC, ACC, 16, "bold")
mbs = ["CP4", "CP1", "CP2", "CP1"]
for i, cp in enumerate(mbs):
    x = 202 + i * 164
    s.box(x, 196, 150, H, [f"micro-batch {i + 1}", cp], NEU_F, NEU, size=17)
s.save(os.path.join(HERE, "fig_dcp_sched.svg"))
