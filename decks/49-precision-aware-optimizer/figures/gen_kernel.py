"""fig_kernel: TE FusedAdam with --use-precision-aware-optimizer (BF16 params, FP32 main grads, BF16/FP8 moments,
FP32 master stored as an int16 remainder): load compact state, compute in FP32 registers, write compact state."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 296, "fig_kernel: stored state -> FP32 kernel -> stored state")

CH, GAP = 40, 9
left = [("BF16 weight", "2 B", False), ("FP32 gradient", "4 B", False), ("master remainder", "2 B", True),
        ("first moment m", "1–2 B", True), ("second moment v", "1–2 B", True)]
right = [("BF16 weight", False), ("master remainder", True), ("first moment m", True), ("second moment v", True)]
LX, LW = 8, 248
RX, RW = 646, 216
MX, MW, MY, MH = 340, 222, 64, 196

s.text(LX, 22, "stored in HBM", 18, INK, anchor="start", weight="bold")
s.text(RX, 22, "written back", 18, INK, anchor="start", weight="bold")
ys = []
for i, (name, b, new) in enumerate(left):
    y = 36 + i * (CH + GAP)
    ys.append(y + CH / 2)
    s.rect(LX, y, LW, CH, ACC_F if new else NEU_F, ACC if new else NEU, 2, 4)
    s.text(LX + 12, y + 26, name, 17, ACC if new else INK, anchor="start", weight="bold" if new else "normal")
    s.text(LX + LW - 10, y + 26, b, 17, ACC if new else INK, anchor="end")
# kernel box
s.rect(MX, MY, MW, MH, "#FFFFFF", ACC, 2.6, 6)
s.text(MX + MW / 2, MY + 28, "FusedAdam kernel", 18, ACC, weight="bold")
for k, t in enumerate(["upcast to FP32", "Adam math in FP32", "round and split"]):
    yy = MY + 56 + k * 46
    s.rect(MX + 14, yy, MW - 28, 34, ACC_F, ACC, 1.6, 4)
    s.text(MX + MW / 2, yy + 23, t, 16, INK)
    if k:
        s.line(MX + MW / 2, yy - 11, MX + MW / 2, yy - 1, INK, 1.8, "arr")
# fan-in arrows
for y in ys:
    s.line(LX + LW + 3, y, MX - 3, MY + 20 + (y - 36) * (MH - 40) / (5 * (CH + GAP)), INK, 1.8, "arr")
# right chips and fan-out
for i, (name, new) in enumerate(right):
    y = 60 + i * (CH + GAP + 6)
    s.rect(RX, y, RW, CH, ACC_F if new else NEU_F, ACC if new else NEU, 2, 4)
    s.text(RX + 12, y + 26, name, 17, ACC if new else INK, anchor="start", weight="bold" if new else "normal")
    s.line(MX + MW + 3, MY + 40 + i * (MH - 60) / 3, RX - 3, y + CH / 2, INK, 1.8, "arr")
s.save(os.path.join(HERE, "fig_kernel.svg"))
