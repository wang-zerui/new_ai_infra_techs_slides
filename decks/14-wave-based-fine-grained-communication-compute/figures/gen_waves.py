"""fig_waves: steady-state snapshot of one rank's experts grouped into waves (DeepSeek-V4 §3.1).

Running example: V4-Flash under EP8 -> 32 local experts per rank; 4 waves of 8 experts is schematic
(the real wave size is chosen per call by the heuristic, see fig_wavesize).
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 298
s = SVG(W, H, "fig_waves: W1 sending, W2 computing, W3 receiving, W4 waiting")

CY0, CY1 = 80, 214          # rank container
GX = [28, 238, 448, 658]    # wave group x
GW, GH = 190, 92
GY = CY0 + 12
CW, CHh, CG = 38, 26, 8     # expert cell

s.rect(8, CY0, W - 16, CY1 - CY0, "#FFFFFF", NEU, 1.8, 6)
s.text(W - 22, CY1 - 9, "rank 0 · 32 local experts (V4-Flash, EP8)", 16, MUTED, anchor="end", style="italic")

states = [
    ("W1", "sending results", "send"),
    ("W2", "computing", "comp"),
    ("W3", "receiving tokens", "recv"),
    ("W4", "waiting", "wait"),
]
for i, (name, state, kind) in enumerate(states):
    x = GX[i]
    if kind == "comp":
        s.rect(x, GY, GW, GH, ACC_F, ACC, 3, 5)
    elif kind in ("send", "recv"):
        s.rect(x, GY, GW, GH, "#FFFFFF", ACC, 2, 5, dash=None if kind == "send" else "6,4")
    else:
        s.rect(x, GY, GW, GH, "#FFFFFF", "#A6A6A6", 1.8, 5, dash="4,4")
    col = MUTED if kind == "wait" else ACC
    s.text(x + 10, GY + 22, name, 18, col, anchor="start", weight="bold")
    s.text(x + GW - 10, GY + 22, state, 16, col if kind != "wait" else MUTED, anchor="end",
           weight="bold" if kind == "comp" else "normal")
    # 8 expert cells, 2 rows x 4
    for r in range(2):
        for c in range(4):
            e = i * 8 + r * 4 + c
            cx = x + 10 + c * (CW + CG)
            cy = GY + 34 + r * (CHh + 4)
            if kind == "comp":
                s.box(cx, cy, CW, CHh, f"E{e}", "#FFFFFF", ACC, ACC, 15, sw=1.5, rx=2)
            elif kind == "wait":
                s.box(cx, cy, CW, CHh, f"E{e}", "#FFFFFF", "#BFBFBF", MUTED, 15, sw=1.2, rx=2)
            else:
                s.box(cx, cy, CW, CHh, f"E{e}", "#FFFFFF", ACC, ACC, 15, sw=1.2, rx=2)

# top strip: token owners' FP8 send buffers (other ranks), above W3/W4
s.box(448, 4, 400, 36, "other ranks · FP8 tokens", NEU_F, NEU, INK, 17, sw=1.8)
x3 = GX[2] + GW / 2
s.line(x3, 42, x3, GY - 2, ACC, 2.6, "arrAcc")
s.text(x3 + 10, 66, "pull over NVLink", 16, ACC, anchor="start", weight="bold")

# bottom strip: token owners' combine buffers, below W1/W2
s.box(28, 256, 400, 36, "token owners · combine buffers", NEU_F, NEU, INK, 17, sw=1.8)
x1 = GX[0] + GW / 2
s.line(x1, GY + GH + 2, x1, 254, ACC, 2.6, "arrAcc")
s.text(x1 + 10, 240, "write BF16 results", 16, ACC, anchor="start", weight="bold")

s.save(os.path.join(HERE, "fig_waves.svg"))
