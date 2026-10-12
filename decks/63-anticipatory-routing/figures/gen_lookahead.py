"""fig_lookahead: anticipatory routing as a lookahead pipeline (DeepSeek-V4 §4.2.3), drawn with dt = 2 as an example
(the paper does not state dt). At step s the data of step s+dt is fetched early and a routing-only forward pass with
theta(s), the parameters already resident, computes and caches its routing indices I(s+dt)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 276
s = SVG(W, H, "fig_lookahead")

X0, CWID, BW = 128, 248, 228
LANES = {"data": 40, "route": 108, "train": 206}
BH = 40
names = ["t−2", "t−1", "t"]
ahead = ["t", "t+1", "t+2"]

for lane, y in (("data", "data loader"), ("route", "routing pass"), ("train", "training step")):
    s.text(4, LANES[lane] + BH / 2 + 6, y, 17, MUTED, anchor="start", weight="bold")

for c, (st, la) in enumerate(zip(names, ahead)):
    x = X0 + c * CWID
    s.text(x + BW / 2, 22, f"step {st}", 18, INK, weight="bold")
    s.box(x, LANES["data"], BW, BH, f"fetch batch B({la})", NEU_F, NEU, INK, 17, sw=1.6)
    s.line(x + BW / 2, LANES["data"] + BH + 2, x + BW / 2, LANES["route"] - 2, INK, 2, "arr")
    s.box(x, LANES["route"], BW, BH, f"route B({la}) with θ({st})", ACC_F, ACC, ACC, 17, weight="bold", sw=2)
    s.box(x, LANES["train"], BW, BH, f"train B({st}) with θ({st})", NEU_F, NEU, INK, 17, sw=1.6)

# the narrated hand-off: indices for step t computed at step t-2, used at step t
x0 = X0 + BW / 2 - 40
x2 = X0 + 2 * CWID + BW / 2 - 40
s.path(f"M{x0},{LANES['route'] + BH + 2} L{x0},{178} L{x2},{178} L{x2},{LANES['train'] - 2}", ACC, 2.6, "arrAcc")
s.text((x0 + x2) / 2, 172, "cached indices I(t)", 17, ACC, weight="bold", halo=True)
s.text(W - 8, 270, "each routing pass feeds the step Δt later (Δt = 2 here)", 15, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_lookahead.svg"))
