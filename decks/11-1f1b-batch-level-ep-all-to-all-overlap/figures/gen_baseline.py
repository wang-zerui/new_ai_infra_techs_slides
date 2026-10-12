"""fig_baseline: one micro-batch, forward of two MoE layers; every all-to-all sits on the critical path."""
import os
from svgkit import SVG, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_baseline: sequential attention -> dispatch -> experts -> combine, compute idles during a2a")

Y_C, Y_N, H = 66, 140, 48          # compute lane, network lane
X0 = 148
seg = [("Attention", 108, "c"), ("dispatch", 84, "n"), ("Experts", 84, "c"), ("combine", 78, "n")]

s.text(136, Y_C + H / 2 + 6.3, "compute", 18, INK, anchor="end", weight="bold")
s.text(136, Y_N + H / 2 + 6.3, "network", 18, INK, anchor="end", weight="bold")

x = X0
prev = None
layer_starts = []
for layer in range(2):
    layer_starts.append(x)
    for name, w, lane in seg:
        if lane == "c":
            s.box(x, Y_C, w, H, name, NEU_F, NEU)
        else:
            s.box(x, Y_N, w, H, name, PRB_F, PRB, color=PRB)
            s.box(x, Y_C, w, H, "idle", "#FFFFFF", "#A6A6A6", color=MUTED, dash="6,4", sw=1.8)
        # dependency hand-off at the boundary
        if prev is not None:
            if prev == "c" and lane == "n":
                s.line(x, Y_C + H + 2, x, Y_N - 1, INK, 2, "arr")
            elif prev == "n" and lane == "c":
                s.line(x, Y_N - 2, x, Y_C + H + 1, INK, 2, "arr")
        prev = lane
        x += w
x_end = x
mid = layer_starts[1]
s.line(mid, 40, mid, Y_N + H + 8, MUTED, 1.5, dash="6,5")
s.text((layer_starts[0] + mid) / 2, 34, "MoE layer ℓ", 18, MUTED, style="italic")
s.text((mid + x_end) / 2, 34, "MoE layer ℓ+1", 18, MUTED, style="italic")
# time axis
s.line(X0 - 4, 214, 862, 214, INK, 2, "arr")
s.text(858, 242, "time", 18, INK, anchor="end")
s.text(X0, 242, "one micro-batch, forward pass", 17, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_baseline.svg"))
