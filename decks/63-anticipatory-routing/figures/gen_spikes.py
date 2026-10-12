"""fig_spikes: schematic of the failure DeepSeek-V4 §4.2.3 describes: rollbacks restore the training state but the
loss spikes recur. Curve shape is illustrative; no values are published."""
import math, os
from svgkit import SVG, ACC, PRB, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 262
s = SVG(W, H, "fig_spikes")
X0, X1, Y0, Y1 = 70, 850, 20, 214   # plot box


def base(x):            # smooth decreasing loss in plot coords
    u = (x - X0) / (X1 - X0)
    return Y0 + 70 + 100 * (1 - math.exp(-3.2 * u)) + 6 * math.sin(40 * u)


def curve(xa, xb, spike_at=None, color=INK, dash=None):
    pts = []
    for i in range(int(xa), int(xb) + 1, 2):
        y = base(i)
        if spike_at is not None:
            d = i - spike_at
            if 0 <= d <= 24:
                y -= 120 * math.exp(-((d - 8) / 7.0) ** 2)
        pts.append(f"{i},{y:.1f}")
    s.path("M" + " L".join(pts), color, 2.4, dash=dash)


# axes
s.line(X0, Y1, X1 + 8, Y1, INK, 1.8, "arr")
s.line(X0, Y1, X0, Y0 - 4, INK, 1.8, "arr")
s.text(X1 + 8, Y1 + 22, "training step", 16, MUTED, anchor="end", style="italic")
s.text(X0 - 10, Y0 + 10, "loss", 16, MUTED, anchor="end", style="italic")

# run 1 with a spike, rollback, run 2 with a later spike, rollback, run 3
sp1, ck1 = 300, 250
sp2, ck2 = 600, 548
curve(X0, sp1 + 24, spike_at=sp1)
curve(ck1, sp2 + 24, spike_at=sp2, dash=None)
curve(ck2, X1)
for sp, ck in ((sp1, ck1), (sp2, ck2)):
    top = base(sp + 8) - 120
    s.text(sp + 8, top - 10, "spike", 17, PRB, weight="bold")
    s.add(f'<circle cx="{ck}" cy="{base(ck):.1f}" r="6" fill="#FFFFFF" stroke="{NEU}" stroke-width="2"/>')
    s.path(f"M{sp + 20},{top + 30} Q{(sp + ck) / 2 + 6},{top - 2} {ck + 4},{base(ck) - 14:.1f}", MUTED, 2, "arrGray",
           dash="5,4")
    s.text((sp + ck) / 2 + 8, base(ck) + 30, "rollback", 16, MUTED, style="italic")
# overlay the spikes in red
for sp in (sp1, sp2):
    pts = []
    for i in range(sp - 2, sp + 25, 1):
        d = i - sp
        pts.append(f"{i},{base(i) - 120 * math.exp(-((d - 8) / 7.0) ** 2):.1f}")
    s.path("M" + " L".join(pts), PRB, 2.8)
s.text(X1, Y0 + 40, "spikes come back", 18, PRB, anchor="end", weight="bold")
s.save(os.path.join(HERE, "fig_spikes.svg"))
