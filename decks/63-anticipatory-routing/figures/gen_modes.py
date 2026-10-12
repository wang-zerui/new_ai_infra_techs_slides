"""fig_modes: spike-triggered use of anticipatory routing (DeepSeek-V4 §4.2.3): an automatic detector triggers a short
rollback and turns anticipatory routing on; after a period, training returns to the standard mode.
Schematic; the paper gives no detector threshold, rollback length or mode duration."""
import math, os
from svgkit import SVG, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 282
s = SVG(W, H, "fig_modes")
X0, X1 = 60, 850
SP, CK, END = 420, 360, 640        # spike, checkpoint, end of anticipatory mode (x)


def base(x):
    u = (x - X0) / (X1 - X0)
    return 90 + 90 * (1 - math.exp(-2.6 * u)) + 5 * math.sin(30 * u)


def seg(xa, xb, color=INK, sw=2.4, dash=None, spike=False):
    pts = []
    for i in range(int(xa), int(xb) + 1, 2):
        y = base(i)
        if spike:
            y -= 105 * math.exp(-((i - SP - 10) / 8.0) ** 2)
        pts.append(f"{i},{y:.1f}")
    s.path("M" + " L".join(pts), color, sw, dash=dash)


# loss curve: standard run up to the spike (abandoned branch dashed red), resumed run from the checkpoint
seg(X0, CK)
seg(CK, SP + 30, PRB, 2.2, dash="5,4", spike=True)
seg(CK, END, ACC, 3.0)
seg(END, X1)
s.text(SP + 10, base(SP + 10) - 118, "spike detected", 17, PRB, weight="bold")
s.add(f'<circle cx="{CK}" cy="{base(CK):.1f}" r="6" fill="#FFFFFF" stroke="{NEU}" stroke-width="2"/>')
s.path(f"M{SP - 4},{base(SP) - 70:.1f} Q{(SP + CK) / 2},{base(SP) - 90:.1f} {CK + 6},{base(CK) - 12:.1f}", MUTED, 2,
       "arrGray", dash="5,4")
s.text(CK - 10, base(CK) - 30, "short rollback", 16, MUTED, anchor="end", style="italic")
s.text(X0 - 8, 100, "loss", 16, MUTED, anchor="end", style="italic")

# mode lane
LY = 212
s.box(X0, LY, CK - X0, 34, "standard training", NEU_F, NEU, INK, 17, sw=1.6)
s.box(CK, LY, END - CK, 34, "anticipatory routing", ACC_F, ACC, ACC, 17, weight="bold", sw=2.2)
s.box(END, LY, X1 - END, 34, "standard training", NEU_F, NEU, INK, 17, sw=1.6)
s.line(X0, 256, X1 + 8, 256, INK, 1.6, "arr")
s.text(X1 + 8, 276, "training step", 15, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_modes.svg"))
