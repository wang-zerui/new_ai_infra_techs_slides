"""fig_sphere_1/2: QK-Norm puts every q and k on a sphere of radius sqrt(d) (unit gains), so the logit is bounded.

Step 1: raw q/k vectors whose length grows with ||W_q||, ||W_k|| -> per-head RMSNorm -> all on the circle.
Step 2: + RoPE as a rotation on the circle (norm preserved) + the bound |logit| <= sqrt(d) g_q g_k = 11.3 (d = 128, g = 1).
Analytic; derivation in notes/sources.md. RMSNorm then RoPE order from transformers modeling_glm4_moe.py.
"""
import math, os
from svgkit import SVG, rich, ACC, ACC_F, PRB, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 280
ANG = [18, 42, 70]
LEN = [70, 135, 200]


def draw(step):
    s = SVG(W, H, f"fig_sphere step {step}")
    # left: raw vectors
    ox, oy = 30, 250
    s.text(0, 22, "raw q, k", 20, INK, anchor="start", weight="bold")
    for a, L in zip(ANG, LEN):
        x2, y2 = ox + L * math.cos(math.radians(a)), oy - L * math.sin(math.radians(a))
        col = PRB if L > 100 else INK
        s.line(ox, oy, x2, y2, col, 2.6, "arrPrb" if L > 100 else "arr")
    s.add(f'<circle cx="{ox}" cy="{oy}" r="3.5" fill="{INK}"/>')
    rich(s, 0, 50, [("length grows with ‖W",), ("q", "sub"), ("‖",)], 17, anchor="start", color=PRB, style="italic")
    # block arrow
    bx0, bx1, by = 262, 372, 150
    s.path(f"M{bx0},{by-16} L{bx1-24},{by-16} L{bx1-24},{by-30} L{bx1},{by} L{bx1-24},{by+30} L{bx1-24},{by+16} "
           f"L{bx0},{by+16} Z", ACC, 2, fill=ACC_F)
    s.text((bx0 + bx1) / 2 - 8, by - 42, "RMSNorm", 19, ACC, weight="bold")
    s.text((bx0 + bx1) / 2 - 8, by + 56, "per head", 19, ACC, weight="bold")
    # right: circle
    cx, cy, R = 506, 150, 92
    s.text(cx - R - 10, 22, "after QK-Norm", 20, ACC, anchor="start", weight="bold")
    s.add(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{ACC}" stroke-width="2" stroke-dasharray="6,4"/>')
    for a in ANG:
        x2, y2 = cx + R * math.cos(math.radians(a)), cy - R * math.sin(math.radians(a))
        s.line(cx, cy, x2, y2, ACC, 2.6, "arrAcc")
    s.add(f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="{INK}"/>')
    # radius label along the lowest vector
    rich(s, cx + 48, cy + 26, [("radius √d",)], 18, color=ACC)
    s.text(cx, cy + R + 22, "(unit gains)", 16, MUTED, style="italic")
    if step >= 2:
        a0, a1 = 70, 125
        r2 = R + 14
        p0 = (cx + r2 * math.cos(math.radians(a0)), cy - r2 * math.sin(math.radians(a0)))
        p1 = (cx + r2 * math.cos(math.radians(a1)), cy - r2 * math.sin(math.radians(a1)))
        s.path(f"M{p0[0]:.1f},{p0[1]:.1f} A{r2},{r2} 0 0 0 {p1[0]:.1f},{p1[1]:.1f}", INK, 2.2, "arr")
        s.text(p1[0] - 14, p1[1] + 4, "RoPE rotates", 17, INK, anchor="end")
        # bound panel
        fx, fy, fw, fh = 644, 60, 224, 196
        s.rect(fx, fy, fw, fh, "#FFFFFF", ACC, 2.2, rx=6)
        rich(s, fx + fw / 2, fy + 36, [("max logit ≤ √d · g",), ("q", "sub"), (" · g",), ("k", "sub")], 20, color=ACC,
             weight="bold")
        s.line(fx + 16, fy + 58, fx + fw - 16, fy + 58, "#BFBFBF", 1.2)
        s.text(fx + fw / 2, fy + 90, "d = 128, unit gains", 18, INK)
        s.text(fx + fw / 2, fy + 128, "max logit ≤ 11.3", 23, ACC, weight="bold")
        s.text(fx + fw / 2, fy + 168, "g: max learned gain", 16, MUTED, style="italic")
    s.save(os.path.join(HERE, f"fig_sphere_{step}.svg"))


draw(1)
draw(2)
