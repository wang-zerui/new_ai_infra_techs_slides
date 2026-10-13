"""fig_ffa: (a) FFA runs a list of AttnSlices in parallel: each slice is tiled, tiles are spread over SMs,
and slices that share query rows merge partial O/LSE atomically ([BLOG] Fig. 5: slice-level parallelism with
atomic reduction). Example: a causal mask after redistribution = 3 slices. (b) the 4 mask types ([BLOG] Fig. 6)."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 262, "fig_ffa: slice-level parallelism + 4 mask types")


def badge(x, y, n, r=11):
    s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{ACC}"/>')
    s.text(x, y + 5.5, str(n), 15, "#FFFFFF", weight="bold")


# ---------------- (a) grid with three slices
s.text(14, 22, "(a) Slices run in parallel", 19, INK, anchor="start", weight="bold")
C = 25
X0, Y0 = 44, 50
s.text(X0 + 4 * C, Y0 - 8, "K", 16, MUTED, style="italic")
s.text(X0 - 14, Y0 + 4 * C + 5, "Q", 16, MUTED, style="italic")
for r in range(8):
    for c in range(8):
        s.rect(X0 + c * C, Y0 + r * C, C, C, "#FFFFFF", "#D9D9D9", 1, 0)
H = 4 * C
# slice 1: Q0-4, K0-4 causal
s.add(f'<path d="M{X0},{Y0} L{X0},{Y0 + H} L{X0 + H},{Y0 + H} Z" fill="{ACC_F}" stroke="{ACC}" stroke-width="2.2"/>')
# slice 2: Q4-8, K0-4 full
s.rect(X0, Y0 + H, H, H, ACC_F, ACC, 2.2, 0)
# slice 3: Q4-8, K4-8 causal
s.add(f'<path d="M{X0 + H},{Y0 + H} L{X0 + H},{Y0 + 2 * H} L{X0 + 2 * H},{Y0 + 2 * H} Z" fill="{ACC_F}" '
      f'stroke="{ACC}" stroke-width="2.2"/>')
badge(X0 + 30, Y0 + 70, 1)
badge(X0 + 50, Y0 + H + 50, 2)
badge(X0 + H + 30, Y0 + H + 70, 3)

# tiles on SMs
TX = 300
s.text(TX + 70, 42, "tiles on SMs", 17, MUTED, weight="bold")
rows = [(1, 2, 70), (2, 4, 132), (3, 2, 194)]
for n, k, y in rows:
    badge(TX - 16, y + 15, n)
    for t in range(k):
        s.rect(TX + 4 + t * 30, y, 25, 30, ACC_F, ACC, 1.6, 2)
# outputs
OX = 492
s.box(OX, 66, 146, 40, "O, LSE rows 0-3", "#FFFFFF", NEU, INK, 17)
s.box(OX, 160, 146, 40, "O, LSE rows 4-7", "#FFFFFF", NEU, INK, 17)
s.line(TX + 68, 85, OX - 3, 86, INK, 2.2, "arr")
s.path(f"M{TX + 128},147 C{OX - 30},147 {OX - 34},178 {OX - 3},178", INK, 2.2, "arr")
s.path(f"M{TX + 68},209 C{OX - 60},209 {OX - 50},184 {OX - 3},183", INK, 2.2, "arr")
s.text(OX + 73, 226, "atomic merge", 17, ACC, weight="bold")

# ---------------- (b) mask types
s.line(654, 14, 654, 252, "#BFBFBF", 1.2)
s.text(668, 22, "(b) Mask types", 19, INK, anchor="start", weight="bold")
m = 9   # mini cell
types = [("FULL", lambda r, c: True), ("CAUSAL", lambda r, c: c <= r),
         ("INV-CAUSAL", lambda r, c: c >= r), ("BI-CAUSAL", lambda r, c: c == r)]
for i, (name, f) in enumerate(types):
    gx = 690 + (i % 2) * 106
    gy = 46 + (i // 2) * 104
    for r in range(5):
        for c in range(5):
            s.rect(gx + c * m, gy + r * m, m, m, ACC if f(r, c) else "#FFFFFF", "#BFBFBF", 0.6, 0)
    s.text(gx + 22, gy + 66, name, 15, INK, weight="bold")
s.save(os.path.join(HERE, "fig_ffa.svg"))
