"""fig_signstep: (a) the loss-free balancing loop, (b) schematic bias trajectories for small vs large step γ."""
import os
from svgkit import SVG, ACC, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 300, "fig_signstep: sign-update loop and its step-size dilemma (schematic, no data)")

# ---------------- panel (a): the loop
X, W, H = 20, 350, 46
ys = [14, 88, 162]
s.rect(X, ys[0], W, H, NEU_F, NEU, 2, 5)
s.rich(X + W / 2, ys[0] + H / 2 + 6.5, [("score ", INK, "normal"), ("s", INK, "bold", "italic"),
                                       ("ij", INK, "normal", "sub"), (" + bias ", INK, "normal"),
                                       ("b", INK, "bold", "italic"), ("j", INK, "normal", "sub")], 20)
s.box(X, ys[1], W, H, "Top-k routes tokens", NEU_F, NEU, size=19)
s.rect(X, ys[2], W, H, NEU_F, NEU, 2, 5)
s.rich(X + W / 2, ys[2] + H / 2 + 6.5, [("count load ", INK, "normal"), ("ℓ", INK, "normal", "italic"),
                                       ("j", INK, "normal", "sub"), (" per expert", INK, "normal")], 19)
# update rule box (the status quo, with its two problems in red)
Y4 = 236
s.rect(X, Y4, W, 54, PRB_F, PRB, 2, 5)
s.rich(X + W / 2, Y4 + 34, [("b", INK, "bold", "italic"), ("j", INK, "normal", "sub"), (" ← ", INK, "normal"),
                            ("b", INK, "normal", "italic"), ("j", INK, "normal", "sub"), (" + ", INK, "normal"),
                            ("γ", PRB, "bold"), (" · ", INK, "normal"), ("sign", PRB, "bold"),
                            ("(", INK, "normal"), ("q", INK, "normal", "italic"), (" − ", INK, "normal"), ("ℓ", INK, "normal", "italic"), ("j", INK, "normal", "sub"),
                            (")", INK, "normal")], 22)
for a, b in ((ys[0] + H, ys[1]), (ys[1] + H, ys[2]), (ys[2] + H, Y4)):
    s.line(X + W / 2, a + 2, X + W / 2, b - 1, INK, 2.2, "arr")
# return path: next step
s.path(f"M {X + W + 2} {Y4 + 27} H {X + W + 30} V {ys[0] + H / 2} H {X + W + 3}", INK, 2.2, "arr")
s.add(f'<text x="{X + W + 47}" y="{(ys[0] + Y4) / 2 + 20}" text-anchor="middle" font-size="18" fill="{MUTED}" '
      f'font-style="italic" transform="rotate(-90 {X + W + 47} {(ys[0] + Y4) / 2 + 20})">next step</text>')

# ---------------- panel (b): schematic trajectories
OX, OY, XE, YT = 478, 262, 858, 18
s.line(OX, OY, XE, OY, INK, 2, "arr")
s.line(OX, OY, OX, YT, INK, 2, "arr")
s.text(XE - 4, OY + 26, "training step", 18, INK, anchor="end")
s.add(f'<text x="{OX - 14}" y="{(OY + YT) / 2}" text-anchor="middle" font-size="18" fill="{INK}" '
      f'transform="rotate(-90 {OX - 14} {(OY + YT) / 2})">bias of one expert</text>')
TGT = 133
s.line(OX + 2, TGT, XE - 10, TGT, ACC, 2, dash="7,5")
s.text(XE - 8, TGT - 34, "balancing", 18, ACC, anchor="end", weight="bold")
s.text(XE - 8, TGT - 12, "bias", 18, ACC, anchor="end", weight="bold")

# small γ: slow staircase (gray)
pts, x, y = [], OX + 6, 238
while x < XE - 100:
    pts.append((x, y)); x += 26; y -= 7.5
d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
s.path(d, MUTED, 2.6)
for px, py in pts:
    s.circle(px, py, 3.2, MUTED, MUTED, 1)
s.text(pts[5][0] + 14, pts[5][1] + 34, "small γ: slow", 19, MUTED, anchor="start", weight="bold")

# large γ: oscillates around the target (red)
pts, x, y, up = [], OX + 6, 238, True
seq = [238, 168, 98, 168, 98, 168, 98, 168, 98, 168, 98, 168, 98, 168, 98]
for i, yy in enumerate(seq):
    xx = OX + 6 + i * 26
    if xx > XE - 100:
        break
    pts.append((xx, yy))
d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
s.path(d, PRB, 2.6)
for px, py in pts:
    s.circle(px, py, 3.4, PRB, PRB, 1)
s.text(pts[2][0] + 6, 70, "large γ: oscillates", 19, PRB, anchor="start", weight="bold")
s.text(OX + 10, YT + 14, "schematic", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_signstep.svg"))
