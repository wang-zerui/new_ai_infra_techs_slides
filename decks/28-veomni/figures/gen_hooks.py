"""fig_hooks: one MoE decoder layer; FSDP2, Ulysses SP and EP attach at three narrow boundaries."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 282, "fig_hooks: FSDP2 on parameters, SP around attention, EP around experts")

Y, H = 112, 52
seq = [("QKV proj", 104, "c"), ("a2a", 54, "a"), ("attention", 114, "c"), ("a2a", 54, "a"),
       ("out proj", 100, "c"), ("router", 88, "c"), ("a2a", 54, "a"), ("experts", 100, "c"), ("a2a", 54, "a")]
GAP = 13
x = 16
pos = []
for name, w, kind in seq:
    if kind == "c":
        s.box(x, Y, w, H, name, NEU_F, NEU, INK, 19, "bold")
    else:
        s.box(x, Y + 6, w, H - 12, name, ACC_F, ACC, ACC, 18, "bold", 2.2, 4)
    pos.append((x, w, name, kind))
    x += w + GAP
# connectors
for (x0, w0, _, k0), (x1, _, _, k1) in zip(pos, pos[1:]):
    s.line(x0 + w0, Y + H / 2, x1, Y + H / 2, INK, 2)

# FSDP band on top, arrows to every weight-holding module
s.rect(16, 22, 838, 40, ACC_F, ACC, 2, 5, dash="7,4")
s.text(435, 48, "FSDP2: all-gather weights / reduce-scatter grads", 19, ACC, weight="bold")
for x0, w, name, kind in pos:
    if kind == "c" and name != "attention":
        s.line(x0 + w / 2, 62, x0 + w / 2, Y - 2, ACC, 2.2, "arrAcc")

# brackets below for SP and EP
def bracket(xa, xb, label):
    yb = Y + H + 14
    s.path(f"M{xa:.1f},{yb} L{xa:.1f},{yb + 10} L{xb:.1f},{yb + 10} L{xb:.1f},{yb}", ACC, 2.2)
    s.text((xa + xb) / 2, yb + 38, label, 19, ACC, weight="bold")

sp_a, sp_b = pos[1][0], pos[3][0] + pos[3][1]
ep_a, ep_b = pos[6][0], pos[8][0] + pos[8][1]
bracket(sp_a, sp_b, "Ulysses SP: attention I/O")
bracket(ep_a, ep_b, "EP: expert weights")

# legend
s.rect(16, 252, 26, 20, NEU_F, NEU, 2, 3)
s.text(50, 268, "HF model code", 17, INK, anchor="start")
s.rect(200, 252, 26, 20, ACC_F, ACC, 2, 3)
s.text(234, 268, "attached from outside the model", 17, INK, anchor="start")
s.text(854, 268, "one MoE decoder layer (forward)", 17, MUTED, anchor="end", style="italic")
s.save(os.path.join(HERE, "fig_hooks.svg"))
