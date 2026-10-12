"""fig_ops: in a decoder layer only attention (with RoPE positions) needs sample boundaries; all other ops act per token."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 236, "fig_ops")
ops = [("embed", 92, 0), ("norm", 80, 0), ("QKV proj", 104, 0), ("attention", 150, 1), ("out proj", 100, 0),
       ("MLP / MoE", 112, 0), ("loss", 80, 0)]
G = 15
x = 14
Y, H = 60, 56
pos = []
for name, w, hot in ops:
    if hot:
        s.box(x, Y, w, H, name, ACC_F, ACC, ACC, 19, "bold", 2.4)
    else:
        s.box(x, Y, w, H, name, NEU_F, NEU, INK, 18, "bold")
    pos.append((x, w, hot))
    x += w + G
for (x0, w0, _), (x1, _, _) in zip(pos, pos[1:]):
    s.line(x0 + w0 + 1, Y + H / 2, x1 - 1, Y + H / 2, INK, 2, "arr")
# input tensor label
s.text(14, 34, "input: one packed stream [T, H]", 18, INK, anchor="start", weight="bold")
# attention annotation
ax, aw, _ = pos[3]
s.text(ax + aw / 2, Y + H + 30, "needs cu_seqlens", 17, ACC, weight="bold")
s.text(ax + aw / 2, Y + H + 52, "and position_ids", 17, ACC, weight="bold")
# bracket for per-token ops (left group and right group)
def bracket(xa, xb, y, label):
    s.path(f"M{xa:.1f},{y} L{xa:.1f},{y + 10} L{xb:.1f},{y + 10} L{xb:.1f},{y}", MUTED, 2)
    s.text((xa + xb) / 2, y + 34, label, 17, MUTED)
by = Y + H + 70
bracket(pos[0][0], pos[2][0] + pos[2][1], by, "per token")
bracket(pos[4][0], pos[5][0] + pos[5][1], by, "per token")
lx, lw, _ = pos[6]
s.text(lx + lw / 2, Y + H + 30, "per valid", 16, MUTED)
s.text(lx + lw / 2, Y + H + 50, "token", 16, MUTED)
s.save(os.path.join(HERE, "fig_ops.svg"))
