"""fig_ranks: the same 8 GPUs factored twice ([PF] App. 6.2 mapping: attention TP2 CP2 DP2, MoE ETP1 EP8 EDP1).

Attention rank generator: r = t + 2c + 4d (order tp-cp-ep-dp-pp, ep = 1).
Expert rank generator:    r = ep (ETP = 1, EDP = 1).
"""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 296, "fig_ranks: attention coordinates (d c t) vs expert coordinates (EP rank) on GPU0..7")
X0, W = 184, 74
def xi(i):
    return X0 + i * 80 + (i // 2) * 6 + (i // 4) * 14
Y1, Y2, H = 44, 150, 54

s.text(170, Y1 + 24, "attention", 18, INK, anchor="end", weight="bold")
s.text(170, Y1 + 46, "TP2 · CP2 · DP2", 16, MUTED, anchor="end")
s.text(170, Y2 + 24, "MoE", 18, ACC, anchor="end", weight="bold")
s.text(170, Y2 + 46, "ETP1 · EP8 · EDP1", 16, MUTED, anchor="end")
for i in range(8):
    x = xi(i)
    t, c, d = i % 2, (i // 2) % 2, i // 4
    s.text(x + W / 2, Y1 - 12, f"GPU{i}", 17, INK, weight="bold")
    s.box(x, Y1, W, H, f"d{d} c{c} t{t}", NEU_F, NEU, size=17)
    s.line(x + W / 2, Y1 + H + 4, x + W / 2, Y2 - 4, MUTED, 1.6, dash="4,4")
    s.box(x, Y2, W, H, f"E{i}", ACC_F, ACC, color=ACC, weight="bold", size=18)
# one EP group over all 8 GPUs
xa, xb = xi(0), xi(7) + W
yb = Y2 + H + 10
s.path(f"M{xa},{yb} v10 H{xb} v-10", ACC, 2.2)
s.text((xa + xb) / 2, yb + 34, "one EP8 group across all TP × CP × DP ranks", 17, ACC, weight="bold")
s.text(X0, 288, "rank order tp-cp-ep-dp-pp: TP fastest, PP outermost and shared", 16, MUTED, anchor="start", style="italic")
s.save(os.path.join(HERE, "fig_ranks.svg"))
