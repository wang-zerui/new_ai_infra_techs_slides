"""fig_tiles: a row-parallel projection tiled so each tile's partial leaves as soon as it is computed."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 296, "fig_tiles: tiled matmul fused with its reduce-scatter, one thread block per tile pair")

# ---- left: the output tile grid
s.text(20, 28, "Output tiles", 19, INK, anchor="start", weight="bold")
TX, TY, TW, TH, TG = 20, 42, 52, 36, 8
for r in range(4):
    for c in range(4):
        fill = ACC_F if r * 4 + c < 6 else "#FFFFFF"
        s.rect(TX + c * (TW + TG), TY + r * (TH + TG), TW, TH, fill, ACC if r * 4 + c < 6 else NEU, 1.8, 2)
s.text(TX + 2 * (TW + TG) - 4, TY + 4 * (TH + TG) + 22, "computed so far", 17, ACC, style="italic")

# ---- arrow into the lanes
s.line(268, 120, 310, 120, INK, 2.2, "arr")
s.text(289, 104, "per tile", 17, MUTED)

# ---- right: two lanes
X0 = 420
s.text(406, 76, "compute", 18, INK, anchor="end")
s.text(406, 150, "comm", 18, INK, anchor="end")
TWD, TGP = 62, 6
for i in range(5):
    s.box(X0 + i * (TWD + TGP), 52, TWD, 36, f"T{i + 1}", NEU_F, NEU, size=17)
    s.rect(X0 + (i + 1) * (TWD + TGP), 126, TWD, 36, ACC_F, ACC, 2, 2)
    s.text(X0 + (i + 1) * (TWD + TGP) + TWD / 2, 150, f"T{i + 1}", 17, ACC, weight="bold")
    s.line(X0 + i * (TWD + TGP) + TWD, 88, X0 + (i + 1) * (TWD + TGP) + TWD / 2, 124, MUTED, 1.6, dash="4,3")

s.line(X0 - 8, 196, 862, 196, INK, 2, "arr")
s.text(858, 220, "time", 18, INK, anchor="end")
s.text(X0, 250, "one fused kernel · reduce-scatter rides the tiles", 18, ACC, anchor="start", weight="bold")
s.save(os.path.join(HERE, "gen_tiles_tmp.svg").replace("gen_tiles_tmp.svg", "fig_tiles.svg"))
