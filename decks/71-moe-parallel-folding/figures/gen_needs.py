"""fig_needs: what attention and MoE layers each want from TP / CP / EP ([MC] Table 2)."""
import os
from svgkit import SVG, ACC, PRB, NEU, NEU_F, INK, MUTED, GOOD

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(480, 266, "fig_needs: attention vs MoE parallelism requirements")
C1, C2, W, H = 64, 272, 200, 66
rows = [("TP", ("ok", "high TP shards", "large QKV"), ("bad", "high TP slices", "small experts")),
        ("CP", ("ok", "high CP splits", "long sequences"), ("na", "no sequence", "dependency")),
        ("EP", ("na", "no experts", "to spread"), ("ok", "high EP spreads", "many experts"))]
s.text(C1 + W / 2, 24, "Attention", 19, INK, weight="bold")
s.text(C2 + W / 2, 24, "MoE layer", 19, INK, weight="bold")
sym = {"ok": ("✓", GOOD), "bad": ("✗", PRB), "na": ("–", MUTED)}
for i, (name, a, m) in enumerate(rows):
    y = 40 + i * (H + 8)
    s.text(50, y + H / 2 + 7, name, 20, INK, anchor="end", weight="bold")
    for x, (kind, l1, l2) in ((C1, a), (C2, m)):
        stroke = PRB if kind == "bad" else NEU
        fill = "#FBE5E5" if kind == "bad" else ("#FFFFFF" if kind == "na" else NEU_F)
        s.rect(x, y, W, H, fill, stroke, 2, 4, dash="6,4" if kind == "na" else None)
        g, col = sym[kind]
        s.text(x + 22, y + H / 2 + 8, g, 24, col, weight="bold")
        tc = MUTED if kind == "na" else (PRB if kind == "bad" else INK)
        s.text(x + 44, y + 29, l1, 17, tc, anchor="start")
        s.text(x + 44, y + 51, l2, 17, tc, anchor="start")
s.save(os.path.join(HERE, "fig_needs.svg"))
