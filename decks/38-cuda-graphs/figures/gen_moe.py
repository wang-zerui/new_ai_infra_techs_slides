"""fig_moe: partial CUDA graph of one dropless MoE layer (forward), --cuda-graph-modules attn moe_router
moe_preprocess. Static parts (attention, router, shared expert if not EP-overlapped, EP preprocessing) form one
graph; dispatch, routed experts and combine run eagerly because token counts per expert are dynamic.
Sources: arXiv 2603.07685 Sec. 4.3.6 p.40-41 (Fig. 24-25); cuda_graph.html --cuda-graph-modules table."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(870, 214, "fig_moe: partial graph scopes in a MoE layer")

Y, H = 58, 54
boxes = [("attention", 8, 110, "g"), ("router", 136, 100, "g"), (["EP", "preprocess"], 254, 108, "g"),
         ("dispatch", 410, 96, "d"), (["routed", "experts"], 524, 106, "d"), ("combine", 648, 96, "d")]
# containers first
s.rect(2, 30, 372, 176, "#FFFFFF", ACC, 2.8, 6)
s.text(8, 22, "one CUDA graph (attn + moe_router + moe_preprocess)", 17, ACC, anchor="start", weight="bold")
s.rect(396, 30, 362, 92, "#FFFFFF", PRB, 2.2, 6, dash="7,4")
s.text(577, 146, "eager: tokens per expert", 17, PRB, weight="bold")
s.text(577, 168, "known only at run time", 17, PRB)
for lab, x, w, kind in boxes:
    if kind == "g":
        s.box(x, Y, w, H, lab, ACC_F, ACC, size=17, sw=2)
    else:
        s.box(x, Y, w, H, lab, PRB_F, PRB, color=INK, size=17, sw=2)
# shared expert inside the graph, fed by the attention output
s.box(136, 136, 100, 50, ["shared", "expert"], ACC_F, ACC, size=17, sw=2)
s.path(f"M{8 + 110 / 2},{Y + H + 2} L{8 + 110 / 2},161 L134,161", ACC, 2, "arrAcc")
# main flow arrows
xs = [(8, 110), (136, 100), (254, 108), (410, 96), (524, 106), (648, 96)]
for (x1, w1), (x2, _) in zip(xs, xs[1:]):
    s.line(x1 + w1 + 2, Y + H / 2, x2 - 2, Y + H / 2, INK, 2.2, "arr")
s.box(782, Y, 84, H, ["layer", "ℓ+1"], "#FFFFFF", MUTED, color=MUTED, size=17, dash="5,3")
s.line(746, Y + H / 2, 780, Y + H / 2, INK, 2.2, "arr")
s.save(os.path.join(HERE, "fig_moe.svg"))
