"""fig_layer_1 (HC) and fig_layer_2 (mHC): one layer with an n = 4 lane residual stream.
x_{l+1} = H_res x_l + H_post^T F(H_pre x_l). mHC: H_pre = sigma, H_post = 2 sigma, H_res = Sinkhorn-Knopp
(doubly stochastic). [mHC] eq. 3, 8; [V4] eq. 1, 6-8."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, sup_text

HERE = os.path.dirname(os.path.abspath(__file__))

def draw(mhc, name):
    W, H = 870, 292
    s = SVG(W, H, name)
    lanes = [52 + i * 34 for i in range(4)]
    X0, X1 = 80, 792
    fill, stroke, col = (ACC_F, ACC, ACC) if mhc else (NEU_F, NEU, INK)
    # lanes
    for y in lanes:
        s.line(X0, y, X1, y, MUTED, 2.6, "arrGray")
    sup_text(s, 8, lanes[1] + 4, "x", "l", 22, INK, anchor="start", sub=True)
    s.text(8, lanes[2] + 10, "4 × C", 17, MUTED, anchor="start")
    sup_text(s, 800, lanes[1] + 4, "x", "l+1", 22, INK, anchor="start", sub=True)
    # H_pre tap bus
    BX = 136
    s.line(BX, lanes[0], BX, 198, INK, 2.2, "arr")
    for y in lanes:
        s.add(f'<circle cx="{BX}" cy="{y}" r="5" fill="{INK}"/>')
    s.box(96, 200, 80, 42, "", fill, stroke, color=col, weight="bold", sw=2.2)
    sup_text(s, 136, 228, "H", "pre", 21, col)
    s.line(178, 221, 214, 221, INK, 2.2, "arr")
    s.box(216, 200, 360, 42, "Layer F  (attention or MoE)", "#FFFFFF", NEU)
    s.line(578, 221, 616, 221, INK, 2.2, "arr")
    s.box(618, 200, 84, 42, "", fill, stroke, color=col, weight="bold", sw=2.2)
    sup_text(s, 660, 228, "H", "post", 21, col)
    # H_post write-back bus with adders
    PX = 660
    s.line(PX, 198, PX, lanes[0] + 10, INK, 2.2)
    for y in lanes:
        s.add(f'<circle cx="{PX}" cy="{y}" r="9" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
        s.line(PX - 5, y, PX + 5, y, INK, 1.8); s.line(PX, y - 5, PX, y + 5, INK, 1.8)
    # H_res box on the lanes
    s.rect(330, lanes[0] - 18, 130, lanes[-1] - lanes[0] + 36, fill, stroke, 2.4, 5)
    sup_text(s, 395, lanes[1] + 10, "H", "res", 23, col)
    s.text(395, lanes[2] + 14, "4 × 4", 18, col)
    if mhc:
        s.text(395, 22, "doubly stochastic (Sinkhorn)", 18, ACC, weight="bold")
        s.text(136, 266, "σ ≥ 0", 18, ACC, weight="bold")
        s.text(660, 266, "2σ ≥ 0", 18, ACC, weight="bold")
    else:
        s.text(395, 22, "unconstrained 4 × 4 mixing", 18, PRB, weight="bold")
    s.save(os.path.join(HERE, name + ".svg"))

draw(False, "fig_layer_1")
draw(True, "fig_layer_2")
