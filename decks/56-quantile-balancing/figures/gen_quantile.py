"""fig_quantile_1/2: per-expert margins (toy, m = 8, n = 4, k = 1, q = 2, b(t) = 0), top 4 shown per expert.
Step 1: sorted margins and current loads. Step 2: the (q+1)-th margin is the threshold; bias b_hat = -theta;
the two margins above it are the tokens the expert keeps or gains; loads become 2/2/2/2."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))

# (expert, load before, top-4 margins as (token, margin text, moved-from), theta, b_hat)
COLS = [
    ("E1", 4, [("t1", "+.40", None), ("t2", "+.30", None), ("t3", "+.10", None), ("t4", "+.05", None)],
     "+.10", "−.10"),
    ("E2", 3, [("t5", "+.35", None), ("t6", "+.25", None), ("t7", "+.08", None), ("t1", "0", None)],
     "+.08", "−.08"),
    ("E3", 1, [("t8", "+.30", None), ("t3", "0", "E1"), ("t6", "−.10", None), ("t5", "−.15", None)],
     "−.10", "+.10"),
    ("E4", 0, [("t4", "0", "E1"), ("t7", "0", "E2"), ("t8", "−.15", None), ("t3", "−.25", None)],
     "−.15", "+.15"),
]
q = 2
CW, GAP, X0 = 200, 15, 10
CH, CG, Y0 = 36, 6, 50


def draw(step):
    s = SVG(870, 292, f"fig_quantile_{step}: per-expert margin quantile (toy example)")
    for c, (e, load, chips, theta, bh) in enumerate(COLS):
        x = X0 + c * (CW + GAP)
        if step == 1:
            s.text(x + CW / 2, 30, f"{e} · load {load}", 20, INK, weight="bold")
        else:
            col = PRB if load != q else INK
            s.rich(x + CW / 2, 30, [(f"{e} · load ", INK, "bold"), (f"{load}", col, "bold"),
                                    (" → ", INK, "bold"), (f"{q}", ACC, "bold")], 20)
        for r, (tok, mg, moved) in enumerate(chips):
            y = Y0 + r * (CH + CG)
            if step == 2 and r < q:
                fill, stroke, tcol, sw = ACC_F, ACC, ACC, 2.4
            elif step == 2 and r == q:
                fill, stroke, tcol, sw = "#FFFFFF", PRB, PRB, 2.4
            else:
                fill, stroke, tcol, sw = "#FFFFFF", "#A6A6A6", INK, 1.6
            s.rect(x, y, CW, CH, fill, stroke, sw, 4)
            s.text(x + 12, y + CH / 2 + 6.5, tok, 19, tcol, anchor="start", weight="bold")
            s.text(x + CW - 12, y + CH / 2 + 6.5, mg, 19, tcol, anchor="end", weight="bold")
            if moved and step == 2:
                s.text(x + CW / 2 - 4, y + CH / 2 + 6, f"from {moved}", 16, PRB, style="italic")
        yd = Y0 + 4 * (CH + CG) + 14
        s.text(x + CW / 2, yd, "⋯ 4 more", 17, MUTED, style="italic")
        if step == 2:
            yl = Y0 + q * (CH + CG) - CG / 2
            s.line(x - 4, yl, x + CW + 4, yl, PRB, 2.4, dash="7,4")
            s.rich(x + CW / 2, 272, [("b̂", ACC, "bold", "italic"), (" = −θ = ", ACC, "bold"),
                                     (bh, ACC, "bold")], 20)
    s.save(os.path.join(HERE, f"fig_quantile_{step}.svg"))


draw(1)
draw(2)
