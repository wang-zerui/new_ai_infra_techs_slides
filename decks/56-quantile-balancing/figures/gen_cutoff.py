"""fig_cutoff: one token's biased scores (toy t3, k = 1, b(t) = 0). Top-(k+1) gives the route and the cutoff;
margins s_ij - alpha_i say how far each expert is from entering this token's top-k."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
s = SVG(490, 290, "fig_cutoff: token t3 scores, cutoff and margins")

rows = [("E1", 0.60, "+.10"), ("E3", 0.50, "0"), ("E2", 0.30, "−.20"), ("E4", 0.25, "−.25")]
X0, SC, H = 58, 300, 38
Y = [64, 116, 168, 220]
alpha = 0.50
xa = X0 + alpha * SC

s.text(4, 26, "token t3, k = 1", 18, INK, anchor="start", weight="bold")
s.text(486, 26, "margin", 18, INK, anchor="end", weight="bold")
for (e, v, mg), y in zip(rows, Y):
    s.text(X0 - 10, y + H / 2 + 6.5, e, 19, INK, anchor="end", weight="bold")
    w = v * SC
    if e == "E1":
        s.rect(X0, y, w, H, ACC_F, ACC, 2, 3)
        s.text(X0 + w + 8, y + H / 2 + 6.5, ".60 routed", 18, ACC, anchor="start", weight="bold")
    elif e == "E3":
        s.rect(X0, y, w, H, PRB_F, PRB, 2, 3)
        s.text(X0 + w + 8, y + H / 2 + 6.5, ".50 (k+1)-th", 18, PRB, anchor="start", weight="bold")
    else:
        s.rect(X0, y, w, H, NEU_F, NEU, 2, 3)
        s.text(X0 + w + 8, y + H / 2 + 6.5, f"{v:.2f}".lstrip("0"), 18, INK, anchor="start")
    col = ACC if mg.startswith("+") else (PRB if mg == "0" else INK)
    s.text(486, y + H / 2 + 6.5, mg, 19, col, anchor="end", weight="bold")

# cutoff line
s.line(xa, 48, xa, Y[-1] + H + 2, PRB, 2.2, dash="6,4")
s.rich(xa, 284, [("cutoff ", PRB, "bold"), ("α", PRB, "bold", "italic"), ("3", PRB, "bold", "sub"),
                 (" = .50", PRB, "bold")], 18)
s.save(os.path.join(HERE, "fig_cutoff.svg"))
