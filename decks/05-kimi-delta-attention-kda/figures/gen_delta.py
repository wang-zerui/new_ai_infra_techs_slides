"""fig_delta_{1,2,3}.svg: one KDA token step as decay -> predict -> correct (K3 Eq. 1 rewritten:
S' = Diag(alpha_t) S_{t-1};  e_t = v_t - S'^T k_t;  S_t = S' + beta_t k_t e_t^T). Toy 4x4 head, shades only."""
import pathlib
ACC, INK, GRAY = "#1F4E79", "#222222", "#595959"
W, H = 870, 330
def shade(a):  # blend white->#5B9BD5 by a in [0,1]
    r0, g0, b0 = 255, 255, 255; r1, g1, b1 = 0x5B, 0x9B, 0xD5
    return "#%02X%02X%02X" % (round(r0 + (r1 - r0) * a), round(g0 + (g1 - g0) * a), round(b0 + (b1 - b0) * a))
alpha = [0.95, 0.25, 1.0, 0.6]
base = [[0.7, 0.5, 0.6, 0.8], [0.6, 0.8, 0.5, 0.7], [0.5, 0.6, 0.8, 0.6], [0.8, 0.7, 0.6, 0.5]]
kv = [0.9, 0.2, 0.5, 0.1]; ev = [0.8, 0.3, 0.6, 0.45]
def build(step):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker></defs>']
    def t(x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
        o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
    def cell(x, y, w, h, f, s=ACC, sw=1.2):
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" fill="{f}" stroke="{s}" stroke-width="{sw}"/>')
    def arrow(x1, y1, x2, y2, sw=2.2):
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="{sw}" marker-end="url(#a)"/>')
    sb = lambda b, s: f'{b}<tspan font-size="14" dy="5">{s}</tspan><tspan dy="-5"></tspan>'
    # ---------------- region A: decay
    t(174, 30, "Decay", 22, "bold", ACC)
    c = 28
    for i in range(4):
        for j in range(4):
            cell(20 + j * c, 84 + i * c, c, c, shade(base[i][j]))
    t(76, 72, sb("S", "t−1"), 19, "bold")
    arrow(134, 140, 158, 140)
    for i in range(4):
        cell(160, 84 + i * c, 26, c, shade(alpha[i]), GRAY)
    t(173, 72, sb("α", "t"), 19, "bold", ACC)
    arrow(188, 140, 212, 140)
    for i in range(4):
        for j in range(4):
            cell(214 + j * c, 84 + i * c, c, c, shade(base[i][j] * alpha[i]))
    t(270, 72, "S′", 19, "bold")
    t(174, 238, "S′ = Diag(α<tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\">) S</tspan><tspan font-size=\"14\" dy=\"5\">t−1</tspan>", 19)
    t(174, 276, "forget per key channel", 18, "normal", GRAY, "middle", "italic")
    if step >= 2:
        o.append('<line x1="346" y1="20" x2="346" y2="290" stroke="#D9D9D9" stroke-width="1.2"/>')
        t(470, 30, "Predict", 22, "bold", ACC)
        rows = [("v", "t", [0.85, 0.35, 0.75, 0.55], "#FFFFFF"), ("S′ᵀk", "t", [0.05, 0.05, 0.15, 0.1], None), ("e", "t", ev, None)]
        ys = [92, 140, 188]
        vals_pred = [0.05, 0.05, 0.15, 0.1]
        for (b, s, vals, _), y in zip(rows, ys):
            t(442, y + 19, sb(b, s), 19, "bold" if b == "e" else "normal", ACC if b == "e" else INK, "end")
            for j in range(4):
                f = shade(vals[j]) if b != "e" else shade(vals[j])
                cell(452 + j * 26, y, 26, 26, f, ACC if b == "e" else GRAY, 2 if b == "e" else 1.2)
        t(474, 238, "e<tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\"> = v</tspan><tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\"> − S′ᵀ k</tspan><tspan font-size=\"14\" dy=\"5\">t</tspan>", 19)
        t(474, 276, "what memory gets wrong", 18, "normal", GRAY, "middle", "italic")
    if step >= 3:
        o.append('<line x1="584" y1="20" x2="584" y2="290" stroke="#D9D9D9" stroke-width="1.2"/>')
        t(728, 30, "Correct", 22, "bold", ACC)
        c2 = 24
        for i in range(4):
            for j in range(4):
                cell(604 + j * c2, 92 + i * c2, c2, c2, shade(kv[i] * ev[j]), ACC, 1.2)
        t(652, 76, "β k eᵀ", 19, "bold", ACC)
        t(716, 148, "+", 26, "bold")
        t(716, 176, "S′", 18)
        arrow(732, 140, 752, 140)
        for i in range(4):
            for j in range(4):
                cell(754 + j * 26, 88 + i * 26, 26, 26, shade(min(1, base[i][j] * alpha[i] + 0.6 * kv[i] * ev[j])))
        t(806, 76, sb("S", "t"), 19, "bold")
        t(728, 238, "S<tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\"> = S′ + β</tspan><tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\"> k</tspan><tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\"> e</tspan><tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-12\" font-size=\"14\">T</tspan>", 19)
        t(728, 276, "rank-1 write of the error", 18, "normal", GRAY, "middle", "italic")
        t(435, 318, "output  o<tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\"> = S</tspan><tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-12\" font-size=\"14\">T</tspan><tspan dy=\"7\"> q</tspan><tspan font-size=\"14\" dy=\"5\">t</tspan><tspan dy=\"-5\">   ·   state stays d</tspan><tspan font-size=\"14\" dy=\"5\">k</tspan><tspan dy=\"-5\"> × d</tspan><tspan font-size=\"14\" dy=\"5\">v</tspan>", 18, "bold", ACC)
    o.append('</svg>')
    return "\n".join(o)
for k in (1, 2, 3):
    p = pathlib.Path(__file__).with_name(f"fig_delta_{k}.svg"); p.write_text(build(k)); print("wrote", p)
