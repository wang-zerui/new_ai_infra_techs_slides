"""fig_overview.svg: one KDA layer as used in Kimi K3 (K3 report Eq. 1, 2, 5, 6; Fig. 2), slide scale.
Badges: 1 gated delta rule (state update), 2 chunkwise form, 3 3:1 hybrid with Gated MLA."""
import pathlib
ACC, ACC_F, GRAY, GRAY_F, INK, RED = "#1F4E79", "#DEEBF7", "#595959", "#F2F2F2", "#222222", "#C00000"
W, H = 870, 300
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker></defs>']
def t(x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
def box(x, y, w, h, fill="#FFFFFF", stroke=GRAY, sw=1.5, rx=5):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def arrow(x1, y1, x2, y2, sw=1.8, extra=""):
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="{sw}" marker-end="url(#a)" {extra}/>')
def path(d, sw=1.8):
    o.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{sw}" marker-end="url(#a)"/>')
def badge(cx, cy, n):
    o.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{INK}"/>')
    t(cx, cy + 6, str(n), 17, "bold", "#FFFFFF")
sub = lambda base, s: f'{base}<tspan font-size="13" dy="4">{s}</tspan><tspan dy="-4"></tspan>'
# input
box(8, 131, 50, 38, "#FFFFFF", INK)
t(33, 157, sub("x", "t"), 19)
# bus
rows = [(50, "q", "ShortConv + L2Norm", False), (100, "k", "ShortConv + L2Norm", False), (150, "v", "ShortConv", False),
        (200, "β", "Sigmoid", True), (250, "α", "Bounded decay", True)]
o.append(f'<line x1="58" y1="150" x2="72" y2="150" stroke="{INK}" stroke-width="1.8"/>')
o.append(f'<line x1="72" y1="50" x2="72" y2="250" stroke="{INK}" stroke-width="1.8"/>')
for y, name, label, acc in rows:
    arrow(72, y, 90, y)
    box(92, y - 19, 220, 38, ACC_F if acc else GRAY_F, ACC if acc else GRAY, 2 if acc else 1.5)
    t(202, y + 6, label, 18, "bold" if acc else "normal", ACC if acc else INK)
    t(334, y + 6, sub(name, "t"), 19, "bold" if acc else "normal", ACC if acc else INK)
    arrow(352, y, 398, y)
t(202, 290, "new in K3", 15, "normal", ACC, "middle", "italic")
# state box
box(400, 36, 192, 228, ACC_F, ACC, 2.5, 6)
t(496, 64, sub("State S", "t"), 20, "bold", ACC)
t(496, 88, "d<tspan font-size=\"13\" dy=\"4\">k</tspan><tspan dy=\"-4\"> × d</tspan><tspan font-size=\"13\" dy=\"4\">v</tspan><tspan dy=\"-4\"> per head</tspan>", 17)
c = 20
for i in range(4):
    for j in range(5):
        o.append(f'<rect x="{446+j*c}" y="{100+i*c}" width="{c}" height="{c}" fill="#FFFFFF" stroke="{ACC}" stroke-width="1.2"/>')
badge(424, 207, 1); t(442, 213, "gated delta rule", 17, "normal", INK, "start")
badge(424, 241, 2); t(442, 247, "chunkwise form", 17, "normal", INK, "start")
# output path
arrow(592, 150, 610, 150)
box(612, 131, 100, 38, GRAY_F, GRAY)
t(662, 157, "RMSNorm", 17)
arrow(712, 150, 767, 150)
o.append(f'<circle cx="785" cy="150" r="16" fill="#FFFFFF" stroke="{INK}" stroke-width="1.8"/>')
t(785, 157, "×", 22, "bold")
box(712, 52, 146, 38, ACC_F, ACC, 2)
t(785, 77, "Full-rank gate", 18, "bold", ACC)
path("M33,131 V18 H785 V50")
arrow(785, 90, 785, 132)
arrow(785, 166, 785, 196)
box(722, 198, 126, 38, GRAY_F, GRAY)
t(785, 223, "Out proj", 18)
arrow(785, 236, 785, 262)
t(785, 285, sub("y", "t"), 19)
t(795, 115, "new in K3", 15, "normal", ACC, "start", "italic")
# hybrid strip (3)
for j in range(4):
    f, s = (ACC, ACC) if j < 3 else ("#BFBFBF", "#BFBFBF")
    o.append(f'<rect x="{612+j*22}" y="{230}" width="17" height="30" fill="{f}" stroke="{s}"/>')
badge(624, 208, 3)
t(652, 285, "3:1 hybrid", 17)
o.append('</svg>')
p = pathlib.Path(__file__).with_name("fig_overview.svg"); p.write_text("\n".join(o)); print("wrote", p)
