"""fig_statusquo.svg: three ways to keep context, and how each fails (slide scale, 1 unit = 1pt).
(a) softmax attention: KV cache grows; (b) linear attention: additive writes into a fixed state never forget;
(c) head-wise scalar gate (Mamba2/GDN): one decay rate for every feature of the head."""
import pathlib
ACC, ACC_F, GRAY, GRAY_F, INK, RED, RED_F = "#1F4E79", "#DEEBF7", "#595959", "#F2F2F2", "#222222", "#C00000", "#FBE5E5"
W, H = 870, 285
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker>'
     '<marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#C00000"/></marker>'
     '</defs>']
def t(x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
def grid(x, y, n, c, fills, stroke=GRAY):
    for i in range(n):
        for j in range(n):
            o.append(f'<rect x="{x+j*c:.1f}" y="{y+i*c:.1f}" width="{c}" height="{c}" fill="{fills[i][j]}" stroke="{stroke}" stroke-width="1.2"/>')
# separators
for sx in (290, 580):
    o.append(f'<line x1="{sx}" y1="10" x2="{sx}" y2="270" stroke="#BFBFBF" stroke-width="1.2"/>')
# ---------- (a) softmax attention: growing KV cache
t(145, 30, "Softmax attention", 20, "bold")
# query box
qx, qy = 205, 60
o.append(f'<rect x="{qx}" y="{qy}" width="56" height="36" rx="4" fill="#FFFFFF" stroke="{INK}" stroke-width="1.5"/>')
t(qx + 28, qy + 25, "q<tspan font-size=\"14\" dy=\"4\">t</tspan>", 19)
cells = [18 + i * 26 for i in range(9)]
cy = 150
for i, cxx in enumerate(cells):
    o.append(f'<rect x="{cxx}" y="{cy}" width="22" height="40" fill="{RED_F}" stroke="{RED}" stroke-width="1.3"/>')
    o.append(f'<line x1="{qx+28}" y1="{qy+37}" x2="{cxx+11}" y2="{cy-1}" stroke="#7F7F7F" stroke-width="1"/>')
t(145, 222, "KV cache: one entry per token", 17)
o.append(f'<line x1="22" y1="240" x2="262" y2="240" stroke="{RED}" stroke-width="2.5" marker-end="url(#r)"/>')
t(142, 264, "grows with every token", 18, "bold", RED)
# ---------- (b) linear attention: additive writes
t(435, 30, "Linear attention", 20, "bold")
gx, gy, c = 455, 72, 26
fills = [[RED_F] * 4 for _ in range(4)]
grid(gx, gy, 4, c, fills, RED)
t(gx + 2 * c, gy - 10, "state S", 18, "bold")
for k, yy in enumerate((80, 120, 160)):
    o.append(f'<rect x="305" y="{yy-14}" width="78" height="30" rx="4" fill="#FFFFFF" stroke="{INK}" stroke-width="1.3"/>')
    t(344, yy + 7, f"+ k<tspan font-size=\"13\" dy=\"4\">{k+1}</tspan><tspan dy=\"-4\">v</tspan><tspan font-size=\"13\" dy=\"4\">{k+1}</tspan><tspan dy=\"-12\" font-size=\"13\">T</tspan>", 18)
    o.append(f'<line x1="385" y1="{yy+1}" x2="{gx-3}" y2="{gy+2*c+(k-1)*20:.1f}" stroke="{INK}" stroke-width="1.6" marker-end="url(#a)"/>')
t(435, 222, "fixed d<tspan font-size=\"13\" dy=\"4\">k</tspan><tspan dy=\"-4\"> × d</tspan><tspan font-size=\"13\" dy=\"4\">v</tspan><tspan dy=\"-4\"> matrix</tspan>", 17)
t(435, 264, "writes pile up without forgetting", 18, "bold", RED)
# ---------- (c) head-wise scalar gate
t(725, 30, "Head-wise gate", 20, "bold")
gx2 = 742
shade = "#D9D9D9"
grid(gx2, gy, 4, c, [[shade] * 4 for _ in range(4)], GRAY)
t(gx2 + 2 * c, gy - 10, "state S", 18, "bold")
o.append(f'<circle cx="672" cy="{gy+2*c}" r="26" fill="#FFFFFF" stroke="{RED}" stroke-width="2"/>')
t(672, gy + 2 * c + 7, "× α<tspan font-size=\"13\" dy=\"4\">t</tspan>", 19, "bold", RED)
o.append(f'<line x1="699" y1="{gy+2*c}" x2="{gx2-3}" y2="{gy+2*c}" stroke="{RED}" stroke-width="2" marker-end="url(#r)"/>')
t(725, 222, "Mamba2 / GDN: one α per head", 17)
t(725, 264, "same decay for all features", 18, "bold", RED)
o.append('</svg>')
p = pathlib.Path(__file__).with_name("fig_statusquo.svg"); p.write_text("\n".join(o)); print("wrote", p)
