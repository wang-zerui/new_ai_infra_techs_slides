"""fig_chunk.svg: chunkwise-parallel KDA (K3 Eq. 3-4; Kimi Linear Eq. 8-9). Chunks of C tokens are processed
with dense matmuls inside; only the d_k x d_v state S[t] is handed from chunk to chunk."""
import pathlib
ACC, ACC_F, INK, GRAY, GRAY_F, RED = "#1F4E79", "#DEEBF7", "#222222", "#595959", "#F2F2F2", "#C00000"
W, H = 870, 290
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1F4E79"/></marker>'
     '<marker id="k" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker></defs>']
def t(x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
def rect(x, y, w, h, f, s, sw=1.5, rx=0, extra=""):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}" {extra}/>')
sb = lambda b, s: f'{b}<tspan font-size="14" dy="5">{s}</tspan><tspan dy="-5"></tspan>'
# ---- row 1: chunk chain
cw, sw_, gap = 150, 44, 12
xs = [16, 16 + cw + sw_ + 2 * gap, 16 + 2 * (cw + sw_ + 2 * gap), 16 + 3 * (cw + sw_ + 2 * gap)]
labels = ["chunk t−1", "chunk t", "chunk t+1"]
y0, hh = 34, 40
for n, x in enumerate(xs[:3]):
    hot = n == 1
    rect(x, y0, cw, hh, ACC_F if hot else GRAY_F, ACC if hot else GRAY, 2 if hot else 1.5, 4)
    for k in range(1, 8):
        o.append(f'<line x1="{x+k*cw/8:.1f}" y1="{y0+4}" x2="{x+k*cw/8:.1f}" y2="{y0+hh-4}" stroke="#BFBFBF" stroke-width="1"/>')
    t(x + cw / 2, y0 - 9, labels[n], 18, "bold" if hot else "normal", ACC if hot else INK)
    sx = x + cw + gap
    if n < 3:
        o.append(f'<line x1="{x+cw+1}" y1="{y0+hh/2}" x2="{sx-2}" y2="{y0+hh/2}" stroke="{ACC}" stroke-width="2.5" marker-end="url(#a)"/>')
        rect(sx, y0 + 2, sw_, hh - 4, "#FFFFFF", ACC, 2.5, 3)
        t(sx + sw_ / 2, y0 + hh / 2 + 6, "S", 18, "bold", ACC)
        nxt = sx + sw_ + gap
        o.append(f'<line x1="{sx+sw_+1}" y1="{y0+hh/2}" x2="{nxt-2}" y2="{y0+hh/2}" stroke="{ACC}" stroke-width="2.5" marker-end="url(#a)"/>')
t(xs[3] + 20, y0 + hh / 2 + 6, "⋯", 22, "bold", INK, "start")
t(xs[3] + 60, y0 + hh / 2 + 6, "C = 64 tokens", 17, "normal", GRAY, "start", "italic")
# ---- zoom of chunk t
zx, zy, zw, zh = 16, 104, 838, 182
cx = xs[1]
o.append(f'<line x1="{cx}" y1="{y0+hh}" x2="{zx}" y2="{zy}" stroke="{GRAY}" stroke-width="1.2" stroke-dasharray="4,3"/>')
o.append(f'<line x1="{cx+cw}" y1="{y0+hh}" x2="{zx+zw}" y2="{zy}" stroke="{GRAY}" stroke-width="1.2" stroke-dasharray="4,3"/>')
rect(zx, zy, zw, zh, "#FFFFFF", GRAY, 1.5, 6, 'stroke-dasharray="6,4"')
# panel A: intra-chunk causal block
t(150, zy + 28, "Intra-chunk: parallel", 19, "bold", ACC)
n, c = 8, 13
gx, gy = 40, zy + 42
for i in range(n):
    for j in range(n):
        f = "#9DC3E6" if j <= i else "#FFFFFF"
        rect(gx + j * c, gy + i * c, c, c, f, "#7F7F7F", 0.8)
t(gx + n * c + 14, gy + 30, "C × C causal", 17, "normal", INK, "start")
t(gx + n * c + 14, gy + 52, "mask (Tril)", 17, "normal", INK, "start")
t(gx + n * c + 14, gy + 84, "dense matmuls", 17, "normal", ACC, "start", "italic")
t(150, zy + 172, "Tril[(QΓ)(K/Γ)ᵀ] · (U − W S)", 18)
# divider
o.append(f'<line x1="300" y1="{zy+12}" x2="300" y2="{zy+zh-12}" stroke="#D9D9D9" stroke-width="1.2"/>')
# panel B: inter-chunk read of incoming state
t(436, zy + 28, "Inter-chunk: read state", 19, "bold", ACC)
rect(352, zy + 46, 34, 104, GRAY_F, GRAY, 1.5)
t(369, zy + 104, "Q", 18, "bold")
t(404, zy + 104, "×", 22, "bold")
rect(424, zy + 80, 40, 40, ACC_F, ACC, 2.5, 3)
t(444, zy + 106, "S", 18, "bold", ACC)
t(482, zy + 104, "=", 22, "bold")
rect(500, zy + 46, 34, 104, "#FFFFFF", GRAY, 1.5)
t(517, zy + 104, "O", 18, "bold")
t(436, zy + 172, "(Γ ⊙ Q) · S  +  intra term", 18)
o.append(f'<line x1="574" y1="{zy+12}" x2="574" y2="{zy+zh-12}" stroke="#D9D9D9" stroke-width="1.2"/>')
# panel C: hand-off
t(712, zy + 28, "Hand-off: next state", 19, "bold", ACC)
rect(600, zy + 62, 40, 40, ACC_F, ACC, 2.5, 3)
t(620, zy + 88, "S", 18, "bold", ACC)
o.append(f'<line x1="642" y1="{zy+82}" x2="772" y2="{zy+82}" stroke="{ACC}" stroke-width="2.5" marker-end="url(#a)"/>')
t(707, zy + 54, "decay + chunk write", 17, "normal", INK)
rect(774, zy + 62, 40, 40, ACC_F, ACC, 2.5, 3)
t(794, zy + 88, "S′", 18, "bold", ACC)
t(712, zy + 136, "only d<tspan font-size=\"13\" dy=\"4\">k</tspan><tspan dy=\"-4\"> × d</tspan><tspan font-size=\"13\" dy=\"4\">v</tspan><tspan dy=\"-4\"> crosses chunks</tspan>", 18, "bold", INK)
t(712, zy + 172, "serial: one hand-off per chunk", 17, "normal", RED, "middle", "italic")
o.append('</svg>')
p = pathlib.Path(__file__).with_name("fig_chunk.svg"); p.write_text("\n".join(o)); print("wrote", p)
