"""fig_tiles.svg: intra-chunk tile paths before/after K3's lower-bounded decay (K3 Sec. 2.1.1, Eq. 5, Fig. 3b).
A 64-token chunk = 4 x 4 tiles of 16 tokens. Kimi Linear: diagonal tiles need an explicit position-pair path;
K3 (g >= g_min = -5): every causal tile is a dense Tensor-Core matmul."""
import pathlib
ACC, ACC_F, INK, GRAY, RED, RED_F = "#1F4E79", "#DEEBF7", "#222222", "#595959", "#C00000", "#FBE5E5"
W, H = 870, 285
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1F4E79"/></marker>'
     '<pattern id="hr" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#FBE5E5"/><line x1="0" y1="0" x2="0" y2="6" stroke="#C00000" stroke-width="2"/></pattern></defs>']
def t(x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
def rect(x, y, w, h, f, s, sw=1.5):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{f}" stroke="{s}" stroke-width="{sw}"/>')
def tiles(x0, y0, diag_red):
    c = 44
    for i in range(4):
        for j in range(4):
            if j > i:
                rect(x0 + j * c, y0 + i * c, c, c, "#FFFFFF", "#BFBFBF", 1)
    for i in range(4):
        for j in range(4):
            if j > i:
                continue
            elif j == i and diag_red:
                rect(x0 + j * c, y0 + i * c, c, c, "url(#hr)", RED, 2)
            else:
                rect(x0 + j * c, y0 + i * c, c, c, ACC_F, ACC, 2)
    t(x0 - 10, y0 + 2 * c + 6, "query", 16, "normal", GRAY, "end")
    t(x0 + 2 * c, y0 + 4 * c + 22, "key tile (16 tokens)", 16, "normal", GRAY)
# panel A
t(190, 28, "Kimi Linear: g ∈ (−∞, 0)", 20, "bold")
tiles(102, 46, True)
# panel B
t(680, 28, "Kimi K3: g ∈ (−5, 0)", 20, "bold", ACC)
tiles(592, 46, False)
# middle arrow
o.append(f'<line x1="330" y1="134" x2="520" y2="134" stroke="{ACC}" stroke-width="3" marker-end="url(#a)"/>')
t(425, 120, "bounded log-decay", 18, "bold", ACC)
t(425, 160, "g = −5 · σ(eᴬ z)", 18, "normal", INK)
# legend
ly = 268
rect(60, ly - 15, 22, 18, ACC_F, ACC, 2); t(92, ly, "Tensor Core matmul", 17, "normal", INK, "start")
rect(330, ly - 15, 22, 18, "url(#hr)", RED, 2); t(362, ly, "position-pair path", 17, "normal", RED, "start")
rect(600, ly - 15, 22, 18, "#FFFFFF", "#BFBFBF", 1); t(632, ly, "masked (future)", 17, "normal", INK, "start")
o.append('</svg>')
p = pathlib.Path(__file__).with_name("fig_tiles.svg"); p.write_text("\n".join(o)); print("wrote", p)
