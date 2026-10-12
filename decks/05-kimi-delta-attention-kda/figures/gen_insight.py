"""fig_insight.svg: the delta rule as an online-learning loop (Kimi Linear Sec. 2.2; K3 Eq. 1):
memory guesses S^T k, compares with v, and writes only the error e = v - S^T k (after per-channel decay)."""
import pathlib
ACC, ACC_F, INK, GRAY, GRAY_F = "#1F4E79", "#DEEBF7", "#222222", "#595959", "#F2F2F2"
W, H = 470, 280
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker>'
     '<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1F4E79"/></marker></defs>']
def t(x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
def box(x, y, w, h, f, s, sw=1.5, rx=5):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"/>')
# memory
box(150, 88, 140, 84, ACC_F, ACC, 2.5, 6)
t(220, 124, "Memory S", 20, "bold", ACC)
t(220, 150, "fixed size", 17)
# key in
box(8, 112, 96, 36, "#FFFFFF", INK)
t(56, 136, "key k", 18)
o.append(f'<line x1="104" y1="130" x2="148" y2="130" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
# guess out -> compare
o.append(f'<line x1="290" y1="130" x2="336" y2="130" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
t(312, 118, "Sᵀk", 18, "bold")
o.append(f'<circle cx="356" cy="130" r="18" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
t(356, 137, "−", 24, "bold")
box(316, 14, 80, 36, "#FFFFFF", INK)
t(356, 38, "value v", 18)
o.append(f'<line x1="356" y1="50" x2="356" y2="110" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
# error
o.append(f'<line x1="356" y1="148" x2="356" y2="196" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
box(306, 198, 100, 38, ACC_F, ACC, 2.5)
t(356, 223, "error e", 18, "bold", ACC)
o.append(f'<path d="M306,217 H254 V174" fill="none" stroke="{ACC}" stroke-width="3" marker-end="url(#b)"/>')
t(282, 262, "write β·k·eᵀ", 18, "bold", ACC)
# decay
box(8, 198, 130, 38, GRAY_F, GRAY)
t(73, 223, "decay α", 18)
o.append(f'<path d="M138,217 H186 V174" fill="none" stroke="{INK}" stroke-width="2" marker-end="url(#a)"/>')
t(73, 262, "per key channel", 17, "normal", GRAY, "middle", "italic")
o.append('</svg>')
p = pathlib.Path(__file__).with_name("fig_insight.svg"); p.write_text("\n".join(o)); print("wrote", p)
