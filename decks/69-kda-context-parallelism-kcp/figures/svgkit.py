"""Tiny SVG helper shared by this deck's figure generators (slide scale, 1 unit = 1pt)."""
ACC, ACC_F, INK, GRAY, GRAY_F, RED, RED_F = "#1F4E79", "#DEEBF7", "#222222", "#595959", "#F2F2F2", "#C00000", "#FBE5E5"
class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif">',
                  '<defs>'
                  '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222222"/></marker>'
                  '<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1F4E79"/></marker>'
                  '<marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#C00000"/></marker>'
                  '<pattern id="hr" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#FBE5E5"/><line x1="0" y1="0" x2="0" y2="6" stroke="#C00000" stroke-width="1.6"/></pattern>'
                  '</defs>']
    def t(self, x, y, s, size=18, w="normal", fill=INK, anchor="middle", style="normal"):
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-style="{style}">{s}</text>')
    def box(self, x, y, w, h, f="#FFFFFF", s=GRAY, sw=1.5, rx=5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"{d}/>')
    def line(self, x1, y1, x2, y2, s=INK, sw=2, head="a", dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{head})"' if head else ""
        self.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{s}" stroke-width="{sw}"{m}{d}/>')
    def path(self, d, s=INK, sw=2, head="a", dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{head})"' if head else ""
        self.o.append(f'<path d="{d}" fill="none" stroke="{s}" stroke-width="{sw}"{m}{dd}/>')
    def badge(self, cx, cy, n, r=12):
        self.o.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{INK}"/>')
        self.t(cx, cy + 6, str(n), 17, "bold", "#FFFFFF")
    def raw(self, s):
        self.o.append(s)
    def save(self, path):
        import pathlib
        p = pathlib.Path(path); p.write_text("\n".join(self.o + ["</svg>"])); print("wrote", p)
def sub(base, s, size=14):
    return f'{base}<tspan font-size="{size}" dy="5">{s}</tspan><tspan dy="-5">\u200b</tspan>'
