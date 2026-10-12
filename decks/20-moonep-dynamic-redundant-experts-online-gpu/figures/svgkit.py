"""Tiny SVG helper for slide-scale figures (1 user unit = 1 pt). Shared by gen_*.py in this folder."""
import html

ACCENT, ACCENT_FILL = "#1F4E79", "#DEEBF7"
PROBLEM, PROBLEM_FILL = "#C00000", "#FBE5E5"
NEUTRAL, NEUTRAL_FILL = "#595959", "#F2F2F2"
MUTED = "#7F7F7F"
INK = "#222222"
GOOD = "#2E7D32"


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.items = []

    # ---- primitives
    def rect(self, x, y, w, h, fill="#fff", stroke=NEUTRAL, sw=2, rx=5, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} {extra}/>')

    def text(self, x, y, s, size=18, color=INK, weight="normal", anchor="middle", italic=False, family=None):
        st = ' font-style="italic"' if italic else ""
        fam = f' font-family="{family}"' if family else ""
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                          f'text-anchor="{anchor}"{st}{fam}>{html.escape(s)}</text>')

    def ctext(self, cx, cy, lines, size=18, color=INK, weight="normal", anchor="middle", lh=1.18, family=None):
        """Vertically centred multi-line text around (cx, cy)."""
        if isinstance(lines, str):
            lines = [lines]
        n = len(lines)
        y0 = cy - 0.575 * size * (n - 1) * (lh / 1.15) + 0.35 * size
        for i, s in enumerate(lines):
            self.text(cx, y0 + i * size * lh, s, size, color, weight, anchor, family=family)

    def box(self, x, y, w, h, lines, kind="neutral", size=18, weight="normal", sw=2, rx=5, dash=None, tcolor=None):
        fill, stroke, tc = {"neutral": (NEUTRAL_FILL, NEUTRAL, INK), "white": ("#fff", NEUTRAL, INK),
                            "accent": (ACCENT_FILL, ACCENT, ACCENT), "problem": (PROBLEM_FILL, PROBLEM, PROBLEM)}[kind]
        self.rect(x, y, w, h, fill, stroke, sw, rx, dash)
        self.ctext(x + w / 2, y + h / 2, lines, size, tcolor or (INK if kind in ("neutral", "white") else tc), weight)

    def line(self, x1, y1, x2, y2, color=INK, sw=2, dash=None, arrow=False, arrow_start=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        ms = f' marker-start="url(#a{color[1:]})"' if arrow_start else ""
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                          f'stroke-width="{sw}"{d}{m}{ms}/>')

    def path(self, d, color=INK, sw=2, dash=None, arrow=False, fill="none"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.items.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{da}{m}/>')

    def raw(self, s):
        self.items.append(s)

    def badge(self, cx, cy, n, color=ACCENT, r=13, size=17):
        self.items.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}"/>')
        self.text(cx, cy + 0.36 * size, str(n), size, "#fff", "bold")

    # ---- output
    def save(self, path, colors=(INK, ACCENT, PROBLEM, NEUTRAL, MUTED, GOOD)):
        defs = []
        for c in colors:
            defs.append(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" '
                        f'markerHeight="11" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
                        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
        defs.append('<pattern id="hatchRed" width="8" height="8" patternUnits="userSpaceOnUse" '
                    'patternTransform="rotate(45)"><rect width="8" height="8" fill="#FBE5E5"/>'
                    '<line x1="0" y1="0" x2="0" y2="8" stroke="#C00000" stroke-width="2.5"/></pattern>')
        defs.append('<pattern id="hatchGray" width="8" height="8" patternUnits="userSpaceOnUse" '
                    'patternTransform="rotate(45)"><rect width="8" height="8" fill="#fff"/>'
                    '<line x1="0" y1="0" x2="0" y2="8" stroke="#A6A6A6" stroke-width="2"/></pattern>')
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" '
               f'viewBox="0 0 {self.w} {self.h}" font-family="Helvetica, Arial, sans-serif">',
               "<defs>" + "".join(defs) + "</defs>",
               f'<rect width="{self.w}" height="{self.h}" fill="#fff"/>'] + self.items + ["</svg>"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        print("wrote", path)
