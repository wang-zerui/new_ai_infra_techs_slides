"""Tiny SVG helper for slide-scale IPADS-style diagrams (1 viewBox unit = 1 pt).

Colour roles follow docs/deck-recipe.md: technique fill #DEEBF7 / stroke #1F4E79, problem fill #FBE5E5 /
stroke #C00000, neutral fill #F2F2F2 or white / stroke #595959, text #222222.
"""
from html import escape

ACC, ACC_FILL, ACC_LIGHT = "#1F4E79", "#DEEBF7", "#9DC3E6"
RED, RED_FILL = "#C00000", "#FBE5E5"
GRAY, GRAY_FILL, GRAY_MID, INK = "#595959", "#F2F2F2", "#A6A6A6", "#222222"
FONT = "Helvetica, Arial, sans-serif"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    # ---------------------------------------------------------------- primitives
    def rect(self, x, y, w, h, fill="#fff", stroke=GRAY, sw=2, rx=6, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d} {extra}/>')

    def text(self, x, y, s, size=18, weight="normal", color=INK, anchor="middle", italic=False, family=None):
        st = ' font-style="italic"' if italic else ""
        fam = f' font-family="{family}"' if family else ""
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                          f'text-anchor="{anchor}"{st}{fam}>{s}</text>')

    def lines(self, cx, cy, rows, size=18, weight="normal", color=INK, anchor="middle", step=1.18):
        """Vertically centred multi-line label around (cx, cy)."""
        n = len(rows)
        y0 = cy - step * size * (n - 1) / 2 + 0.35 * size
        for i, r in enumerate(rows):
            w = weight[i] if isinstance(weight, (list, tuple)) else weight
            c = color[i] if isinstance(color, (list, tuple)) else color
            sz = size[i] if isinstance(size, (list, tuple)) else size
            self.text(cx, round(y0 + i * step * (size if not isinstance(size, (list, tuple)) else max(size)), 1),
                      r, sz, w, c, anchor)

    def box(self, x, y, w, h, rows, fill="#fff", stroke=GRAY, sw=2, size=18, weight="bold", color=INK, rx=6,
            dash=None):
        self.rect(x, y, w, h, fill, stroke, sw, rx, dash)
        if isinstance(rows, str):
            rows = [rows]
        self.lines(x + w / 2, y + h / 2, rows, size, weight, color)

    def path(self, d, color=INK, sw=2.2, head=True, dash=None, fill="none"):
        mk = f' marker-end="url(#{self._marker(color)})"' if head else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{ds}{mk}/>')

    def arrow(self, x1, y1, x2, y2, color=INK, sw=2.2, dash=None, head=True):
        self.path(f"M{x1},{y1} L{x2},{y2}", color, sw, head, dash)

    def badge(self, cx, cy, n, color=INK, r=13):
        self.items.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
        self.text(cx, cy + 6.3, str(n), 18, "bold", "#fff")

    def circle(self, cx, cy, r, fill="#fff", stroke=INK, sw=2):
        self.items.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def raw(self, s):
        self.items.append(s)

    # ---------------------------------------------------------------- output
    _markers = {}

    def _marker(self, color):
        mid = "m" + color.strip("#")
        SVG._markers[mid] = color
        return mid

    def save(self, path):
        defs = "".join(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" '
            f'markerWidth="12" markerHeight="12" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{c}"/></marker>' for mid, c in SVG._markers.items())
        defs += ('<pattern id="hatchRed" width="7" height="7" patternUnits="userSpaceOnUse" '
                 'patternTransform="rotate(45)"><rect width="7" height="7" fill="#FBE5E5"/>'
                 f'<line x1="0" y1="0" x2="0" y2="7" stroke="{RED}" stroke-width="2"/></pattern>'
                 '<pattern id="hatchGray" width="7" height="7" patternUnits="userSpaceOnUse" '
                 'patternTransform="rotate(45)"><rect width="7" height="7" fill="#fff"/>'
                 '<line x1="0" y1="0" x2="0" y2="7" stroke="#A6A6A6" stroke-width="2"/></pattern>')
        P = 3  # pad so strokes on the drawing's outer edge are not clipped
        W, H = self.w + 2 * P, self.h + 2 * P
        out = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" '
               f'viewBox="0 0 {W} {H}" font-family="{FONT}">\n<defs>{defs}</defs>\n'
               f'<rect width="{W}" height="{H}" fill="#fff"/>\n<g transform="translate({P},{P})">\n'
               + "\n".join(self.items) + "\n</g>\n</svg>\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print("wrote", path)


def esc(s):
    return escape(s)
