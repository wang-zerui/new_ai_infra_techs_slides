"""svgkit: tiny helpers for slide-scale IPADS-style SVG diagrams (1 viewBox unit = 1 pt).

Deck palette (docs/deck-recipe.md): technique fill #DEEBF7 / stroke #1F4E79; problem #FBE5E5 / #C00000;
neutral #F2F2F2 or white / #595959; text #222222.
"""
from xml.sax.saxutils import escape

ACC, ACC_F = "#1F4E79", "#DEEBF7"
PROB, PROB_F = "#C00000", "#FBE5E5"
NEU, NEU_F = "#595959", "#F2F2F2"
GRAY = "#7F7F7F"
INK = "#222222"
GOOD = "#2E7D32"
FONT = "Helvetica, Arial, sans-serif"


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.items = []
        self._colors = {}

    # ---- primitives -------------------------------------------------------
    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=NEU, sw=2, rx=4, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{o} '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=18, anchor="middle", weight="normal", fill=INK, italic=False, family=None,
             halo=False):
        st = ' font-style="italic"' if italic else ""
        fam = f' font-family="{family}"' if family else ""
        hl = ' paint-order="stroke" stroke="#FFFFFF" stroke-width="4"' if halo else ""
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" fill="{fill}"{st}{fam}{hl}>{escape(s)}</text>')

    def ctext(self, cx, cy, s, size=18, **kw):
        """Text vertically centred on cy (single line)."""
        self.text(cx, round(cy + 0.35 * size, 1), s, size=size, **kw)

    def lines(self, cx, cy, rows, size=18, step=None, **kw):
        """Several centred lines around cy. rows: list of str or (str, dict)."""
        step = step or round(1.2 * size, 1)
        n = len(rows)
        y0 = cy - step * (n - 1) / 2
        for i, r in enumerate(rows):
            s, extra = (r, {}) if isinstance(r, str) else r
            args = dict(kw); args.update(extra)
            self.ctext(cx, y0 + i * step, s, size=args.pop("size", size), **args)

    def line(self, x1, y1, x2, y2, color=INK, sw=2.5, arrow=True, dash=None, start=False):
        m = f' marker-end="url(#a{self._mk(color)})"' if arrow else ""
        ms = f' marker-start="url(#a{self._mk(color)})"' if start else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{m}{ms}{d}/>')

    def path(self, d, color=INK, sw=2.5, arrow=True, dash=None, fill="none"):
        m = f' marker-end="url(#a{self._mk(color)})"' if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{m}{ds}/>')

    def badge(self, cx, cy, n, color=ACC, r=13, size=17):
        self.items.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
        self.text(cx, round(cy + 0.36 * size, 1), str(n), size=size, weight="bold", fill="#FFFFFF")

    def raw(self, s):
        self.items.append(s)

    # ---- markers ----------------------------------------------------------
    def _mk(self, color):
        key = color.lstrip("#")
        self._colors[key] = color
        return key

    def svg(self):
        defs = "".join(
            f'<marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for k, c in self._colors.items())
        defs += ('<pattern id="hatchGray" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                 '<rect width="7" height="7" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="7" stroke="#A6A6A6" stroke-width="2.5"/></pattern>'
                 '<pattern id="hatchRed" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                 '<rect width="7" height="7" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="7" stroke="#E6A0A0" stroke-width="2.5"/></pattern>')
        body = "\n  ".join(self.items)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">\n  <defs>{defs}</defs>\n  {body}\n</svg>\n')

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.svg())
        print("wrote", path)
