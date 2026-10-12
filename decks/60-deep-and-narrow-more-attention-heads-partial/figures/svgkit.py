"""Tiny SVG helper for slide-scale figures (1 user unit = 1 pt). Deck-wide colour roles."""
import html

ACC, ACC_F = "#1F4E79", "#DEEBF7"      # technique / new
PRB, PRB_F = "#C00000", "#FBE5E5"      # problem
NEU, NEU_F = "#595959", "#F2F2F2"      # neutral / pre-existing
INK, MUTED = "#222222", "#7F7F7F"
GOOD = "#2E7D32"

DEFS = f"""<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11"
          markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>
  <marker id="arrAcc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="12" markerHeight="12"
          markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{ACC}"/></marker>
  <marker id="arrPrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="12" markerHeight="12"
          markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{PRB}"/></marker>
  <marker id="arrGray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11"
          markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>
  <pattern id="hatchAcc" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="7" height="7" fill="{ACC_F}"/><line x1="0" y1="0" x2="0" y2="7" stroke="{ACC}" stroke-width="1.6"/>
  </pattern>
  <pattern id="hatchGray" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="7" height="7" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="7" stroke="#A6A6A6" stroke-width="2"/>
  </pattern>
</defs>"""


class SVG:
    def __init__(self, w, h, comment=""):
        self.w, self.h, self.parts = w, h, []
        if comment:
            self.parts.append(f"<!-- {comment} -->")

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=NEU, sw=2, rx=4, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        st = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else 'stroke="none"'
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'{st}{d} {extra}/>')

    def text(self, x, y, s, size=18, color=INK, anchor="middle", weight="normal", style="normal", halo=False,
             family=None, rotate=None):
        h = ' paint-order="stroke" stroke="#FFFFFF" stroke-width="5" stroke-linejoin="round"' if halo else ""
        f = f' font-family="{family}"' if family else ""
        r = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}" '
                 f'font-weight="{weight}" font-style="{style}"{h}{f}{r}>{html.escape(s)}</text>')

    def box(self, x, y, w, h, label, fill="#FFFFFF", stroke=NEU, color=INK, size=18, weight="normal", sw=2, rx=4,
            dash=None):
        self.rect(x, y, w, h, fill, stroke, sw, rx, dash)
        if isinstance(label, str):
            label = [label]
        n = len(label)
        step = size * 1.15
        y0 = y + h / 2 - step * (n - 1) / 2 + 0.35 * size
        for i, t in enumerate(label):
            self.text(x + w / 2, y0 + i * step, t, size, color, weight=weight)

    def badge(self, cx, cy, n, r=13, fill=ACC, size=17):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>')
        self.text(cx, cy + 0.35 * size, str(n), size, "#FFFFFF", weight="bold")

    def line(self, x1, y1, x2, y2, color=INK, sw=2.2, marker=None, dash=None, start_marker=None):
        m = f' marker-end="url(#{marker})"' if marker else ""
        ms = f' marker-start="url(#{start_marker})"' if start_marker else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                 f'stroke-width="{sw}"{m}{ms}{d}/>')

    def path(self, d, color=INK, sw=2.2, marker=None, dash=None, fill="none"):
        m = f' marker-end="url(#{marker})"' if marker else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{m}{da}/>')

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" '
                f'viewBox="0 0 {self.w} {self.h}" font-family="Helvetica, Arial, sans-serif">')
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + "\n" + DEFS + "\n" + "\n".join(self.parts) + "\n</svg>\n")
        print("wrote", path)
