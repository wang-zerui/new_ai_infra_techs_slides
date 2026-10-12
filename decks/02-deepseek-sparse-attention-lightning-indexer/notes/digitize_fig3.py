"""Read the plotted cost curves of DeepSeek-V3.2 Fig. 3 (arXiv 2512.02556v1, page 6) from the PDF's vector data.

The two panels are drawn as straight polylines (matplotlib), so the line vertices give the plotted values exactly
(up to PDF coordinate precision, ~0.01 pt ~ 0.0001 $ prefill / 0.0002 $ decode).
Axis calibration comes from the tick-label positions on the same page:
  prefill: x 0K..128K = 105.1..275.9 pt, y $0..$0.7 = 228.6..97.6 pt
  decode : x 0K..128K = 334.4..505.2 pt, y $0..$2.4 = 228.6..97.6 pt
Legend: blue (0.12,0.47,0.71) = DeepSeek-V3.1-Terminus, orange (1.0,0.5,0.05) = DeepSeek-V3.2.
Run: uv run --with pymupdf python -I notes/digitize_fig3.py <path-to-arxiv-2512.02556.pdf>
"""
import sys
import pymupdf

PANELS = {"prefill": (105.1, 275.9, 0.7), "decode": (334.4, 505.2, 2.4)}
Y0, Y1 = 228.6, 97.6


def to_val(panel, x, y):
    x0, x1, ymax = PANELS[panel]
    return (x - x0) / (x1 - x0) * 128.0, (Y0 - y) / (Y0 - Y1) * ymax


def curves(pdf):
    page = pymupdf.open(pdf)[5]
    out = {}
    for d in page.get_drawings():
        c = d.get("color")
        if not c or d.get("width", 0) < 0.6:
            continue
        name = {(0.12, 0.47, 0.71): "V3.1-Terminus", (1.0, 0.5, 0.05): "V3.2"}.get(tuple(round(v, 2) for v in c))
        if not name:
            continue
        pts = []
        for it in d["items"]:
            pts += [it[1], it[2]]
        xs = sorted({(round(p.x, 2), round(p.y, 2)) for p in pts})
        if xs[0][0] < 140 and xs[-1][0] > 270:
            panel = "prefill"
        elif xs[0][0] > 330 and xs[-1][0] > 500:
            panel = "decode"
        else:
            continue  # legend swatches
        out[(panel, name)] = [to_val(panel, x, y) for x, y in xs]
    return out


def interp(poly, k):
    k = min(max(k, poly[0][0]), poly[-1][0])
    for (xa, ya), (xb, yb) in zip(poly, poly[1:]):
        if xa <= k <= xb:
            return ya + (yb - ya) * (k - xa) / (xb - xa)
    raise ValueError(k)


if __name__ == "__main__":
    cv = curves(sys.argv[1])
    for (panel, name), poly in sorted(cv.items()):
        print(f"{panel:8s} {name:14s} vertices (K tokens, $/M tokens):", [(round(a, 1), round(b, 4)) for a, b in poly])
    for panel in ("prefill", "decode"):
        a, b = cv[(panel, "V3.1-Terminus")], cv[(panel, "V3.2")]
        print(f"{panel}: at 0K  V3.1={interp(a,0):.4f}  V3.2={interp(b,0):.4f}")
        print(f"{panel}: at 128K V3.1={interp(a,128):.4f}  V3.2={interp(b,128):.4f}  ratio={interp(a,128)/interp(b,128):.2f}x")
        print(f"{panel}: V3.1 128K/0K growth = {interp(a,128)/interp(a,0):.1f}x ; V3.2 = {interp(b,128)/interp(b,0):.2f}x")
        # crossover: first position where V3.2 becomes cheaper than V3.1
        k = 0.0
        while k <= 128 and interp(b, k) >= interp(a, k):
            k += 0.01
        print(f"{panel}: V3.2 cheaper than V3.1 from ~{k:.1f}K tokens on")
