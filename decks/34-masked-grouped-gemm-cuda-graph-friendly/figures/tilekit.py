"""Shared tile-grid visual language for the grouped-GEMM decks (32, 33, 34).

Expert identity = one pastel fill + darker stroke per expert (E0 green, E1 orange, E2 purple, E3 gold);
alignment padding = gray hatch; problem = red; DeepGEMM's own mechanism (tile borders, lookups,
scheduler) = accent blue. 1 user unit = 1 pt.
"""
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED, GOOD  # noqa: F401

EXP = [("#E2F0D9", "#548235"), ("#FCE4D6", "#C55A11"), ("#E4DFEC", "#7030A0"), ("#FFF2CC", "#BF9000")]


def seg(s, x, y, w, h, e, label=None, size=17, sw=1.6, weight="bold", halo=False):
    """A block of rows (or a weight slab) owned by expert e."""
    f, st = EXP[e]
    s.rect(x, y, w, h, f, st, sw, rx=0)
    if label:
        s.text(x + w / 2, y + h / 2 + 0.35 * size, label, size, INK, weight=weight, halo=halo)


def pad(s, x, y, w, h, label=None, size=15):
    """Alignment padding rows: zero data, index -1."""
    s.rect(x, y, w, h, "url(#hatchGray)", MUTED, 1.2, rx=0)
    if label:
        s.text(x + w / 2, y + h / 2 + 0.35 * size, label, size, MUTED, halo=True)


def waste(s, x, y, w, h, label=None, size=15):
    """Wasted computation (problem)."""
    s.rect(x, y, w, h, "url(#hatchPrb)", PRB, 1.4, rx=0)
    if label:
        s.text(x + w / 2, y + h / 2 + 0.35 * size, label, size, PRB, weight="bold", halo=True)


def hlines(s, x, y0, w, ys, color=ACC, dash="5,3", sw=1.6):
    for y in ys:
        s.line(x, y, x + w, y, color, sw, dash=dash)


def vlines(s, xs, y, h, color=ACC, dash="5,3", sw=1.6):
    for x in xs:
        s.line(x, y, x, y + h, color, sw, dash=dash)


def outline(s, x, y, w, h, color=INK, sw=2.2):
    s.rect(x, y, w, h, "none", color, sw, rx=0)


def mono(s, x, y, t, size=16, color=INK, anchor="middle", weight="normal", halo=False):
    s.text(x, y, t, size, color, anchor=anchor, weight=weight, family="Menlo, Consolas, monospace", halo=halo)
