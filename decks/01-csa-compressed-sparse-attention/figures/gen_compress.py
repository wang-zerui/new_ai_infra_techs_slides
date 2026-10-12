"""Mechanism 1: overlapping token-level compression (m = 4). Entry i pools 2m = 8 token projections:
its own block via C^a and the previous block via C^b, softmax-normalised per channel (V4 Eq. 11-12).
Two build steps: (1) entry 1 only, (2) entry 2 added, showing the 4-token overlap."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_FILL, GRAY, GRAY_FILL, INK

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 290
CW, PITCH, BGAP, X0, TY = 54, 60, 24, 64, 196


def cell_x(t):
    b, j = divmod(t, 4)
    return X0 + b * (4 * PITCH - 6 + BGAP) + j * PITCH


def block_cx(b):
    return cell_x(4 * b) + (4 * PITCH - 6) / 2


def draw(step):
    s = SVG(W, H)
    # legend (top-left)
    s.arrow(4, 22, 52, 22, ACC, 2.5, head=False)
    s.text(60, 28, "own block (Cᵃ)", 18, "normal", INK, "start")
    s.arrow(4, 52, 52, 52, ACC, 2.5, dash="7,5", head=False)
    s.text(60, 58, "previous block (Cᵇ)", 18, "normal", INK, "start")
    s.circle(20, 86, 14, "#fff", ACC, 2.5)
    s.text(20, 92, "Σ", 18, "bold", ACC)
    s.text(42, 92, "softmax over 8", 18, "normal", INK, "start")
    s.text(42, 114, "weights + bias", 18, "normal", INK, "start")

    entries = [1] if step == 1 else [1, 2]
    for e in entries:
        nx, ny = block_cx(e), 120
        own = range(4 * e, 4 * e + 4)
        prev = range(4 * (e - 1), 4 * e)
        for t in prev:
            s.arrow(cell_x(t) + CW / 2, TY - 2, nx, ny + 15, ACC, 2, dash="7,5", head=False)
        for t in own:
            s.arrow(cell_x(t) + CW / 2, TY - 2, nx, ny + 15, ACC, 2.2, head=False)
        s.circle(nx, ny, 16, "#fff", ACC, 2.5)
        s.text(nx, ny + 6.5, "Σ", 20, "bold", ACC)
        s.arrow(nx, ny - 17, nx, 80, ACC, 2.5)
        s.rect(nx - 80, 22, 160, 56, ACC_FILL, ACC, 2.5, 6)
        s.text(nx, 46, f"Entry {e}", 18, "bold", INK)
        s.text(nx, 68, "one 512-d vector", 16, "normal", INK)

    # tokens
    for t in range(12):
        x = cell_x(t)
        hot = step == 2 and 4 <= t < 8
        s.rect(x, TY, CW, 40, GRAY_FILL, ACC if hot else GRAY, 2.5 if hot else 2, 4)
        s.text(x + CW / 2, TY + 27, f"t{t}", 18, "bold" if hot else "normal", INK)
    for b in range(3):
        if step == 2 and b == 1:
            s.text(block_cx(b), TY + 70, "block 1 feeds both", 18, "bold", ACC)
        else:
            s.text(block_cx(b), TY + 70, f"block {b}", 18, "normal", GRAY)
    s.text(cell_x(11) + CW + 6, TY + 27, "…", 22, "bold", GRAY, "start")
    s.save(os.path.join(HERE, f"fig_compress_{step}.svg"))


draw(1)
draw(2)
