"""fig_bits: what each cuBLAS option does to an FP32 router weight (bit layout; kept vs dropped mantissa bits)."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 870, 292
s = SVG(W, H, "fig_bits: FP32 keeps 23 mantissa bits on CUDA cores; TF32 keeps 10, BF16 keeps 7 on Tensor Cores")

X0, CW, CH = 118, 17, 36          # first cell x, cell width, cell height (32 cells -> 560 pt)
XR = X0 + 32 * CW + 18              # right label column
rows = [("FP32", 23, "CUDA cores", True), ("TF32", 10, "Tensor Cores", False), ("BF16", 7, "Tensor Cores", False)]

# field header
y_h = 30
def bracket(c0, c1, label):
    xa, xb = X0 + c0 * CW + 2, X0 + c1 * CW - 2
    s.path(f"M{xa:.1f},{y_h + 10} L{xa:.1f},{y_h + 2} L{xb:.1f},{y_h + 2} L{xb:.1f},{y_h + 10}", MUTED, 1.5)
    s.text((xa + xb) / 2, y_h - 6, label, 17, MUTED)
bracket(0, 1, "sign")
bracket(1, 9, "exponent")
bracket(9, 32, "mantissa (23 bits)")

y = 52
for name, keep, engine, slow in rows:
    s.text(X0 - 14, y + CH / 2 + 6.5, name, 20, INK, anchor="end", weight="bold")
    for c in range(32):
        x = X0 + c * CW
        if c == 0:
            fill, stroke = NEU_F, NEU
        elif c < 9:
            fill, stroke = "#FFFFFF", NEU
        elif c < 9 + keep:
            fill, stroke = "#FFFFFF", NEU
        else:
            fill, stroke = "url(#hatchPrb)", PRB
        s.rect(x, y, CW, CH, fill, stroke, 1.2, 0)
    # kept-bits label inside the mantissa run
    sig = keep + 1
    if slow:
        s.text(XR, y + CH / 2 + 6.5, "CUDA cores only", 19, PRB, anchor="start", weight="bold")
    else:
        s.text(XR, y + CH / 2 + 6.5, engine, 19, INK, anchor="start")
    s.text(X0 + 9 * CW + 4, y + CH + 20, f"{sig} significant bits kept", 17,
           PRB if not slow else INK, anchor="start", style="italic")
    y += CH + 34

# legend
s.rect(X0, y - 4, 22, 18, "url(#hatchPrb)", PRB, 1.2, 0)
s.text(X0 + 30, y + 10, "weight bits dropped before the multiply", 17, INK, anchor="start")
s.save(os.path.join(HERE, "fig_bits.svg"))
