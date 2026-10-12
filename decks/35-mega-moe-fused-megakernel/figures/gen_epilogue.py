"""fig_epilogue: what the two GEMM epilogues and the final combine do inside the kernel."""
import os
from svgkit import SVG, ACC, ACC_F, NEU, NEU_F, INK, MUTED

W, H = 870, 292
s = SVG(W, H, "epilogues absorb SwiGLU, FP8 cast, combine send and reduce")

LX = 146
R1, R2, R3 = 16, 116, 220
BH = 56


def header(y, title, where):
    s.text(LX, y + 26, title, 18, INK, anchor="end", weight="bold")
    s.text(LX, y + 46, where, 15, MUTED, anchor="end", style="italic")


def chain(y, specs, gap=26):
    xs = []
    x = 160
    for w, lab, kind in specs:
        fill, stroke = (ACC_F, ACC) if kind == "acc" else (NEU_F, NEU) if kind == "neu" else ("#FFFFFF", ACC)
        s.box(x, y, w, BH, lab, fill=fill, stroke=stroke, sizes=[17, 16][:len(lab)] if isinstance(lab, list) else None,
              size=17)
        xs.append((x, x + w))
        x += w + gap
    for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
        s.line(a1, y + BH / 2, b0 - 1, y + BH / 2, INK, 2.2, "arr")
    return xs


# row 1: linear-1 epilogue
header(R1, "L1 epilogue", "expert's GPU")
chain(R1, [(150, ["TMEM", "FP32 gate, up"], "neu"),
           (160, ["silu(gate) · up", "× routing weight"], "acc"),
           (160, ["amax per 32", "→ UE8M0 scale"], "acc"),
           (150, ["FP8 row", "→ L2 ring"], "acc")])

# row 2: linear-2 epilogue
header(R2, "L2 epilogue", "expert's GPU")
r2 = chain(R2, [(150, ["TMEM", "FP32 row"], "neu"), (150, ["BF16 row", "in shared mem"], "acc")])

# row 3: combine on the token's GPU
header(R3, "combine", "token's GPU")
labels = ["S", "0", "1", "2", "3", "4", "5"]
cw, cg = 34, 6
cx0 = 160
for i, lab in enumerate(labels):
    x = cx0 + i * (cw + cg)
    hl = lab == "2"
    s.box(x, R3 + 10, cw, 36, lab, fill=ACC if hl else "#FFFFFF", stroke=ACC, color="#FFFFFF" if hl else INK, size=17,
          weight="bold" if hl else "normal")
s.text(cx0 + (7 * (cw + cg) - cg) / 2, R3 + 66, "slots: shared + top-6", 15, MUTED, style="italic")
slots_end = cx0 + 7 * (cw + cg) - cg
sx0 = slots_end + 26
s.line(slots_end + 2, R3 + 28, sx0 - 1, R3 + 28, INK, 2.2, "arr")
s.box(sx0, R3, 200, BH, ["Σ in FP32", "fixed slot order"], fill=ACC_F, stroke=ACC, sizes=[17, 16])
yx0 = sx0 + 200 + 26
s.line(sx0 + 200, R3 + 28, yx0 - 1, R3 + 28, INK, 2.2, "arr")
s.box(yx0, R3, 110, BH, ["y", "BF16"], fill=NEU_F, stroke=NEU, sizes=[18, 16], weights=["bold", "normal"])

# NVLink store: from the BF16 row (expert's GPU) into slot k=2 of token t (token's GPU)
bx = (r2[1][0] + r2[1][1]) / 2
slot_x = cx0 + 3 * (cw + cg) + cw / 2
s.path(f"M {bx + 30} {R2 + BH} L {slot_x + 6} {R3 + 9}", ACC, 2.8, "arrAcc")
s.text(300, R2 + BH + 24, "NVLink store", 17, ACC, anchor="end", weight="bold")
s.text(300, R2 + BH + 43, "into slot k of token t", 16, ACC, anchor="end")

s.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_epilogue.svg"))
