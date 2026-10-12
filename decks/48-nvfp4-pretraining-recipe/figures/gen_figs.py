"""Generate the SVG diagrams for deck 48 (NVFP4 pretraining recipe). Run: python3 gen_figs.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PROB, PROB_F, NEU, NEU_F, GRAY, INK  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(OUT, name)


def box(s, x, y, w, h, lab, kind="neu", sub=None, size=17, subsize=15):
    f, st, col = {"acc": (ACC_F, ACC, ACC), "neu": (NEU_F, NEU, INK), "prob": (PROB_F, PROB, PROB),
                  "white": ("#FFFFFF", NEU, INK)}[kind]
    s.rect(x, y, w, h, fill=f, stroke=st, sw=2)
    if sub:
        s.lines(x + w / 2, y + h / 2, [lab, (sub, {"size": subsize, "weight": "normal"})], size=size,
                weight="bold", fill=col, step=size + 4)
    else:
        s.ctext(x + w / 2, y + h / 2, lab, size=size, weight="bold", fill=col)


def bits(s, x, y, layout, cw=26, ch=30):
    for i, b in enumerate(layout):
        f, st = {"s": (NEU_F, NEU), "e": (ACC_F, ACC), "m": ("#FFFFFF", NEU)}[b]
        s.rect(x + i * cw, y, cw, ch, fill=f, stroke=st, sw=1.6, rx=0)


def sub_text(s, x, y, parts, size=21, anchor="middle", fill=INK):
    """parts: list of (text, kind) with kind in {'', 'sub', 'sup'}."""
    spans = []
    for t, k in parts:
        if k == "sub":
            spans.append(f'<tspan baseline-shift="sub" font-size="{size - 6}">{t}</tspan>')
        elif k == "sup":
            spans.append(f'<tspan baseline-shift="super" font-size="{size - 6}">{t}</tspan>')
        else:
            spans.append(t)
    s.raw(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="bold" fill="{fill}">'
          + "".join(spans) + "</text>")


# ---------------------------------------------------------------- fig_block: NVFP4 block anatomy
def fig_block():
    s = SVG(870, 300)
    x0, cw = 24, 34
    s.text(x0 + 8 * cw, 24, "16 values q in E2M1 (4 bits each)", size=17, weight="bold")
    for i in range(16):
        s.rect(x0 + i * cw, 36, cw, 40, fill="#FFFFFF", stroke=NEU, sw=1.4, rx=0)
    s.rect(x0 + 2 * cw, 36, cw, 40, fill=NEU_F, stroke=INK, sw=2.4, rx=0)
    bx = x0 + 16 * cw + 18
    s.text(bx + 62, 24, "block scale", size=17, weight="bold", fill=ACC)
    s.rect(bx, 36, 124, 40, fill=ACC_F, stroke=ACC, sw=2.4, rx=2)
    s.ctext(bx + 62, 56, "E4M3 · 1 byte", size=16, weight="bold", fill=ACC)
    tx = bx + 124 + 16
    s.text(tx + 70, 24, "tensor scale", size=17, weight="bold", fill=GRAY)
    s.rect(tx, 36, 140, 40, fill=NEU_F, stroke=NEU, sw=2, rx=2, dash="6,4")
    s.ctext(tx + 70, 56, "FP32", size=17, weight="bold")
    s.text(tx + 70, 98, "1 per tensor", size=15, fill=GRAY)
    # zooms
    ey = 150
    ex = 80
    s.line(x0 + 2.5 * cw, 78, ex + 52, ey - 40, color=GRAY, sw=1.4, arrow=False, dash="4,3")
    s.text(ex + 52, ey - 18, "E2M1 value", size=17, weight="bold")
    bits(s, ex, ey, "seem")
    s.text(ex + 52, ey + 54, "±0 0.5 1 1.5 2 3 4 6", size=16, fill=GRAY)
    zx = 520
    s.line(bx + 62, 78, zx + 104, ey - 40, color=ACC, sw=1.4, arrow=False, dash="4,3")
    s.text(zx + 104, ey - 18, "E4M3 block scale", size=17, weight="bold", fill=ACC)
    bits(s, zx, ey, "seeeemmm")
    s.text(zx + 104, ey + 54, "fractional scale, not a power of two", size=16, fill=GRAY)
    s.rect(230, 240, 410, 50, fill="#FFFFFF", stroke=INK, sw=1.6, rx=6)
    sub_text(s, 435, 273, [("x = q × s", ""), ("block", "sub"), (" × s", ""), ("tensor", "sub")], size=22)
    s.save(out("fig_block.svg"))


# ---------------------------------------------------------------- fig_samples: what the scale does to FP4 samples
def fig_samples(step):
    samples = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    def X(v):
        return 196 + v * 90

    H = 190 if step == 1 else 280
    s = SVG(870, H)
    rows = [("MXFP4", "power-of-two scale", "mx")]
    if step == 2:
        rows.append(("NVFP4", "E4M3 scale", "nv"))
    y = 56
    for name, sub, kind in rows:
        col = INK if kind == "mx" else ACC
        s.text(160, y + 2, name, size=19, anchor="end", weight="bold", fill=col)
        s.text(160, y + 24, sub, size=15, anchor="end", fill=GRAY)
        s.line(X(0), y, X(6), y, color="#BFBFBF", sw=1.5, arrow=False)
        for v in samples:
            if kind == "mx" and v in (4, 6):
                s.raw(f'<circle cx="{X(v)}" cy="{y}" r="10" fill="#FFFFFF" stroke="{PROB}" stroke-width="2.5"/>')
            else:
                fill = NEU if kind == "mx" else ACC
                s.raw(f'<circle cx="{X(v)}" cy="{y}" r="10" fill="{fill}" stroke="{fill}" stroke-width="2"/>')
        mx = 3 if kind == "mx" else 6
        mcol = PROB if kind == "mx" else ACC
        s.path(f"M{X(mx)},{y - 40} V{y - 16}", color=mcol, sw=2.4)
        s.text(X(mx), y - 44, "block max", size=15, fill=mcol, weight="bold")
        if kind == "mx":
            s.text((X(4) + X(6)) / 2, y + 34, "never used", size=16, fill=PROB, weight="bold")
            s.text(862, y - 4, "2.58 of 3.58", size=16, anchor="end", fill=PROB, weight="bold")
            s.text(862, y + 18, "binades", size=16, anchor="end", fill=PROB)
        else:
            s.text(862, y - 4, "all 3.58", size=16, anchor="end", fill=ACC, weight="bold")
            s.text(862, y + 18, "binades", size=16, anchor="end", fill=ACC)
        y += 96
    ay = H - 44
    s.line(X(0), ay, X(6) + 20, ay, color=INK, sw=2)
    for v in samples:
        s.line(X(v), ay - 6, X(v), ay + 6, color=INK, sw=2, arrow=False)
        s.text(X(v), ay + 28, f"{v:g}", size=17)
    s.text(160, ay + 6, "E2M1", size=17, anchor="end", weight="bold")
    s.save(out(f"fig_samples_{step}.svg"))


# ---------------------------------------------------------------- fig_overview: the recipe on one linear layer
def fig_overview():
    s = SVG(870, 336)
    y = 6
    segs = [("blocks 1–2 BF16", "neu", 168), ("blocks 3–54 NVFP4", "acc", 486), ("last 8 blocks BF16", "neu", 188)]
    x = 14
    for lab, kind, w in segs:
        box(s, x, y, w, 40, lab, kind, size=16)
        x += w + 4
    s.text(250, 82, "NVFP4 operands", size=18, weight="bold", fill=ACC)
    s.badge(340, 76, 1)
    s.text(607, 82, "GEMM", size=18, weight="bold", fill=ACC)
    s.text(795, 82, "output", size=18, weight="bold", fill=GRAY)
    rows = [
        ("Fprop", ("X", "1×16 · RNE", []), ("W", "16×16 · RNE", [2]), "Y BF16"),
        ("Dgrad", ("dY", "1×16 · SR", [4]), ("Wᵀ", "16×16 · RNE", [2]), "dX BF16"),
        ("Wgrad", ("Xᵀ", "RHT · 1×16 · RNE", [3]), ("dYᵀ", "RHT · 1×16 · SR", [3, 4]), "dW FP32"),
    ]
    y = 98
    for name, a, b, o in rows:
        s.text(80, y + 34, name, size=19, anchor="end", weight="bold")
        for (nm, tag, bdg), x in ((a, 90), (b, 290)):
            s.rect(x, y, 172, 60, fill=ACC_F, stroke=ACC, sw=2)
            s.ctext(x + 86, y + 20, nm, size=19, weight="bold", fill=ACC)
            s.ctext(x + 86, y + 44, tag, size=15, fill=INK)
            for k, n in enumerate(bdg):
                s.badge(x + 172, y + 4 + k * 28, n, r=12, size=15)
        s.ctext(276, y + 30, "×", size=22, weight="bold")
        s.line(478, y + 30, 520, y + 30, color=ACC, sw=2.4)
        s.rect(524, y, 166, 60, fill=ACC_F, stroke=ACC, sw=2)
        s.lines(607, y + 30, ["E2M1 × E2M1", ("FP32 accumulate", {"size": 15, "fill": INK})], size=17,
                weight="bold", fill=ACC, step=21)
        s.line(692, y + 30, 726, y + 30, sw=2.4)
        s.rect(730, y, 130, 60, fill=NEU_F, stroke=NEU, sw=2)
        s.ctext(795, y + 30, o, size=17, weight="bold")
        y += 76
    s.save(out("fig_overview.svg"))


# ---------------------------------------------------------------- fig_twolevel: encode and decode with two scales
def fig_twolevel():
    s = SVG(870, 290)
    s.text(14, 24, "encode (quantize)", size=18, anchor="start", weight="bold")
    y = 38
    bx = [(14, 150, "tensor amax", "extra pass", "neu"),
          (190, 200, "s_tensor in FP32", "amax ÷ (6 × 448)", "neu"),
          (416, 216, "s_block in E4M3", "block amax ÷ 6 ÷ s_tensor", "acc"),
          (658, 200, "q in E2M1", "x ÷ (s_block × s_tensor)", "acc")]
    for i, (x, w, a, b, k) in enumerate(bx):
        box(s, x, y, w, 64, a, k, sub=b, size=17, subsize=15)
        if i < len(bx) - 1:
            nx = bx[i + 1][0]
            s.line(x + w + 2, y + 32, nx - 3, y + 32, sw=2.2)
    s.text(14, 160, "decode (inside the GEMM)", size=18, anchor="start", weight="bold", fill=ACC)
    y = 174
    dx = [(14, 190, "Tensor Core", "16-value dot product", "acc"),
          (230, 200, "× both s_block", "per block in the MMA", "acc"),
          (456, 180, "FP32 accumulate", "over all of K", "neu"),
          (662, 196, "× both s_tensor", "once after the GEMM", "neu")]
    for i, (x, w, a, b, k) in enumerate(dx):
        box(s, x, y, w, 64, a, k, sub=b, size=17, subsize=15)
        if i < len(dx) - 1:
            nx = dx[i + 1][0]
            s.line(x + w + 2, y + 32, nx - 3, y + 32, color=ACC if i == 0 else INK, sw=2.2)
    s.save(out("fig_twolevel.svg"))


# ---------------------------------------------------------------- fig_2d: 1x16 vs 16x16 weight scales
def mini_grid(s, x, y, n, c, mode, color):
    h = n // 2
    if mode == "row":
        for i in range(n):
            for t in range(2):
                if (i + t) % 2 == 0:
                    s.rect(x + t * h * c, y + i * c, h * c, c, fill=color[1], stroke="none", sw=0, rx=0)
    elif mode == "col":
        for i in range(n):
            for t in range(2):
                if (i + t) % 2 == 0:
                    s.rect(x + i * c, y + t * h * c, c, h * c, fill=color[1], stroke="none", sw=0, rx=0)
    else:
        for bi in range(2):
            for bj in range(2):
                if (bi + bj) % 2 == 0:
                    s.rect(x + bj * h * c, y + bi * h * c, h * c, h * c, fill=color[1], stroke="none", sw=0, rx=0)
    for i in range(n):
        for j in range(n):
            s.rect(x + j * c, y + i * c, c, c, fill="none", stroke="#C9C9C9", sw=0.8, rx=0)
    if mode == "row":
        for i in range(n):
            for t in range(2):
                s.rect(x + t * h * c, y + i * c, h * c, c, fill="none", stroke=color[0], sw=1.6, rx=0)
    elif mode == "col":
        for i in range(n):
            for t in range(2):
                s.rect(x + i * c, y + t * h * c, c, h * c, fill="none", stroke=color[0], sw=1.6, rx=0)
    else:
        for bi in range(2):
            for bj in range(2):
                s.rect(x + bj * h * c, y + bi * h * c, h * c, h * c, fill="none", stroke=color[0], sw=2.2, rx=0)


def fig_2d():
    s = SVG(870, 290)
    n, c = 8, 15
    G = n * c
    red, acc = (PROB, PROB_F), (ACC, ACC_F)
    s.text(210, 24, "1×16 weight scales", size=20, weight="bold", fill=PROB)
    mini_grid(s, 30, 70, n, c, "row", red)
    mini_grid(s, 270, 70, n, c, "col", red)
    s.text(30 + G / 2, 60, "W in Fprop", size=17, weight="bold")
    s.text(270 + G / 2, 60, "W in Dgrad", size=17, weight="bold")
    s.ctext(210, 70 + G / 2, "≠", size=40, weight="bold", fill=PROB)
    s.text(210, 232, "two different quantized W", size=17, weight="bold", fill=PROB)
    s.text(210, 256, "backward differentiates another W", size=16, fill=PROB)
    s.line(435, 14, 435, 280, color="#BFBFBF", sw=1.5, arrow=False, dash="6,5")
    s.text(655, 24, "16×16 weight scales", size=20, weight="bold", fill=ACC)
    mini_grid(s, 475, 70, n, c, "blk", acc)
    mini_grid(s, 715, 70, n, c, "blk", acc)
    s.text(475 + G / 2, 60, "W in Fprop", size=17, weight="bold")
    s.text(715 + G / 2, 60, "W in Dgrad", size=17, weight="bold")
    s.ctext(655, 70 + G / 2, "=", size=40, weight="bold", fill=ACC)
    s.text(655, 232, "one quantized W", size=17, weight="bold", fill=ACC)
    s.text(655, 256, "activations and gradients keep 1×16", size=16, fill=GRAY)
    s.save(out("fig_2d.svg"))


# ---------------------------------------------------------------- fig_rht: Hadamard rotation on Wgrad inputs
def fig_rht():
    s = SVG(870, 300)
    lanes = [("X BF16", "1×16 · RNE", 30), ("dY BF16", "1×16 · SR", 120)]
    for lab, q, y in lanes:
        box(s, 14, y, 110, 56, lab, "neu", size=17)
        box(s, 160, y, 128, 56, "× H", "acc", sub="16×16 random", size=18, subsize=15)
        box(s, 324, y, 150, 56, "NVFP4", "acc", sub=q, size=17, subsize=15)
        s.line(126, y + 28, 156, y + 28, sw=2.2)
        s.line(290, y + 28, 320, y + 28, color=ACC, sw=2.2)
        s.path(f"M476,{y + 28} H520", color=ACC, sw=2.2, arrow=False)
    s.path("M520,58 V148", color=ACC, sw=2.2, arrow=False)
    s.line(520, 103, 556, 103, color=ACC, sw=2.2)
    box(s, 560, 70, 156, 66, "Wgrad GEMM", "acc", sub="H · Hᵀ = I cancels", size=17, subsize=15)
    s.line(718, 103, 744, 103, sw=2.2)
    box(s, 748, 76, 110, 54, "dW FP32", "neu", size=17)
    # inset: one block of 16 values before and after rotation
    s.text(14, 214, "one block of 16", size=16, anchor="start", weight="bold", fill=GRAY)
    s.text(14, 234, "magnitudes", size=16, anchor="start", weight="bold", fill=GRAY)
    before = [0.10, 0.06, 0.12, 0.08, 0.05, 0.09, 1.00, 0.07, 0.11, 0.04, 0.10, 0.06, 0.08, 0.12, 0.05, 0.09]
    after = [0.30, 0.22, 0.34, 0.27, 0.25, 0.31, 0.29, 0.20, 0.33, 0.26, 0.28, 0.24, 0.32, 0.21, 0.30, 0.27]
    for k, (vals, x0, lab, col) in enumerate(((before, 170, "before RHT: one outlier", PROB),
                                             (after, 520, "after RHT: spread out", ACC))):
        base = 284
        for i, v in enumerate(vals):
            h = v * 70
            c = PROB if (k == 0 and v == 1.0) else (NEU if k == 0 else ACC)
            s.rect(x0 + i * 16, base - h, 11, h, fill=c, stroke=c, sw=0.5, rx=0)
        s.line(x0 - 4, base, x0 + 16 * 16, base, color=INK, sw=1.2, arrow=False)
        s.text(x0 + 128, 202, lab, size=16, fill=col, weight="bold")
    s.save(out("fig_rht.svg"))


# ---------------------------------------------------------------- fig_sr: stochastic rounding on gradients
def fig_sr():
    def X(v):
        return 196 + (v - 0.5) * 300

    s = SVG(870, 250)
    rows = [("round to nearest", "weights and activations", 56), ("stochastic", "gradients", 140)]
    for name, sub, y in rows:
        s.text(176, y, name, size=18, anchor="end", weight="bold", fill=INK if y == 56 else ACC)
        s.text(176, y + 22, sub, size=15, anchor="end", fill=GRAY)
    x12 = X(1.2)
    # RNE row
    s.path(f"M{x12},50 Q{(x12 + X(1)) / 2},18 {X(1) + 4},44", color=PROB, sw=2.4)
    s.text((x12 + X(1)) / 2 - 4, 22, "always", size=16, fill=PROB, weight="bold")
    s.text(X(2) + 30, 64, "bias: E[q] = 1", size=17, anchor="start", fill=PROB, weight="bold")
    # SR row
    s.path(f"M{x12},134 Q{(x12 + X(1)) / 2},102 {X(1) + 4},128", color=ACC, sw=2.4)
    s.path(f"M{x12},134 Q{(x12 + X(1.5)) / 2},96 {X(1.5) - 4},128", color=ACC, sw=2.4)
    s.text((x12 + X(1)) / 2 - 4, 106, "60%", size=16, fill=ACC, weight="bold")
    s.text((x12 + X(1.5)) / 2 + 6, 100, "40%", size=16, fill=ACC, weight="bold")
    s.text(X(2) + 30, 148, "unbiased: E[q] = 1.2", size=17, anchor="start", fill=ACC, weight="bold")
    for y in (56, 140):
        s.line(X(0.5), y, X(2), y, color="#BFBFBF", sw=1.5, arrow=False)
        for v in (0.5, 1, 1.5, 2):
            s.raw(f'<circle cx="{X(v)}" cy="{y}" r="8" fill="{NEU}"/>')
        s.raw(f'<rect x="{x12 - 7}" y="{y - 7}" width="14" height="14" fill="{INK}" transform="rotate(45 {x12} {y})"/>')
    ay = 206
    s.line(X(0.5) - 10, ay, X(2) + 20, ay, color=INK, sw=2)
    for v in (0.5, 1, 1.5, 2):
        s.line(X(v), ay - 6, X(v), ay + 6, color=INK, sw=2, arrow=False)
        s.text(X(v), ay + 28, f"{v:g}", size=17)
    s.raw(f'<rect x="{x12 - 6}" y="{ay - 6}" width="12" height="12" fill="{INK}" transform="rotate(45 {x12} {ay})"/>')
    s.text(x12, ay + 28, "x = 1.2", size=17, weight="bold")
    s.text(176, ay + 6, "E2M1", size=17, anchor="end", weight="bold")
    s.save(out("fig_sr.svg"))


if __name__ == "__main__":
    fig_block()
    fig_samples(1)
    fig_samples(2)
    fig_overview()
    fig_twolevel()
    fig_2d()
    fig_rht()
    fig_sr()
