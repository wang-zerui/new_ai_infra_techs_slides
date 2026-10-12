"""Generate the SVG diagrams for deck 46 (FP8 training on Hopper). Run: python3 gen_figs.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PROB, PROB_F, NEU, NEU_F, GRAY, INK  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(OUT, name)


# ---------------------------------------------------------------- fig_formats: bit layouts
def fig_formats():
    s = SVG(870, 300)
    cw, ch = 26, 44
    x0 = 200
    # legend
    lx = x0
    for lab, fill, stroke in (("sign", NEU_F, NEU), ("exponent", ACC_F, ACC), ("mantissa", "#FFFFFF", NEU)):
        s.rect(lx, 12, 22, 22, fill=fill, stroke=stroke, sw=2, rx=2)
        s.text(lx + 30, 30, lab, size=18, anchor="start")
        lx += 30 + len(lab) * 9.5 + 34
    rows = [
        ("BF16", "1 + 8 + 7 bits", 1, 8, 7, ["FP32 exponent range", ("baseline precision", True)]),
        ("FP8 E4M3", "1 + 4 + 3 bits", 1, 4, 3, ["max ±448 · 18 binades", ("weights and activations", True)]),
        ("FP8 E5M2", "1 + 5 + 2 bits", 1, 5, 2, ["max ±57,344 · 32 binades", ("gradients (hybrid)", True)]),
    ]
    y = 60
    for name, sub, sb, eb, mb, info in rows:
        s.text(x0 - 16, y + 20, name, size=20, anchor="end", weight="bold")
        s.text(x0 - 16, y + 42, sub, size=16, anchor="end", fill=GRAY)
        x = x0
        for i in range(sb + eb + mb):
            if i < sb:
                f, st = NEU_F, NEU
            elif i < sb + eb:
                f, st = ACC_F, ACC
            else:
                f, st = "#FFFFFF", NEU
            s.rect(x, y, cw, ch, fill=f, stroke=st, sw=2, rx=0)
            x += cw
        if sb + eb + mb == 8:  # ghost of the 8 bits FP8 does not carry
            s.rect(x + 4, y, 8 * cw - 4, ch, fill="none", stroke="#BFBFBF", sw=1.5, rx=0, dash="5,4")
        ix = x0 + 16 * cw + 30
        s.text(ix, y + 18, info[0], size=18, anchor="start")
        s.text(ix, y + 42, info[1][0], size=16, anchor="start", fill=GRAY, italic=True)
        y += 82
    s.save(out("fig_formats.svg"))


# ---------------------------------------------------------------- fig_outlier: per-tensor vs per-tile
def fig_outlier(step):
    rows = [
        ("BF16 input", "",
         ["0.012", "−0.004", "0.021", "0.0007", "1.5", "−0.9", "2400", "0.6"], None),
        ("per-tensor E4M3", "1 scale = 448 / 2400",
         ["0.0105", "0", "0.0209", "0", "1.51", "−0.92", "2400", "0.59"], "tensor"),
        ("per-tile E4M3", "1 scale per tile",
         ["0.012", "−0.0041", "0.021", "0.0007", "1.51", "−0.92", "2400", "0.59"], "tile"),
    ]
    nrows = 2 if step == 1 else 3
    H = 44 + nrows * 74 + 4
    s = SVG(870, H)
    x0, cw, gap, tgap = 222, 74, 2, 14
    xs = []
    x = x0
    for i in range(8):
        if i == 4:
            x += tgap
        xs.append(x)
        x += cw + gap
    # tile headers (dashed group brackets)
    for t, (a, b) in enumerate(((0, 3), (4, 7))):
        xa, xb = xs[a] - 4, xs[b] + cw + 4
        s.rect(xa, 10, xb - xa, 26, fill="none", stroke="#A6A6A6", sw=1.5, rx=3, dash="4,3")
        s.ctext((xa + xb) / 2, 23, f"tile {t + 1}  (128 values, 4 shown)", size=16, fill=GRAY)
    y = 46
    for r, (lab, sub, vals, kind) in enumerate(rows[:nrows]):
        s.text(x0 - 14, y + 26 if sub else y + 34, lab, size=19, anchor="end", weight="bold")
        if sub:
            s.text(x0 - 14, y + 50, sub, size=16, anchor="end", fill=GRAY)
        for i, v in enumerate(vals):
            fill, stroke, col, wt = "#FFFFFF", NEU, INK, "normal"
            if kind is None and v == "2400":
                stroke, col, wt = PROB, PROB, "bold"
            if kind == "tensor" and v == "0":
                fill, stroke, col, wt = PROB_F, PROB, PROB, "bold"
            if kind == "tile" and i < 4:
                fill, stroke, col, wt = ACC_F, ACC, ACC, "bold"
            s.rect(xs[i], y, cw, 62, fill=fill, stroke=stroke, sw=2.2 if stroke != NEU else 1.6, rx=3)
            s.ctext(xs[i] + cw / 2, y + 31, v, size=18, fill=col, weight=wt)
        y += 74
    if step == 1:
        s.save(out("fig_outlier_1.svg"))
    else:
        s.save(out("fig_outlier_2.svg"))


# ---------------------------------------------------------------- fig_overview: three FP8 GEMMs per linear layer
def fig_overview():
    s = SVG(870, 336)
    # column headers
    s.text(250, 22, "FP8 operands (scale block)", size=18, weight="bold", fill=ACC)
    s.badge(398, 16, 1)
    s.text(616, 22, "GEMM", size=18, weight="bold", fill=ACC)
    s.badge(660, 16, 2)
    s.text(790, 22, "output", size=18, weight="bold", fill=GRAY)
    rows = [
        ("Fprop", ("X", "1×128"), ("W", "128×128"), ("Y", "BF16")),
        ("Dgrad", ("dY", "1×128"), ("Wᵀ", "128×128"), ("dX", "BF16")),
        ("Wgrad", ("dYᵀ", "128×1"), ("Xᵀ", "128×1"), ("dW", "FP32")),
    ]
    y = 36
    for name, a, b, o in rows:
        s.text(76, y + 34, name, size=20, anchor="end", weight="bold")
        for (nm, blk), x in ((a, 92), (b, 268)):
            s.rect(x, y, 150, 52, fill=ACC_F, stroke=ACC, sw=2)
            s.ctext(x + 38, y + 26, nm, size=20, weight="bold", fill=ACC)
            s.ctext(x + 104, y + 26, blk, size=18, fill=INK)
        s.ctext(255, y + 26, "×", size=22, weight="bold")
        s.line(420, y + 26, 528, y + 26, color=ACC, sw=2.5)
        s.rect(532, y, 168, 52, fill=ACC_F, stroke=ACC, sw=2)
        s.lines(616, y + 26, ["E4M3 × E4M3", ("FP32 accumulate", {"size": 16, "fill": INK})], size=18,
                weight="bold", fill=ACC, step=21)
        s.line(702, y + 26, 736, y + 26, color=INK, sw=2.5)
        s.rect(740, y, 120, 52, fill=NEU_F, stroke=NEU, sw=2)
        s.ctext(800, y + 26, f"{o[0]} {o[1]}", size=18, weight="bold")
        y += 66
    # badge 3: cached activation and FP8 weight all-gather
    s.badge(418, 36 + 66 * 2 + 4, 3)
    s.badge(418, 36 + 4, 3)
    # bottom strip: kept in high precision
    s.rect(10, 248, 850, 80, fill="#FFFFFF", stroke="#A6A6A6", sw=1.5, rx=5, dash="5,4")
    s.text(24, 272, "kept in BF16 / FP32", size=17, anchor="start", weight="bold", fill=GRAY)
    chips = ["Embedding", "LM head", "Router FP32", "Norms", "Attention", "Optimizer FP32"]
    widths = [128, 104, 136, 92, 112, 160]
    x = 24
    for c, w in zip(chips, widths):
        s.rect(x, 282, w, 36, fill=NEU_F, stroke=NEU, sw=1.6)
        s.ctext(x + w / 2, 300, c, size=17)
        x += w + 12
    s.save(out("fig_overview.svg"))


# ---------------------------------------------------------------- fig_granularity: scale blocks
def grid(s, x, y, n, c, fill="#FFFFFF", stroke="#BFBFBF"):
    for i in range(n):
        for j in range(n):
            s.rect(x + j * c, y + i * c, c, c, fill=fill, stroke=stroke, sw=1, rx=0)


def fig_granularity():
    s = SVG(870, 300)
    n, c = 8, 22
    G = n * c
    panels = [("Per-tensor", ["1 FP32 scale", "delayed or current amax"]),
              ("1×128 tiles", ["1 scale per tile", "activations and gradients"]),
              ("128×128 blocks", ["1 scale per block", "weights"])]
    for p, (title, cap) in enumerate(panels):
        px = 20 + p * 290
        gx, gy = px + 30, 50
        new = p > 0
        s.text(gx + G / 2 + 14, 30, title, size=21, weight="bold", fill=ACC if new else INK)
        grid(s, gx, gy, n, c)
        sc_x = gx + G + 14
        if p == 0:
            s.rect(gx, gy, G, G, fill="none", stroke=NEU, sw=2.5, rx=0)
            s.rect(sc_x, gy + G / 2 - 12, 24, 24, fill=NEU_F, stroke=NEU, sw=2, rx=3)
            s.ctext(sc_x + 12, gy + G / 2, "s", size=17, weight="bold")
        elif p == 1:
            half = n // 2
            for i in range(n):
                for t in range(2):
                    if (i + t) % 2 == 0:
                        s.rect(gx + t * half * c, gy + i * c, half * c, c, fill=ACC_F, stroke="none", sw=0, rx=0)
            grid(s, gx, gy, n, c, fill="none")
            for i in range(n):
                for t in range(2):
                    s.rect(gx + t * half * c, gy + i * c, half * c, c, fill="none", stroke=ACC, sw=1.8, rx=0)
                for t in range(2):
                    s.rect(sc_x + t * 16, gy + i * c + 4, 13, c - 8, fill=ACC, stroke=ACC, sw=1, rx=2)
        else:
            half = n // 2
            for bi in range(2):
                for bj in range(2):
                    if (bi + bj) % 2 == 0:
                        s.rect(gx + bj * half * c, gy + bi * half * c, half * c, half * c, fill=ACC_F, stroke="none",
                               sw=0, rx=0)
            grid(s, gx, gy, n, c, fill="none")
            for bi in range(2):
                for bj in range(2):
                    s.rect(gx + bj * half * c, gy + bi * half * c, half * c, half * c, fill="none", stroke=ACC,
                           sw=2.4, rx=0)
            for bi in range(2):
                for bj in range(2):
                    s.rect(sc_x + bj * 18, gy + bi * half * c + half * c / 2 - 8, 15, 16, fill=ACC, stroke=ACC, sw=1, rx=2)
        s.text(gx + G / 2 + 14, gy + G + 36, cap[0], size=18, weight="bold", fill=ACC if new else INK)
        s.text(gx + G / 2 + 14, gy + G + 60, cap[1], size=17, fill=GRAY)
    s.save(out("fig_granularity.svg"))


# ---------------------------------------------------------------- fig_transpose: 1D tiles need two quantizations
def tiled(s, x, y, n, c, horizontal, half):
    """n x n grid of cells grouped in 1 x half (horizontal) or half x 1 (vertical) tiles."""
    for i in range(n):
        for j in range(n):
            s.rect(x + j * c, y + i * c, c, c, fill="#FFFFFF", stroke="#C9C9C9", sw=0.8, rx=0)
    for i in range(n):
        for t in range(n // half):
            if horizontal:
                s.rect(x + t * half * c, y + i * c, half * c, c, fill=ACC_F if (i + t) % 2 == 0 else "none",
                       stroke=ACC, sw=1.6, rx=0, opacity=0.9)
            else:
                s.rect(x + i * c, y + t * half * c, c, half * c, fill=ACC_F if (i + t) % 2 == 0 else "none",
                       stroke=ACC, sw=1.6, rx=0, opacity=0.9)


def fig_transpose():
    s = SVG(870, 300)
    n, c = 8, 14
    G = n * c
    # left panel: 1D tiles
    s.text(280, 24, "1×128 tiles (X and dY)", size=20, weight="bold", fill=ACC)
    s.rect(14, 128, 124, 72, fill=NEU_F, stroke=NEU, sw=2)
    s.lines(76, 164, ["BF16", "tensor"], size=18, weight="bold")
    rx, ry, cy = 250, 44, 176
    tiled(s, rx, ry, n, c, True, 4)
    tiled(s, rx, cy, n, c, False, 4)
    s.path(f"M140,150 C190,150 200,{ry + G / 2} {rx - 4},{ry + G / 2}", color=ACC, sw=2.5)
    s.path(f"M140,178 C190,178 200,{cy + G / 2} {rx - 4},{cy + G / 2}", color=ACC, sw=2.5)
    s.text(150, 82, "quantize", size=16, fill=ACC, anchor="start")
    s.text(112, 262, "quantize again", size=16, fill=ACC, anchor="start")
    s.text(rx + G + 16, ry + G / 2 - 2, "rowwise", size=18, anchor="start", weight="bold")
    s.text(rx + G + 16, ry + G / 2 + 20, "Fprop and Dgrad", size=16, anchor="start", fill=GRAY)
    s.text(rx + G + 16, cy + G / 2 - 2, "columnwise", size=18, anchor="start", weight="bold")
    s.text(rx + G + 16, cy + G / 2 + 20, "Wgrad", size=16, anchor="start", fill=GRAY)
    # divider
    s.line(560, 14, 560, 290, color="#BFBFBF", sw=1.5, arrow=False, dash="6,5")
    # right panel: 2D blocks
    s.text(718, 24, "128×128 blocks (W)", size=20, weight="bold", fill=ACC)
    by = 110
    xs = (586, 586 + G + 50)
    for x in xs:
        h = n // 2
        for bi in range(2):
            for bj in range(2):
                if (bi + bj) % 2 == 0:
                    s.rect(x + bj * h * c, by + bi * h * c, h * c, h * c, fill=ACC_F, stroke="none", sw=0, rx=0)
        grid(s, x, by, n, c, fill="none", stroke="#C9C9C9")
        for bi in range(2):
            for bj in range(2):
                s.rect(x + bj * h * c, by + bi * h * c, h * c, h * c, fill="none", stroke=ACC, sw=2.2, rx=0)
    s.line(xs[0] + G + 4, by + G / 2, xs[1] - 4, by + G / 2, color=ACC, sw=2.5)
    s.text((xs[0] + G + xs[1]) / 2, by - 14, "transpose", size=16, fill=ACC)
    s.text(xs[0] + G / 2, by + G + 30, "rowwise", size=18, weight="bold")
    s.text(xs[1] + G / 2, by + G + 30, "columnwise", size=18, weight="bold")
    s.text((xs[0] + xs[1] + G) / 2, by + G + 56, "same blocks and scales", size=16, fill=GRAY)
    s.save(out("fig_transpose.svg"))


# ---------------------------------------------------------------- fig_promotion: FP32 promotion every 128 of K
def fig_promotion():
    s = SVG(870, 270)
    s.text(150, 82, "Tensor Core", size=19, anchor="end", weight="bold")
    s.text(150, 104, "limited-precision acc", size=15, anchor="end", fill=GRAY)
    s.text(150, 192, "CUDA Core", size=19, anchor="end", weight="bold")
    s.text(150, 214, "FP32 registers", size=15, anchor="end", fill=GRAY)
    x0, chunk, gapc = 168, 160, 12
    for k in range(4):
        cx = x0 + k * (chunk + gapc)
        s.rect(cx, 54, chunk, 62, fill="#FFFFFF", stroke=ACC, sw=2, rx=4)
        for w in range(4):
            s.rect(cx + 8 + w * 37, 64, 32, 42, fill=ACC_F, stroke=ACC, sw=1.4, rx=2)
        s.text(cx + chunk / 2, 40, "4 WGMMAs", size=16, fill=GRAY)
        # promotion arrow
        ax = cx + chunk - 22
        s.line(ax, 118, ax, 166, color=ACC, sw=2.5)
        s.text(ax - 8, 148, "× sX·sW", size=16, anchor="end", fill=ACC, weight="bold", halo=True)
    s.rect(168, 170, 4 * (chunk + gapc) - gapc, 50, fill=ACC_F, stroke=ACC, sw=2.2, rx=4)
    s.ctext(168 + (4 * (chunk + gapc) - gapc) / 2, 195, "FP32 accumulator  +=  dequantized partial sum", size=18,
            weight="bold", fill=ACC)
    # K axis
    s.line(168, 248, 862, 248, color=INK, sw=2)
    for k in range(4):
        tx = x0 + k * (chunk + gapc)
        s.line(tx, 242, tx, 254, color=INK, sw=2, arrow=False)
    s.text(150, 254, "K", size=18, anchor="end", weight="bold")
    for k in range(4):
        cx = x0 + k * (chunk + gapc)
        s.text(cx + chunk / 2, 240, "128 of K", size=15, fill=GRAY)
    s.save(out("fig_promotion.svg"))


# ---------------------------------------------------------------- fig_storage: FP8 storage only vs storage + compute
def fig_storage():
    s = SVG(870, 300)
    s.line(470, 44, 470, 292, color="#BFBFBF", sw=1.5, arrow=False, dash="6,5")
    s.text(250, 22, "forward", size=18, weight="bold", fill=GRAY)
    s.text(680, 22, "backward", size=18, weight="bold", fill=GRAY)

    def box(x, y, w, lab, kind, h=48):
        f, st, col = {"acc": (ACC_F, ACC, ACC), "neu": (NEU_F, NEU, INK)}[kind]
        s.rect(x, y, w, h, fill=f, stroke=st, sw=2)
        s.ctext(x + w / 2, y + h / 2, lab, size=17, weight="bold", fill=col)

    # lane A: Kimi K2
    s.text(14, 62, "Kimi K2: FP8 storage, BF16 compute", size=18, anchor="start", weight="bold")
    ya = 78
    box(14, ya, 92, "X BF16", "neu")
    box(146, ya, 120, "BF16 GEMM", "neu")
    box(300, ya, 140, "1×128 E4M3", "acc")
    s.line(108, ya + 24, 142, ya + 24, sw=2.2)
    s.path(f"M60,{ya + 50} V{ya + 70} H370 V{ya + 52}", color=ACC, sw=2.2)
    box(500, ya, 132, "dequantize", "acc")
    s.line(442, ya + 24, 496, ya + 24, color=ACC, sw=2.2, dash="7,5")
    box(666, ya, 190, "BF16 Wgrad", "neu")
    s.line(634, ya + 24, 662, ya + 24, sw=2.2)
    # lane B: DeepSeek-V3 / TE
    s.text(14, 200, "DeepSeek-V3 / TE: FP8 storage and FP8 GEMMs", size=18, anchor="start", weight="bold")
    yb = 216
    box(14, yb, 92, "X BF16", "neu")
    box(146, yb, 120, "1×128 E4M3", "acc")
    box(300, yb, 140, "FP8 Fprop", "acc")
    s.line(108, yb + 24, 142, yb + 24, sw=2.2)
    s.line(268, yb + 24, 296, yb + 24, color=ACC, sw=2.2)
    box(500, yb, 132, "cached FP8", "acc")
    s.path(f"M206,{yb + 50} V{yb + 66} H566 V{yb + 52}", color=ACC, sw=2.2, dash="7,5")
    box(666, yb, 190, "FP8 Wgrad", "acc")
    s.line(634, yb + 24, 662, yb + 24, color=ACC, sw=2.2)
    s.save(out("fig_storage.svg"))


# ---------------------------------------------------------------- fig_paramag: FP8 primary weights
def fig_paramag():
    s = SVG(870, 250)

    def box(x, y, w, lab, kind, h=54, sub=None):
        f, st, col = {"acc": (ACC_F, ACC, ACC), "neu": (NEU_F, NEU, INK), "prob": (PROB_F, PROB, PROB)}[kind]
        s.rect(x, y, w, h, fill=f, stroke=st, sw=2)
        if sub:
            s.lines(x + w / 2, y + h / 2, [lab, (sub, {"size": 15, "weight": "normal"})], size=17, weight="bold",
                    fill=col, step=20)
        else:
            s.ctext(x + w / 2, y + h / 2, lab, size=17, weight="bold", fill=col)

    s.text(14, 26, "BF16 path", size=19, anchor="start", weight="bold", fill=GRAY)
    y1 = 40
    box(14, y1, 130, "FP32 master", "neu", sub="shard")
    box(180, y1, 120, "BF16 copy", "prob", sub="shard")
    box(470, y1, 120, "BF16 weight", "neu", sub="full")
    box(626, y1, 104, "quantize", "neu")
    box(766, y1, 92, "FP8 W", "acc")
    s.line(146, y1 + 27, 176, y1 + 27, sw=2.2)
    s.line(302, y1 + 27, 466, y1 + 27, color=PROB, sw=3)
    s.text(384, y1 + 18, "all-gather", size=16, fill=PROB, weight="bold")
    s.text(384, y1 + 48, "2 B / param", size=16, fill=PROB)
    s.line(592, y1 + 27, 622, y1 + 27, sw=2.2)
    s.line(732, y1 + 27, 762, y1 + 27, sw=2.2)

    s.text(14, 148, "FP8 primary weights", size=19, anchor="start", weight="bold", fill=ACC)
    y2 = 162
    box(14, y2, 130, "FP32 master", "neu", sub="shard")
    box(180, y2, 120, "quantize", "acc", sub="direct cast")
    box(336, y2, 100, "FP8", "acc", sub="shard")
    box(626, y2, 232, "FP8 W", "acc", sub="full")
    s.line(146, y2 + 27, 176, y2 + 27, color=ACC, sw=2.2)
    s.line(302, y2 + 27, 332, y2 + 27, color=ACC, sw=2.2)
    s.line(438, y2 + 27, 622, y2 + 27, color=ACC, sw=3)
    s.text(530, y2 + 18, "all-gather", size=16, fill=ACC, weight="bold")
    s.text(530, y2 + 48, "1 B / param", size=16, fill=ACC)
    s.save(out("fig_paramag.svg"))


if __name__ == "__main__":
    fig_formats()
    fig_outlier(1)
    fig_outlier(2)
    fig_overview()
    fig_granularity()
    fig_transpose()
    fig_promotion()
    fig_storage()
    fig_paramag()
