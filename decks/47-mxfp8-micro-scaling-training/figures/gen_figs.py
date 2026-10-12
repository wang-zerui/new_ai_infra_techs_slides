"""Generate the SVG diagrams for deck 47 (MXFP8 micro-scaling training). Run: python3 gen_figs.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import SVG, ACC, ACC_F, PROB, PROB_F, NEU, NEU_F, GRAY, INK  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(OUT, name)


def box(s, x, y, w, h, lab, kind="neu", sub=None, size=18, subsize=15):
    f, st, col = {"acc": (ACC_F, ACC, ACC), "neu": (NEU_F, NEU, INK), "prob": (PROB_F, PROB, PROB),
                  "white": ("#FFFFFF", NEU, INK)}[kind]
    s.rect(x, y, w, h, fill=f, stroke=st, sw=2)
    if sub:
        s.lines(x + w / 2, y + h / 2, [lab, (sub, {"size": subsize, "weight": "normal"})], size=size,
                weight="bold", fill=col, step=size + 3)
    else:
        s.ctext(x + w / 2, y + h / 2, lab, size=size, weight="bold", fill=col)


def bits(s, x, y, layout, cw=22, ch=30, hl=False):
    """layout: string of 's','e','m' per bit."""
    for i, b in enumerate(layout):
        f, st = {"s": (NEU_F, NEU), "e": (ACC_F, ACC), "m": ("#FFFFFF", NEU)}[b]
        s.rect(x + i * cw, y, cw, ch, fill=f, stroke=st, sw=1.6, rx=0)
    if hl:
        s.rect(x - 4, y - 4, len(layout) * cw + 8, ch + 8, fill="none", stroke=ACC, sw=2.6, rx=4)


# ---------------------------------------------------------------- fig_ladder: scale granularity ladder
def fig_ladder():
    s = SVG(870, 290)
    x0, cw, n = 252, 29, 16
    s.text(x0 + n * cw / 2, 20, "128 consecutive values (8 per cell)", size=16, fill=GRAY)
    s.text(800, 20, "scale applied", size=16, fill=GRAY)
    rows = [
        ("FP8 per-tensor", "FP32 scale per tensor", None, "neu", ["once per", "GEMM"]),
        ("FP8 blockwise", "FP32 scale per 128", 16, "neu", ["CUDA cores", "(Hopper)"]),
        ("MXFP8", "E8M0 scale per 32", 4, "acc", ["tensor core", "(Blackwell)"]),
        ("NVFP4 (deck 48)", "E4M3 scale per 16", 2, "ghost", ["tensor core", "(Blackwell)"]),
    ]
    y = 40
    for name, sub, g, kind, where in rows:
        col = ACC if kind == "acc" else (GRAY if kind == "ghost" else INK)
        s.text(x0 - 16, y + 34, name, size=19, anchor="end", weight="bold", fill=col)
        s.text(x0 - 16, y + 56, sub, size=16, anchor="end", fill=GRAY)
        cy = y + 30
        for i in range(n):
            s.rect(x0 + i * cw, cy, cw, 30, fill="#FFFFFF", stroke="#BFBFBF", sw=1, rx=0)
        if g is None:
            s.rect(x0, cy, n * cw, 30, fill="none", stroke=NEU, sw=2.2, rx=0, dash="6,4")
            s.rect(x0 - 2, y + 6, 18, 16, fill=NEU_F, stroke=NEU, sw=1.6, rx=2)
            s.text(x0 + 24, y + 20, "one scale for the whole tensor", size=15, anchor="start", fill=GRAY)
        else:
            st = {"neu": NEU, "acc": ACC, "ghost": GRAY}[kind]
            fill = {"neu": NEU_F, "acc": ACC, "ghost": "#D9D9D9"}[kind]
            for k in range(n // g):
                gx = x0 + k * g * cw
                s.rect(gx, cy, g * cw, 30, fill="none", stroke=st, sw=2.4, rx=0)
                cwid = min(18, g * cw - 8)
                s.rect(gx + g * cw / 2 - cwid / 2, y + 8, cwid, 16, fill=fill, stroke=st, sw=1.4, rx=2)
        s.text(800, y + 40, where[0], size=17, weight="bold", fill=col)
        if where[1]:
            s.text(800, y + 60, where[1], size=15, fill=GRAY)
        y += 62
    s.save(out("fig_ladder.svg"))


# ---------------------------------------------------------------- fig_where: where the block scale is applied
def fig_where(step):
    H = 150 if step == 1 else 310
    s = SVG(870, H)
    # (a) Hopper software scaling
    s.text(14, 24, "Hopper: block scales applied in software", size=19, anchor="start", weight="bold")
    y = 44
    box(s, 14, y, 170, 64, "FP8 tiles", "neu", sub="+ FP32 scales")
    box(s, 236, y, 200, 64, "Tensor Core", "neu", sub="MMA over 128 of K")
    box(s, 512, y, 220, 64, "CUDA Core", "prob", sub="× sA·sB then FP32 add")
    s.line(186, y + 32, 232, y + 32, sw=2.4)
    s.line(438, y + 32, 508, y + 32, color=PROB, sw=2.6)
    s.text(473, y + 22, "promote", size=15, fill=PROB, weight="bold")
    s.path(f"M622,{y + 66} V{y + 90} H336 V{y + 70}", color=PROB, sw=2.2, dash="7,5")
    s.text(479, y + 86, "next 128 of K", size=15, fill=PROB, halo=True)
    s.text(760, y + 38, "per block", size=16, anchor="start", fill=PROB)
    s.text(760, y + 58, "dot-product", size=16, anchor="start", fill=PROB)
    if step == 2:
        s.line(14, 156, 856, 156, color="#BFBFBF", sw=1.2, arrow=False, dash="6,5")
        s.text(14, 186, "Blackwell: MXFP8 block-scaled MMA", size=19, anchor="start", weight="bold", fill=ACC)
        y = 206
        box(s, 14, y, 170, 64, "MXFP8 blocks", "acc", sub="+ E8M0 scales")
        box(s, 236, y, 300, 64, "Tensor Core", "acc", sub="applies scale per 32 in the MMA")
        box(s, 600, y, 132, 64, "FP32", "neu", sub="output")
        s.line(186, y + 32, 232, y + 32, color=ACC, sw=2.4)
        s.line(538, y + 32, 596, y + 32, color=ACC, sw=2.4)
    s.save(out(f"fig_where_{step}.svg"))


# ---------------------------------------------------------------- fig_overview: what MXFP8 quantizes
def fig_overview():
    s = SVG(870, 336)
    # model strip
    y = 8
    items = [("Embedding", "neu", 130), ("Block 1", "acc", 104), ("Block 2", "acc", 104), ("⋯", "none", 50),
             ("Block N−1", "acc", 116), ("Block N", "acc", 104), ("Output layer", "neu", 140)]
    x = 14
    pos = {}
    for lab, kind, w in items:
        if kind == "none":
            s.ctext(x + w / 2, y + 24, lab, size=22, weight="bold", fill=GRAY)
        else:
            box(s, x, y, w, 48, lab, kind, size=17)
        pos[lab] = (x, w)
        x += w + 14
    # zoom container
    bx, bw = pos["Block 2"]
    s.rect(14, 92, 842, 196, fill="#FFFFFF", stroke=ACC, sw=1.6, rx=6, dash="6,4")
    s.line(bx + 10, 58, 40, 90, color=ACC, sw=1.4, arrow=False, dash="4,3")
    s.line(bx + bw - 10, 58, 830, 90, color=ACC, sw=1.4, arrow=False, dash="4,3")
    rows = [
        [("LayerNorm", "neu"), ("QKV", "acc"), ("BMM1", "neu"), ("Softmax", "neu"), ("BMM2", "neu"),
         ("Proj", "acc"), ("Add", "neu")],
        [("LayerNorm", "neu"), ("FC1", "acc"), ("Act", "neu"), ("FC2", "acc"), ("Add", "neu")],
    ]
    ry = 112
    for r in rows:
        w, gap = 98, 18
        x = 34
        for i, (lab, kind) in enumerate(r):
            box(s, x, ry, w, 52, lab, kind, size=17)
            if i < len(r) - 1:
                s.line(x + w + 2, ry + 26, x + w + gap - 2, ry + 26, sw=2)
            x += w + gap
        ry += 84
    # legend
    s.rect(34, 302, 22, 22, fill=ACC_F, stroke=ACC, sw=2, rx=2)
    s.text(64, 319, "MXFP8 Fprop, Dgrad and Wgrad", size=17, anchor="start")
    s.rect(380, 302, 22, 22, fill=NEU_F, stroke=NEU, sw=2, rx=2)
    s.text(410, 319, "BF16", size=17, anchor="start")
    s.save(out("fig_overview.svg"))


# ---------------------------------------------------------------- fig_mxblock: one MX block
def fig_mxblock():
    s = SVG(870, 300)
    x0, cw = 40, 20
    s.text(x0 + 16 * cw, 22, "32 elements Q1 … Q32 in E4M3 (1 byte each)", size=17, weight="bold", fill=INK)
    for i in range(32):
        s.rect(x0 + i * cw, 34, cw, 40, fill="#FFFFFF", stroke=NEU, sw=1.4, rx=0)
    s.rect(x0 + 2 * cw, 34, cw, 40, fill=NEU_F, stroke=INK, sw=2.4, rx=0)
    sx = x0 + 32 * cw + 30
    s.text(sx + 70, 22, "scale X in E8M0", size=17, weight="bold", fill=ACC)
    s.rect(sx, 34, 140, 40, fill=ACC_F, stroke=ACC, sw=2.4, rx=2)
    s.ctext(sx + 70, 54, "X (1 byte)", size=17, weight="bold", fill=ACC)
    # zooms
    ex, ey = 70, 156
    s.line(x0 + 2.5 * cw, 76, ex + 88, ey - 40, color=GRAY, sw=1.4, arrow=False, dash="4,3")
    s.text(ex + 88, ey - 18, "E4M3 element", size=17, weight="bold")
    bits(s, ex, ey, "seeeemmm")
    s.text(ex + 88, ey + 56, "sign · 4 exponent · 3 mantissa", size=15, fill=GRAY)
    zx = 560
    s.line(sx + 70, 76, zx + 88, ey - 40, color=ACC, sw=1.4, arrow=False, dash="4,3")
    s.text(zx + 88, ey - 18, "E8M0 scale", size=17, weight="bold", fill=ACC)
    bits(s, zx, ey, "eeeeeeee")
    s.text(zx + 88, ey + 56, "8 exponent bits · no sign or mantissa", size=15, fill=GRAY)
    # decode formula
    s.rect(250, 238, 370, 48, fill="#FFFFFF", stroke=INK, sw=1.6, rx=6)
    s.raw('<text x="435" y="270" font-size="22" text-anchor="middle" font-weight="bold" fill="#222222">'
          'value = Q<tspan baseline-shift="sub" font-size="15">i</tspan> × 2'
          '<tspan baseline-shift="super" font-size="15">X − 127</tspan></text>')
    s.save(out("fig_mxblock.svg"))


# ---------------------------------------------------------------- fig_two_copies: rowwise and columnwise copies
def fig_two_copies():
    s = SVG(870, 300)
    s.text(286, 22, "rowwise 1×32", size=18, weight="bold", fill=ACC)
    s.text(566, 22, "columnwise 32×1", size=18, weight="bold", fill=ACC)
    rows = [("X", "activation", "Fprop", "Wgrad"), ("W", "after BF16 all-gather", "Fprop", "Dgrad"),
            ("dY", "output gradient", "Dgrad", "Wgrad")]
    y = 36
    for name, sub, rg, cg in rows:
        box(s, 14, y, 150, 64, f"{name} BF16", "neu", sub=sub, subsize=14)
        for cx, kind, g in ((196, "row", rg), (476, "col", cg)):
            s.rect(cx, y, 190, 64, fill=ACC_F, stroke=ACC, sw=2)
            if kind == "row":
                for k in range(4):
                    s.rect(cx + 12 + k * 14, y + 26, 14, 12, fill="#FFFFFF" if k % 2 else ACC, stroke=ACC, sw=1, rx=0)
            else:
                for k in range(4):
                    s.rect(cx + 30, y + 6 + k * 13, 12, 13, fill="#FFFFFF" if k % 2 else ACC, stroke=ACC, sw=1, rx=0)
            s.text(cx + 76, y + 30, f"{name} copy", size=17, anchor="start", weight="bold", fill=ACC)
            s.text(cx + 76, y + 52, f"→ {g}", size=17, anchor="start", fill=INK)
        s.path(f"M166,{y + 20} H192", color=ACC, sw=2.2)
        s.path(f"M166,{y + 46} H178 V{y + 77} H452 V{y + 46} H472", color=ACC, sw=2.2)
        y += 90
    s.text(686, 70, "quantized", size=17, anchor="start", fill=GRAY)
    s.text(686, 92, "separately", size=17, anchor="start", fill=GRAY)
    s.text(686, 114, "from BF16", size=17, anchor="start", fill=GRAY)
    s.text(686, 160, "no transpose", size=17, anchor="start", fill=GRAY)
    s.text(686, 182, "needed on", size=17, anchor="start", fill=GRAY)
    s.text(686, 204, "Blackwell", size=17, anchor="start", fill=GRAY)
    s.save(out("fig_two_copies.svg"))


# ---------------------------------------------------------------- fig_encoding: E4M3 for every tensor
def fig_encoding():
    s = SVG(870, 280)
    lx = 250
    for lab, fill, stroke in (("sign", NEU_F, NEU), ("exponent", ACC_F, ACC), ("mantissa", "#FFFFFF", NEU)):
        s.rect(lx, 6, 20, 20, fill=fill, stroke=stroke, sw=1.6, rx=2)
        s.text(lx + 28, 22, lab, size=16, anchor="start")
        lx += 28 + len(lab) * 9 + 30
    s.text(380, 62, "FP8 per-tensor (hybrid)", size=19, weight="bold")
    s.text(700, 62, "MXFP8", size=19, weight="bold", fill=ACC)
    rows = [("Weights", "seeeemmm", "E4M3", "seeeemmm", "E4M3"),
            ("Activations", "seeeemmm", "E4M3", "seeeemmm", "E4M3"),
            ("Act. gradients", "seeeeemm", "E5M2", "seeeemmm", "E4M3")]
    y = 84
    for i, (lab, a, an, b, bn) in enumerate(rows):
        s.text(200, y + 22, lab, size=19, anchor="end", weight="bold")
        bits(s, 260, y, a, cw=22, ch=32)
        s.text(450, y + 23, an, size=18, anchor="start", weight="bold", fill=INK if an == "E4M3" else NEU)
        bits(s, 600, y, b, cw=22, ch=32, hl=(i == 2))
        s.text(790, y + 23, bn, size=18, anchor="start", weight="bold", fill=ACC)
        y += 58
    s.text(380, 272, "2 mantissa bits for gradients", size=16, fill=GRAY)
    s.text(700, 272, "3 mantissa bits everywhere", size=16, fill=ACC)
    s.save(out("fig_encoding.svg"))


# ---------------------------------------------------------------- fig_rounding: round the scale up
def fig_rounding():
    s = SVG(870, 236)
    box(s, 14, 82, 150, 76, "block amax", "neu", sub="= 480")
    s.text(89, 186, "÷ 448 = 1.07", size=16, fill=GRAY)
    rows = [("round down (OCP v1)", PROB, [("X = 2⁰ = 1", "neu"), ("480 ÷ 1 = 480", "neu"),
                                           ("clamps to 448", "prob"), ("decodes to 448", "prob")]),
            ("round up (MXFP8 recipe)", ACC, [("X = 2¹ = 2", "acc"), ("480 ÷ 2 = 240", "acc"),
                                              ("240 is exact", "acc"), ("decodes to 480", "acc")])]
    y = 30
    for title, col, bxs in rows:
        s.text(250, y - 6, title, size=18, anchor="start", weight="bold", fill=col)
        x = 250
        for i, (lab, kind) in enumerate(bxs):
            w = 138
            box(s, x, y + 4, w, 56, lab, kind, size=16)
            if i < len(bxs) - 1:
                s.line(x + w + 2, y + 32, x + w + 18, y + 32, color=col, sw=2.2)
            x += w + 20
        y += 128
    s.path("M166,108 C200,108 214,62 246,62", color=PROB, sw=2.2)
    s.path("M166,132 C200,132 214,190 246,190", color=ACC, sw=2.2)
    s.save(out("fig_rounding.svg"))


if __name__ == "__main__":
    fig_ladder()
    fig_where(1)
    fig_where(2)
    fig_overview()
    fig_mxblock()
    fig_two_copies()
    fig_encoding()
    fig_rounding()
