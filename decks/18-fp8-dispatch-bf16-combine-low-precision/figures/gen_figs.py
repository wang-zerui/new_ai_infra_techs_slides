"""Deck 18 figures (slide scale, 1 unit = 1 pt). Run: python3 gen_figs.py, then render each SVG."""
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(HERE, n)
BAND_F = "#F7F7F7"
DARK = "#404040"          # BF16 (expert -> token) arrows


def badge(s, x, y, n, r=13, fill=ACC):
    s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>')
    s.text(x, y + 6.3, str(n), 18, "#FFFFFF", weight="bold")


def band(s, x, y, w, h, label=None, ly=None):
    s.rect(x, y, w, h, BAND_F, "#D0D0D0", 1.2, rx=0, dash="5,4")
    if label:
        s.text(x + w / 2, ly if ly else y + h - 8, label, 17, MUTED, style="italic")


# ----------------------------------------------------------------------------------------- fig_layer
def fig_layer():
    s = SVG(480, 330, "fig_layer: one token fans out to top-k experts (dispatch) and is summed back (combine)")
    band(s, 112, 30, 78, 266)
    band(s, 330, 30, 52, 266)
    s.text(151, 318, "network", 17, MUTED, style="italic")
    s.text(356, 318, "network", 17, MUTED, style="italic")
    s.box(6, 135, 104, 60, ["token t", "home GPU"], NEU_F, NEU)
    ys = [38, 98, 194, 254]
    names = ["E3 · GPU 5", "E41 · GPU 9", "E97 · GPU 20", "E250 · GPU 61"]
    for y, n in zip(ys, names):
        s.box(192, y, 136, 40, n, "#FFFFFF", NEU, size=17)
    s.text(260, 166, "⋯ top-k", 18, MUTED)
    s.box(384, 135, 90, 60, ["weighted", "sum t′"], NEU_F, NEU)
    for y in ys:
        s.line(110, 165, 190, y + 20, INK, 2.2, "arr")
        s.line(328, y + 20, 382, 165, INK, 2.2, "arr")
    s.text(151, 22, "dispatch", 18, INK, weight="bold")
    s.text(372, 22, "combine", 18, INK, weight="bold")
    s.save(P("fig_layer.svg"))


# ----------------------------------------------------------------------------------------- fig_before
def fig_before():
    s = SVG(870, 270, "fig_before: BF16 dispatch is cast to FP8 on arrival; BF16 combine returns")
    s.text(130, 24, "home GPU", 18, MUTED, weight="bold")
    s.text(345, 24, "network", 18, MUTED, weight="bold")
    s.text(650, 24, "expert GPU", 18, MUTED, weight="bold")
    band(s, 262, 36, 166, 222)
    # dispatch row
    s.box(20, 58, 220, 50, "token in BF16", NEU_F, NEU)
    s.line(242, 83, 446, 83, PRB, 3, "arrPrb")
    s.text(345, 72, "BF16 · 2h bytes", 18, PRB, weight="bold")
    s.box(450, 58, 150, 50, "cast to FP8", PRB_F, PRB, color=PRB, weight="bold")
    s.line(602, 83, 628, 83, INK, 2.2, "arr")
    s.box(630, 58, 220, 50, "FP8 up-proj GEMM", NEU_F, NEU)
    # expert compute down
    s.line(740, 110, 740, 176, INK, 2.2, "arr")
    s.text(728, 149, "SwiGLU · down-proj", 17, MUTED, anchor="end", style="italic")
    # combine row
    s.box(630, 178, 220, 50, "expert out in BF16", NEU_F, NEU)
    s.line(628, 203, 244, 203, INK, 2.2, "arr")
    s.text(345, 192, "BF16 · 2h bytes", 18, INK)
    s.box(20, 178, 220, 50, "weighted sum", NEU_F, NEU)
    s.save(P("fig_before.svg"))


# ----------------------------------------------------------------------------------------- fig_overview
def fig_overview():
    s = SVG(870, 318, "fig_overview: token->expert transfers are FP8, expert->token transfers are BF16, fwd and bwd")
    s.text(140, 22, "home GPU", 18, MUTED, weight="bold")
    s.text(380, 22, "network", 18, MUTED, weight="bold")
    s.text(660, 22, "expert GPU", 18, MUTED, weight="bold")
    band(s, 282, 32, 196, 282)
    rows = [("forward", 40, "quantize 1×128 tiles", "expert FFN (FP8 GEMMs)"),
            ("backward", 184, "quantize grad 1×128", "expert bwd (FP8 GEMMs)")]
    for k, (tag, y0, qlabel, elabel) in enumerate(rows):
        # row tag
        s.text(8, y0 + 12, tag, 17, MUTED, anchor="start", style="italic")
        s.box(30, y0 + 20, 240, 46, qlabel, ACC_F, ACC, color=ACC, weight="bold")
        s.box(30, y0 + 80, 240, 46, "Σ over k · FP32 acc", NEU_F, NEU)
        s.box(490, y0 + 20, 360, 106, elabel, NEU_F, NEU)
        # token -> expert (FP8)
        s.line(272, y0 + 43, 488, y0 + 43, ACC, 3, "arrAcc")
        s.text(390, y0 + 33, "dispatch · FP8", 18, ACC, weight="bold")
        # expert -> token (BF16)
        s.line(488, y0 + 103, 272, y0 + 103, DARK, 3, "arr")
        s.text(390, y0 + 93, "combine · BF16", 18, DARK, weight="bold")
    badge(s, 298, 64, 1)
    badge(s, 298, 124, 2)
    s.add(f'<line x1="2" y1="172" x2="868" y2="172" stroke="#BFBFBF" stroke-width="1" stroke-dasharray="4,4"/>')
    badge(s, 108, 190, 3)
    s.save(P("fig_overview.svg"))


# ----------------------------------------------------------------------------------------- fig_message
def fig_message():
    s = SVG(870, 300, "fig_message: per-token message, BF16 vs FP8 + per-tile scales, drawn to scale (h = 7168)")
    X0, W = 176, 680          # 680 pt = 2h = 14,336 B
    k = W / 14336
    # BF16 row
    s.text(164, 52, "BF16", 19, INK, anchor="end", weight="bold")
    s.rect(X0, 26, W, 40, NEU_F, NEU, 2, rx=2)
    s.text(X0 + W / 2, 52, "BF16 hidden · 14,336 B", 18, INK)
    # FP8 row
    s.text(164, 124, "FP8", 19, ACC, anchor="end", weight="bold")
    wd = 7168 * k
    s.rect(X0, 98, wd, 40, ACC_F, ACC, 2, rx=2)
    s.text(X0 + wd / 2, 124, "E4M3 data · 7,168 B", 18, ACC, weight="bold")
    ws = max(224 * k, 6)
    s.rect(X0 + wd, 98, ws, 40, ACC, ACC, 1, rx=0)
    s.rect(X0 + wd + ws, 98, 6, 40, "#A6A6A6", NEU, 1, rx=0)
    # scale + metadata callouts (right of the FP8 bar)
    sx = X0 + wd + ws / 2
    s.path(f"M{sx:.1f},96 L{sx:.1f},88 L556,88", ACC, 1.6)
    s.text(560, 94, "scales · 224 B FP32 or 56 B UE8M0", 17, ACC, anchor="start")
    mx = X0 + wd + ws + 3
    s.line(mx + 2, 118, 556, 118, MUTED, 1.4)
    s.text(560, 124, "top-k ids and weights", 16, MUTED, anchor="start", style="italic")
    # zoom: tiles
    zy = 214
    s.line(X0, 140, X0, zy - 4, ACC, 1.2, dash="4,3")
    s.line(X0 + wd + ws, 140, 854, zy - 4, ACC, 1.2, dash="4,3")
    tiles = [("128 × E4M3", "s1"), ("128 × E4M3", "s2"), ("128 × E4M3", "s3")]
    x = X0
    for lab, sc in tiles:
        s.rect(x, zy, 118, 40, ACC_F, ACC, 2, rx=2)
        s.text(x + 59, zy + 26, lab, 16, ACC)
        s.rect(x + 118, zy, 30, 40, ACC, ACC, 1, rx=2)
        s.text(x + 133, zy + 26, sc, 16, "#FFFFFF", weight="bold")
        x += 160
    s.text(x + 18, zy + 26, "⋯", 22, ACC)
    s.rect(x + 46, zy, 118, 40, ACC_F, ACC, 2, rx=2)
    s.text(x + 105, zy + 26, "128 × E4M3", 16, ACC)
    s.rect(x + 164, zy, 30, 40, ACC, ACC, 1, rx=2)
    s.text(x + 179, zy + 26, "s56", 15, "#FFFFFF", weight="bold")
    s.text(164, zy + 26, "1×128", 18, ACC, anchor="end", weight="bold")
    s.text(X0, zy + 70, "one scale per 128-value tile · 56 per token", 17, MUTED, anchor="start", style="italic")
    s.save(P("fig_message.svg"))


# ----------------------------------------------------------------------------------------- fig_combine
def fig_combine():
    s = SVG(870, 300, "fig_combine: experts' BF16 outputs are summed per node over NVLink, then once per node over IB")
    # home GPU
    s.rect(6, 70, 268, 160, "#FFFFFF", NEU, 1.8, rx=6)
    s.text(18, 94, "home GPU · node A", 17, MUTED, anchor="start", weight="bold")
    s.box(24, 110, 232, 46, "Σ over nodes · FP32", NEU_F, NEU)
    s.line(140, 158, 140, 172, INK, 2, "arr")
    s.box(24, 174, 232, 44, "BF16 output", NEU_F, NEU)
    # remote nodes
    nodes = [("node B", 12, ["E12", "E14", "E15"]), ("node C", 160, ["E40", "E43"])]
    for name, y0, experts in nodes:
        s.rect(452, y0, 412, 128, "#FFFFFF", NEU, 1.6, rx=6, dash="7,4")
        s.text(462, y0 + 22, name, 17, MUTED, anchor="start", weight="bold")
        s.box(470, y0 + 40, 170, 64, ["forwarding GPU", "partial Σ · FP32"], NEU_F, NEU)
        n = len(experts)
        ys = [y0 + 64 - (n - 1) * 20 + i * 40 for i in range(n)]
        for e, ye in zip(experts, ys):
            s.box(752, ye - 16, 96, 32, e, "#FFFFFF", NEU, size=17)
            s.line(750, ye, 642, y0 + 72, INK, 1.8, "arr")
        s.text(696, y0 + 22, "NVLink", 17, MUTED, style="italic")
        # IB arrow to home
        s.line(468, y0 + 72, 276, 150 if y0 < 100 else 160, DARK, 3, "arr")
    s.text(364, 96, "IB · BF16", 18, DARK, weight="bold", halo=True)
    s.text(364, 236, "IB · BF16", 18, DARK, weight="bold", halo=True)
    s.save(P("fig_combine.svg"))


# ----------------------------------------------------------------------------------------- fig_mirror
def fig_mirror():
    s = SVG(870, 288, "fig_mirror: d(dispatch)=combine and d(combine)=dispatch; fan-out is FP8, fan-in is BF16")

    def mini(x0, y0, fanout, label, tag, sub=None):
        tx, ty = x0 + 40, y0 + 50
        s.add(f'<circle cx="{tx}" cy="{ty}" r="22" fill="{NEU_F}" stroke="{NEU}" stroke-width="2"/>')
        s.text(tx, ty + 6, "t", 18, INK, weight="bold")
        ex = x0 + 316
        eys = [y0 + 16, y0 + 50, y0 + 84]
        for i, ey in enumerate(eys):
            s.box(ex, ey - 14, 92, 28, f"E{i + 1}", "#FFFFFF", NEU, size=17)
        col = ACC if fanout else DARK
        mk = "arrAcc" if fanout else "arr"
        for ey in eys:
            if fanout:
                s.line(tx + 24, ty, ex - 2, ey, col, 2.6, mk)
            else:
                s.line(ex - 2, ey, tx + 24, ty, col, 2.6, mk)
        s.text(x0 + 190, y0 + 2, label, 18, col, weight="bold", halo=True)
        s.text(x0, y0 + 2, tag, 17, MUTED, anchor="start", style="italic")
        if sub:
            s.text(x0, y0 + 102, sub, 16, MUTED, anchor="start", family="Menlo, monospace")

    for px, title, fwd_out in [(10, "(a) dispatch", True), (452, "(b) combine", False)]:
        s.text(px + 204, 20, title, 19, INK, weight="bold")
        mini(px, 46, fwd_out, "copy · FP8" if fwd_out else "sum · BF16", "forward")
        mini(px, 172, not fwd_out, "sum · BF16" if fwd_out else "copy · FP8", "backward",
             "runs combine()" if fwd_out else "runs dispatch()")
    s.add('<line x1="436" y1="6" x2="436" y2="282" stroke="#BFBFBF" stroke-width="1.2"/>')
    s.add('<line x1="10" y1="152" x2="862" y2="152" stroke="#BFBFBF" stroke-width="1" stroke-dasharray="4,4"/>')
    s.save(P("fig_mirror.svg"))


if __name__ == "__main__":
    fig_layer(); fig_before(); fig_overview(); fig_message(); fig_combine(); fig_mirror()
