"""Figures for deck 53 (memory-efficient token permutation). 1 user unit = 1 pt.
Run: python3 gen_figs.py, then render each fig_*.svg with ipads-arch-figures/scripts/render.sh."""
import html
import os
from svgkit import SVG, ACC, ACC_F, PRB, PRB_F, NEU, NEU_F, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(HERE, n)


def chip(s, x, y, w, h, label, kind, size=17):
    if kind == "keep":
        s.box(x, y, w, h, label, NEU_F, NEU, INK, size, sw=1.8, rx=9)
    elif kind == "drop":
        s.box(x, y, w, h, label, "#FFFFFF", ACC, ACC, size, weight="bold", sw=2.2, rx=9, dash="6,4")
    elif kind == "pinned":
        s.box(x, y, w, h, label, PRB_F, PRB, PRB, size, weight="bold", sw=2.2, rx=9)
    elif kind == "new":
        s.box(x, y, w, h, label, ACC_F, ACC, ACC, size, weight="bold", sw=2.2, rx=9)


def rich(s, x, y, parts, size=22, anchor="middle", family="Times New Roman, Times, serif"):
    """parts: list of (text, opts) with opts keys: sub, sup, color, weight, style."""
    out = [f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{INK}" '
           f'font-family="{family}">']
    for t, o in parts:
        attrs = []
        if o.get("sub"):
            attrs.append(f'baseline-shift="sub" font-size="{size * 0.68:.1f}"')
        if o.get("sup"):
            attrs.append(f'baseline-shift="super" font-size="{size * 0.68:.1f}"')
        if o.get("color"):
            attrs.append(f'fill="{o["color"]}"')
        if o.get("weight"):
            attrs.append(f'font-weight="{o["weight"]}"')
        attrs.append(f'font-style="{o.get("style", "italic" if o.get("math") else "normal")}"')
        out.append(f'<tspan {" ".join(attrs)}>{html.escape(t)}</tspan>')
    out.append("</text>")
    s.add("".join(out))


M = {"math": True}
R = {}
SUB = {"sub": True, "math": True}
SUBR = {"sub": True}
SUPR = {"sup": True}


def pA(extra=None):
    d = {"math": True, "color": ACC, "weight": "bold"}
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------- fig_moe (hook)
def fig_moe():
    s = SVG(870, 236, "MoE layer: router weights applied after the expert, at unpermute")
    y = 92
    ops = [(70, 104, "permute"), (196, 104, "dispatch"), (322, 174, "experts"), (518, 104, "combine"),
           (644, 150, "unpermute")]
    s.text(30, y + 29, "x", 22, INK, style="italic", family="Times New Roman, Times, serif")
    s.line(44, y + 22, 68, y + 22, INK, 2.2, "arr")
    for x, w, n in ops:
        s.box(x, y, w, 44, n, "#FFFFFF", NEU, INK, 18)
    for i in range(len(ops) - 1):
        s.line(ops[i][0] + ops[i][1] + 1, y + 22, ops[i + 1][0] - 1, y + 22, INK, 2.2, "arr")
    s.line(795, y + 22, 830, y + 22, INK, 2.2, "arr")
    s.text(846, y + 29, "y", 22, INK, style="italic", family="Times New Roman, Times, serif")
    # expert internals label
    s.text(409, y + 66, "fc1 · SwiGLU · fc2", 16, MUTED, style="italic")
    # router above: x feeds router; router feeds the routing map to permute and p to unpermute
    s.box(70, 14, 104, 40, "router", "#FFFFFF", NEU, INK, 18)
    s.path(f"M52,{y + 22} L52,34 L68,34", INK, 2, "arr")
    s.line(122, 56, 122, y - 1, INK, 2, "arr")
    s.text(130, 78, "map", 16, MUTED, anchor="start", style="italic")
    s.path("M176,34 L719,34 L719,90", INK, 2.2, "arr")
    rich(s, 450, 26, [("routing weights ", {}), ("p", M)], 18, family="Helvetica, Arial, sans-serif")
    # weighting marker at unpermute
    s.text(719, y + 66, "weight by p, then sum", 16, INK, style="italic")
    # equation
    rich(s, 435, 212, [("y", M), (" = Σ", R), ("i", SUB), ("  p", M), ("i", SUB), (" · E", M), ("i", SUB),
                       ("(x),\u2003\u2003E", M), ("i", SUB), ("(x) = W", M), ("2", SUBR), (" φ(W", M), ("1", SUBR),
                       (" x)", M)], 26)
    s.save(P("fig_moe.svg"))


# ---------------------------------------------------------------- fig_before
def fig_before():
    s = SVG(870, 262, "standard order: the router-weight gradient pins the expert output")
    y = 40
    s.text(8, 22, "one routed copy of one token", 16, MUTED, anchor="start", style="italic")
    rich(s, 104, y + 30, [("x", M)], 24)
    ops = [(150, 72, "fc1"), (300, 100, "SwiGLU"), (478, 72, "fc2"), (628, 72, "× p"), (774, 72, "Σ")]
    labels = ["x", "z", "a", "E", ""]
    s.line(118, y + 22, 148, y + 22, INK, 2.2, "arr")
    for (x, w, n) in ops:
        s.box(x, y, w, 44, n, "#FFFFFF", NEU, INK, 19)
    for i in range(len(ops) - 1):
        x1 = ops[i][0] + ops[i][1]
        x2 = ops[i + 1][0]
        s.line(x1 + 1, y + 22, x2 - 1, y + 22, INK, 2.2, "arr")
    # saved tensors under each consumer
    saved = [(186, "x", "keep", "fc1 wgrad"), (350, "z", "keep", "SwiGLU bwd"), (514, "a", "keep", "fc2 wgrad"),
             (664, "E", "pinned", "∂L/∂p only")]
    for cx, lab, kind, why in saved:
        s.line(cx, y + 46, cx, y + 76, PRB if kind == "pinned" else MUTED, 1.8,
               "arrPrb" if kind == "pinned" else "arrGray")
        chip(s, cx - 34, y + 78, 68, 36, lab, kind, 20)
        s.text(cx, y + 140, why, 16, PRB if kind == "pinned" else MUTED, style="italic")
    s.text(8, y + 102, "saved for", 17, INK, anchor="start", weight="bold")
    s.text(8, y + 122, "backward", 17, INK, anchor="start", weight="bold")
    # gradient formula
    rich(s, 664, 236, [("∂L/∂p", pA({"color": PRB})), (" = ⟨ ", R), ("E", {"math": True, "color": PRB, "weight": "bold"}),
                       (", ∂L/∂y ⟩", M)], 24)
    s.text(300, 232, "E: one hidden-size row per routed copy", 18, INK)
    s.save(P("fig_before.svg"))


# ---------------------------------------------------------------- fig_algebra (insight)
def fig_algebra():
    s = SVG(870, 262, "move p_i inside W2: scalar multiplication commutes with a bias-free linear map")
    F = 32
    s.text(30, 58, "(1)", 20, MUTED, anchor="start")
    rich(s, 110, 58, [("y", M), (" = Σ", R), ("i", SUB)], F, anchor="start")
    rich(s, 206, 58, [("p", pA()), ("i", pA({"sub": True}))], F, anchor="start")
    rich(s, 240, 58, [("· W", M), ("2", SUBR), (" φ(z", M), ("i", SUB), (")", M)], F, anchor="start")
    s.text(30, 146, "(2)", 20, MUTED, anchor="start")
    rich(s, 110, 146, [("y", M), (" = Σ", R), ("i", SUB), (" W", M), ("2", SUBR), (" (", R)], F, anchor="start")
    rich(s, 254, 146, [("p", pA()), ("i", pA({"sub": True}))], F, anchor="start")
    rich(s, 288, 146, [("· φ(z", M), ("i", SUB), ("))", M)], F, anchor="start")
    # move arrow from p_i in (1) to p_i in (2)
    s.path("M218,72 C220,100 252,98 264,114", ACC, 2.4, "arrAcc")
    s.text(540, 98, "valid when W", 19, MUTED, anchor="start", style="italic")
    s.text(654, 104, "2", 14, MUTED, anchor="start", style="italic")
    s.text(666, 98, "is linear, no bias", 19, MUTED, anchor="start", style="italic")
    s.line(30, 176, 840, 176, "#D9D9D9", 1.2)
    rich(s, 110, 228, [("∂L/∂p", pA()), ("i", pA({"sub": True})), (" = ⟨ φ(z", M), ("i", SUB), ("), ∂L/∂a", M),
                       ("i", SUB), (" ⟩", R)], 30, anchor="start")
    s.text(540, 214, "φ(z) rebuilt from z in the kernel", 19, ACC, anchor="start", weight="bold")
    s.text(540, 240, "z is saved for SwiGLU anyway", 18, MUTED, anchor="start", style="italic")
    s.save(P("fig_algebra.svg"))


# ---------------------------------------------------------------- fig_pipeline (overview)
def fig_pipeline():
    s = SVG(870, 268, "before vs after: where the router weights are applied")

    def row(y, title, ops, accent_idx, pinned_idx):
        s.text(8, y - 10, title, 19, ACC if accent_idx else INK, anchor="start", weight="bold")
        for i, (x, w, n) in enumerate(ops):
            if i in accent_idx:
                s.box(x, y, w, 46, n, ACC_F, ACC, ACC, 17, sw=2.4)
            elif i in pinned_idx:
                s.box(x, y, w, 46, n, PRB_F, PRB, PRB, 17, sw=2.4)
            else:
                s.box(x, y, w, 46, n, "#FFFFFF", NEU, INK, 17)
        for i in range(len(ops) - 1):
            s.line(ops[i][0] + ops[i][1] + 1, y + 23, ops[i + 1][0] - 1, y + 23, INK, 2, "arr")

    xs = [(6, 118), (142, 124), (284, 56), (358, 124), (500, 56), (574, 100), (692, 176)]
    before = ["permute x", "dispatch x", "fc1", "SwiGLU", "fc2", "combine", "unpermute: p·E, Σ"]
    after = ["permute x, p", "dispatch x, p", "fc1", "p · SwiGLU", "fc2", "combine", "unpermute: Σ"]
    row(44, "Standard", [(x, w, n) for (x, w), n in zip(xs, before)], set(), {6})
    row(176, "Memory-efficient", [(x, w, n) for (x, w), n in zip(xs, after)], {0, 1, 3, 6}, set())
    s.text(780, 112, "keeps E for ∂L/∂p", 16, PRB, style="italic")
    s.text(420, 244, "fused weighted SwiGLU", 16, ACC, style="italic")
    s.text(780, 244, "plain sum", 16, ACC, style="italic")
    s.save(P("fig_pipeline.svg"))


# ---------------------------------------------------------------- fig_dispatch (mechanism 1)
def fig_dispatch():
    s = SVG(870, 300, "worked example: probs permuted and dispatched with their tokens")
    # columns: tokens | permuted rows by expert | experts | unpermute sum
    s.text(60, 22, "tokens", 18, MUTED, weight="bold")
    s.text(300, 22, "rows grouped by expert", 18, MUTED, weight="bold")
    s.text(560, 22, "local experts", 18, MUTED, weight="bold")
    s.text(790, 22, "unpermute", 18, MUTED, weight="bold")
    toks = {"t1": 70, "t2": 190}
    for t, yy in toks.items():
        s.box(20, yy, 80, 44, t, NEU_F, NEU, INK, 19, weight="bold")
    rows = [("t1", "E1", "p", "11", 44), ("t2", "E1", "p", "21", 100), ("t2", "E2", "p", "22", 156),
            ("t1", "E3", "p", "13", 212)]
    for t, e, pp, idx, yy in rows:
        s.rect(220, yy, 110, 44, "#FFFFFF", NEU, 1.8, 4)
        s.text(275, yy + 29, f"{t} → {e}", 18, INK)
        s.rect(334, yy, 64, 44, ACC_F, ACC, 2, 4)
        rich(s, 366, yy + 29, [("p", pA()), (idx, pA({"sub": True}))], 22)
        # token -> row arrows
        ty = toks[t] + 22
        s.path(f"M102,{ty} C160,{ty} 160,{yy + 22} 218,{yy + 22}", MUTED, 1.8, "arrGray")
    # experts
    exps = [("E1", 44, 100), ("E2", 156, 44), ("E3", 212, 44)]
    for name, yy, hh in exps:
        s.box(500, yy, 120, hh, f"{name}: p · SwiGLU", ACC_F, ACC, ACC, 15, weight="bold")
    for _, _, _, _, yy in rows:
        s.line(400, yy + 22, 498, yy + 22, INK, 2, "arr")
    # unpermute: sum back to tokens
    outs = {"t1": 80, "t2": 190}
    for t, yy in outs.items():
        s.box(760, yy, 90, 44, f"Σ → {t}", "#FFFFFF", NEU, INK, 18)
    for t, e, _, _, yy in rows:
        oy = outs[t] + 22
        s.path(f"M622,{yy + 22} C690,{yy + 22} 690,{oy} 758,{oy}", MUTED, 1.8, "arrGray")
    s.text(449, 288, "dispatch: rows and their p in two all-to-alls", 16, MUTED, style="italic")
    s.line(449, 262, 449, 270, MUTED, 1.2)
    s.save(P("fig_dispatch.svg"))


# ---------------------------------------------------------------- fig_swiglu (mechanism 2)
def fig_swiglu():
    s = SVG(870, 290, "fused weighted SwiGLU: forward saves z and p; backward rebuilds SwiGLU(z)")
    s.text(8, 34, "forward", 19, INK, anchor="start", weight="bold")
    s.text(8, 184, "backward", 19, INK, anchor="start", weight="bold")
    # forward row
    chip(s, 110, 12, 70, 36, "z", "keep", 20)
    chip(s, 110, 58, 70, 36, "p", "new", 20)
    s.box(250, 22, 220, 60, "a = p · SwiGLU(z)", ACC_F, ACC, ACC, 19, weight="bold")
    s.line(182, 30, 248, 42, INK, 2, "arr")
    s.line(182, 76, 248, 64, ACC, 2, "arrAcc")
    s.box(540, 30, 80, 44, "fc2", "#FFFFFF", NEU, INK, 19)
    s.line(472, 52, 538, 52, INK, 2.2, "arr")
    s.text(700, 46, "saved: z and p", 18, INK, anchor="start")
    s.text(700, 70, "no E", 18, ACC, anchor="start", weight="bold")
    s.line(8, 116, 862, 116, "#D9D9D9", 1.2)
    # backward row
    chip(s, 110, 150, 70, 36, "g", "keep", 20)
    s.text(145, 210, "∂L/∂a", 16, MUTED, style="italic")
    s.box(250, 136, 300, 44, "∂L/∂z = SwiGLU′(z) · (p g)", "#FFFFFF", NEU, INK, 18)
    s.box(250, 196, 300, 44, "∂L/∂p = Σ  SwiGLU(z) ⊙ g", ACC_F, ACC, ACC, 18, weight="bold")
    s.line(182, 168, 248, 158, INK, 2, "arr")
    s.line(182, 168, 248, 218, INK, 2, "arr")
    s.text(578, 223, "SwiGLU(z) recomputed on the fly", 17, ACC, anchor="start", style="italic")
    s.text(578, 163, "one fused kernel", 17, MUTED, anchor="start", style="italic")
    s.text(400, 274, "Σ runs over the row's FFN dimension", 16, MUTED, style="italic")
    s.save(P("fig_swiglu.svg"))


# ---------------------------------------------------------------- fig_ledger (what disappears)
def fig_ledger():
    """Values saved per routed copy, DeepSeek-V3 (analytic): x 7168 (fc1 input), z 4096 (fc1 out, gated 2x2048),
    a 2048 (fc2 input), E 7168 (expert output kept for dp); after: p, 1 value."""
    s = SVG(870, 176, "values saved per routed token copy, before and after")
    X0 = 160
    k = 700 / (7168 + 4096 + 2048 + 7168)

    def seg(x, y, v, name, num, kind):
        w = v * k
        if kind == "keep":
            s.rect(x, y, w, 52, NEU_F, NEU, 1.6, 2)
            col, wt = INK, "normal"
        else:
            s.rect(x, y, w, 52, PRB_F, PRB, 2.2, 2)
            col, wt = PRB, "bold"
        s.text(x + w / 2, y + 22, name, 18, col, weight="bold", style="italic")
        s.text(x + w / 2, y + 44, num, 16, col, weight=wt)
        return x + w

    s.text(X0 - 12, 50, "Standard", 19, INK, anchor="end", weight="bold")
    x = X0
    for v, n, num, kind in [(7168, "x", "7,168", "keep"), (4096, "z", "4,096", "keep"), (2048, "a", "2,048", "keep"),
                            (7168, "E", "7,168", "pin")]:
        x = seg(x, 20, v, n, num, kind)
    end = x
    s.text(X0 - 12, 138, "Mem-efficient", 19, ACC, anchor="end", weight="bold")
    x = X0
    for v, n, num, kind in [(7168, "x", "7,168", "keep"), (4096, "z", "4,096", "keep"), (2048, "a", "2,048", "keep")]:
        x = seg(x, 108, v, n, num, kind)
    s.rect(x, 108, 8, 52, ACC_F, ACC, 2.2, 1)
    s.text(x + 4, 102, "p", 18, ACC, weight="bold", style="italic")
    s.rect(x + 16, 108, end - x - 16, 52, "#FFFFFF", ACC, 2, 2, dash="6,4")
    s.text((x + 16 + end) / 2, 140, "E not stored", 18, ACC, weight="bold")
    s.save(P("fig_ledger.svg"))


if __name__ == "__main__":
    fig_moe()
    fig_before()
    fig_algebra()
    fig_pipeline()
    fig_dispatch()
    fig_swiglu()
    fig_ledger()
