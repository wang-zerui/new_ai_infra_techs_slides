"""Generate the SVG diagrams of deck 20 (MoonEP mechanisms). Run: python3 gen_figs.py, then render.sh each SVG.

Running example (illustrative, see notes/sources.md): R=4 GPUs, E=8 experts (E/R=2, expert e lives on GPU e//2),
S=4 tokens per GPU, K=2 -> S*K = 8 token copies per GPU. Global expert loads E0..E7 = 9,3,7,5,2,2,2,2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import (SVG, ACCENT, ACCENT_FILL, PROBLEM, PROBLEM_FILL, NEUTRAL, NEUTRAL_FILL, MUTED, INK)

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = "Menlo, Courier New, monospace"

TPE = [  # tokens_per_expert[src][e]
    [3, 1, 2, 1, 0, 1, 0, 0],
    [2, 1, 2, 1, 1, 0, 1, 0],
    [2, 0, 2, 2, 0, 1, 0, 1],
    [2, 1, 1, 1, 1, 0, 1, 1],
]


def out(name):
    return os.path.join(HERE, name)


# ----------------------------------------------------------------------------------------------------------
# Slide 2: the planner's input, an R x E token-count matrix, and per-home-GPU load vs S*K
def fig_counts():
    s = SVG(480, 330)
    x0, cw = 104, 46
    # home-GPU group headers
    for g in range(4):
        xa = x0 + 2 * g * cw
        s.rect(xa + 2, 2, 2 * cw - 4, 26, NEUTRAL_FILL, NEUTRAL, 1.5, 3)
        s.text(xa + cw, 21, f"GPU {g}", 16, INK, "bold")
    s.text(96, 21, "home", 17, MUTED, "bold", "end")
    s.text(96, 50, "from", 17, MUTED, "bold", "end")
    for e in range(8):
        s.text(x0 + e * cw + cw / 2, 50, f"E{e}", 18, INK, "bold")
    y0, rh = 60, 34
    for r in range(4):
        y = y0 + r * rh
        s.text(96, y + rh / 2 + 6, f"GPU {r}", 18, INK, "bold", "end")
        for e in range(8):
            s.rect(x0 + e * cw, y, cw, rh, "#fff", "#BFBFBF", 1.2, 0)
            v = TPE[r][e]
            s.text(x0 + e * cw + cw / 2, y + rh / 2 + 6, str(v), 18, INK if v else "#A6A6A6")
    # totals per expert
    yt = y0 + 4 * rh + 6
    s.text(96, yt + 22, "total", 18, INK, "bold", "end")
    tot = [sum(TPE[r][e] for r in range(4)) for e in range(8)]
    for e in range(8):
        s.rect(x0 + e * cw, yt, cw, rh, NEUTRAL_FILL, NEUTRAL, 1.5, 0)
        s.text(x0 + e * cw + cw / 2, yt + rh / 2 + 6, str(tot[e]), 18, INK, "bold")
    # home load and balance
    yl = yt + rh + 12
    s.text(96, yl + 22, "load", 18, INK, "bold", "end")
    yb = yl + rh + 8
    s.text(96, yb + 22, "− S×K", 18, INK, "bold", "end")
    for g in range(4):
        load = tot[2 * g] + tot[2 * g + 1]
        kind = "problem" if load > 8 else "neutral"
        s.box(x0 + 2 * g * cw + 2, yl, 2 * cw - 4, rh, str(load), kind, 18, "bold", rx=3)
        bal = load - 8
        s.text(x0 + 2 * g * cw + cw, yb + 22, f"{bal:+d}".replace("-", "−"), 18,
               PROBLEM if bal > 0 else MUTED, "bold")
    s.save(out("fig_counts.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 4: the two horns of dynamic shapes
def fig_dilemma():
    s = SVG(870, 250)
    # (a) exact shapes: host sync each layer
    s.text(0, 22, "(a) exact shapes", 20, INK, "bold", "start")
    lx = 70
    hy, gy, hh, gh = 50, 104, 34, 42
    s.text(lx - 10, hy + hh / 2 + 6, "host", 18, INK, "bold", "end")
    s.text(lx - 10, gy + gh / 2 + 6, "GPU", 18, INK, "bold", "end")
    s.box(lx, gy, 112, gh, "dispatch", "white", 18, rx=3)
    s.rect(lx + 112, gy, 96, gh, "url(#hatchRed)", PROBLEM, 2, 3)
    s.raw(f'<rect x="{lx + 134}" y="{gy + 10}" width="52" height="22" fill="#fff"/>')
    s.text(lx + 160, gy + gh / 2 + 6, "idle", 18, PROBLEM, "bold")
    s.box(lx + 208, gy, 132, gh, "expert GEMM", "white", 18, rx=3)
    s.box(lx + 112, hy, 96, hh, "counts", "problem", 17, "bold", rx=3)
    s.line(lx + 116, gy - 2, lx + 116, hy + hh + 3, PROBLEM, 2, arrow=True)
    s.line(lx + 204, hy + hh + 2, lx + 204, gy - 3, PROBLEM, 2, arrow=True)
    s.text(lx + 170, 230, "host sync every layer", 18, PROBLEM, "bold")
    # divider
    s.line(440, 10, 440, 240, "#BFBFBF", 1.5)
    # (b) worst-case shapes: S*K*R buffer
    s.text(462, 22, "(b) worst-case shapes", 20, INK, "bold", "start")
    bx, by, seg, bh = 462, 92, 50, 52
    for i in range(8):
        if i == 0:
            s.rect(bx + i * seg, by, seg, bh, NEUTRAL_FILL, NEUTRAL, 2, 0)
            s.text(bx + seg / 2, by + bh / 2 + 6, "S×K", 17, INK, "bold")
        else:
            s.rect(bx + i * seg, by, seg, bh, "url(#hatchRed)", PROBLEM, 1.5, 0)
    s.rect(bx, by, 8 * seg, bh, "none", INK, 2, 0)
    # brace-like span labels
    s.line(bx, by - 14, bx + 8 * seg, by - 14, INK, 1.5, arrow=True, arrow_start=True)
    s.text(bx + 4 * seg, by - 22, "S×K×R rows reserved (R = 8)", 18, INK)
    s.line(bx + seg, by + bh + 14, bx + 8 * seg, by + bh + 14, PROBLEM, 1.5, arrow=True, arrow_start=True)
    s.text(bx + 4.5 * seg, by + bh + 38, "unused when balanced", 18, PROBLEM, "bold")
    s.text(bx + 4 * seg, 230, "R× buffer memory", 18, PROBLEM, "bold")
    s.save(out("fig_dilemma.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 6: where the three mechanisms act in one MoE layer (forward / backward)
def fig_overview():
    s = SVG(870, 252)
    lx = 104

    def lane(y, blocks):
        x = lx
        prev = None
        for w, lines, kind, badge in blocks:
            s.rect(x, y, w, 70, ACCENT_FILL if kind == "accent" else NEUTRAL_FILL,
                   ACCENT if kind == "accent" else NEUTRAL, 2, 4)
            s.ctext(x + w / 2 + (7 if badge else 0), y + 37, lines, 17,
                    ACCENT if kind == "accent" else INK, "bold" if kind == "accent" else "normal")
            if badge:
                s.badge(x + 14, y + 14, badge, r=11, size=15)
            if prev is not None:
                s.line(prev, y + 35, x - 2, y + 35, INK, 2, arrow=True)
            prev = x + w
            x += w + 16

    s.text(lx - 12, 52, "forward", 18, INK, "bold", "end")
    lane(16, [(78, ["router"], "neutral", None),
              (96, ["plan"], "accent", 2),
              (120, ["dispatch", "in place"], "accent", 3),
              (124, ["prefetch", "weights"], "accent", 1),
              (124, ["expert", "GEMM"], "neutral", None),
              (110, ["combine"], "accent", 3)])
    s.text(lx - 12, 182, "backward", 18, INK, "bold", "end")
    lane(146, [(160, ["dispatch", "saved plan"], "accent", 3),
               (124, ["expert", "GEMM bwd"], "neutral", None),
               (110, ["combine"], "accent", 3),
               (140, ["reduce", "replica grads"], "accent", 1)])
    s.rect(lx, 228, 26, 18, NEUTRAL_FILL, NEUTRAL, 2, 3)
    s.text(lx + 34, 243, "framework", 17, INK, "normal", "start")
    s.rect(lx + 140, 228, 26, 18, ACCENT_FILL, ACCENT, 2, 3)
    s.text(lx + 174, 243, "MoonEP", 17, INK, "normal", "start")
    s.save(out("fig_overview.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 7 (3 build steps): balance -> fill -> split, on the running example
def fig_plan(step):
    s = SVG(870, 280)
    bw, gap, by, bh = 196, 28, 50, 184
    xs = [i * (bw + gap) + 2 for i in range(4)]
    loads = [12, 12, 4, 4]
    chips = {0: [("E0", 9), ("E1", 3)], 1: [("E2", 7), ("E3", 5)], 2: [("E4", 2), ("E5", 2)], 3: [("E6", 2), ("E7", 2)]}
    if step == 3:
        chips[0] = [("E0", 5), ("E1", 3)]
        chips[1] = [("E2", 3), ("E3", 5)]
        loads = [8, 8, 8, 8]
    for g, x in enumerate(xs):
        over = loads[g] > 8
        stroke = PROBLEM if over else (ACCENT if step == 3 else NEUTRAL)
        s.rect(x, by, bw, bh, "#fff", stroke, 2.5 if over else 2, 5)
        s.text(x + bw / 2, by + 26, f"GPU {g}", 18, INK, "bold")
        for k, (e, n) in enumerate(chips[g]):
            s.box(x + 12, by + 38 + k * 36, bw - 24, 30, f"{e}: {n} tokens", "neutral", 17, rx=3)
        yk = by + 38 + 2 * 36
        if step == 3 and g in (2, 3):
            lab = "E0′: 4 tokens" if g == 2 else "E2′: 4 tokens"
            s.box(x + 12, yk, bw - 24, 30, lab, "accent", 17, "bold", rx=3)
        else:
            s.box(x + 12, yk, bw - 24, 30, "2 replica slots", "white", 16, rx=3, dash="5,4", tcolor=MUTED)
        bal = loads[g] - 8
        col = PROBLEM if bal > 0 else (ACCENT if step == 3 else MUTED)
        txt = (f"load {loads[g]}  ({bal:+d})".replace("-", "−")) if bal else f"load {loads[g]}  (0)"
        s.text(x + bw / 2, by + bh - 14, txt, 18, col, "bold")
    if step >= 2:
        c = [x + bw / 2 for x in xs]
        s.path(f"M{c[0] - 40},{by - 3} L{c[0] - 40},{26} L{c[2] - 40},{26} L{c[2] - 40},{by - 4}", ACCENT, 2.5,
               arrow=True)
        s.text((c[0] + c[2]) / 2 - 40, 18, "fill 4 tokens", 18, ACCENT, "bold")
        y2 = by + bh + 26
        s.path(f"M{c[1] + 40},{by + bh + 3} L{c[1] + 40},{y2} L{c[3] + 40},{y2} L{c[3] + 40},{by + bh + 4}",
               ACCENT, 2.5, arrow=True)
        s.text(c[1] + 28, y2 + 6, "fill 4 tokens", 18, ACCENT, "bold", "end")
    s.save(out(f"fig_plan_{step}.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 8: contiguous token ranges per expert (global order = source GPU order)
def fig_route():
    s = SVG(870, 270)
    x0, cw = 128, 58

    def row(y, name, srcs, cut, home, rep, cums):
        s.text(x0 - 14, y + 30, f"{name} copies", 18, INK, "bold", "end")
        for i, g in enumerate(srcs):
            moved = i >= cut
            s.rect(x0 + i * cw, y, cw - 4, 44, ACCENT_FILL if moved else NEUTRAL_FILL,
                   ACCENT if moved else NEUTRAL, 2, 3)
            s.text(x0 + i * cw + (cw - 4) / 2, y + 29, f"G{g}", 18, ACCENT if moved else INK,
                   "bold" if moved else "normal")
            s.text(x0 + i * cw + (cw - 4) / 2, y + 66, str(i), 16, MUTED)
        # brackets
        xa, xb = x0, x0 + cut * cw - 4
        s.path(f"M{xa},{y - 6} L{xa},{y - 14} L{xb},{y - 14} L{xb},{y - 6}", INK, 2)
        s.text((xa + xb) / 2, y - 22, f"stay on {home}: {cut}", 18, INK, "bold")
        xa, xb = x0 + cut * cw, x0 + len(srcs) * cw - 4
        s.path(f"M{xa},{y - 6} L{xa},{y - 14} L{xb},{y - 14} L{xb},{y - 6}", ACCENT, 2)
        s.text((xa + xb) / 2, y - 22, f"to {rep}: {len(srcs) - cut}", 18, ACCENT, "bold")
        s.text(866, y + 30, cums, 16, INK, "normal", "end", family=MONO)

    s.text(866, 18, "cumulative alloc", 17, MUTED, "bold", "end")
    s.text(866, 38, "GPU 0 → 3", 17, MUTED, "bold", "end")
    row(56, "E0", [0, 0, 0, 1, 1, 2, 2, 3, 3], 5, "GPU 0", "E0′ on GPU 2", "[5, 5, 9, 9]")
    row(186, "E2", [0, 0, 1, 1, 2, 2, 3], 3, "GPU 1", "E2′ on GPU 3", "[0, 3, 3, 7]")
    s.save(out("fig_route.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 9: what runs where inside the single planning launch
def fig_kernel():
    s = SVG(870, 286)
    lx = 120
    A, B, C, D = (126, 246), (272, 560), (600, 756), (768, 866)
    for (xa, xb), name in ((A, "A  gather"), (B, "B  plan + sort"), (C, "C  route")):
        s.text((xa + xb) / 2, 20, name, 18, MUTED, "bold")
    for xa in (259, 586, 762):
        s.line(xa, 30, xa, 274, "#BFBFBF", 1.5, dash="6,4")
    lh = 44
    y0, y1, y2, yh = 36, 100, 164, 228
    for name, y in (("GPU 0", y0), ("GPU 1", y1), ("GPU 2…R−1", y2), ("host", yh)):
        s.text(lx - 10, y + lh / 2 + 6, name, 18, INK if name != "host" else MUTED, "bold", "end")

    def blk(rng, y, label, kind="white", weight="normal", size=17):
        s.box(rng[0], y, rng[1] - rng[0], lh, label, kind, size, weight, rx=3)

    # A: every GPU pushes its tokens-per-expert row to GPU 0 (fan-in over NVLink)
    blk(A, y0, "R×E counts", "neutral", "bold")
    blk(A, y1, "push counts")
    blk(A, y2, "push counts")
    s.path(f"M{A[1] + 2},{y2 + lh / 2} L254,{y2 + lh / 2} L254,{y0 + lh / 2} L{A[1] + 3},{y0 + lh / 2}", INK, 2,
           arrow=True)
    s.line(A[1] + 2, y1 + lh / 2, 254, y1 + lh / 2, INK, 2)
    # B: GPU 0 plans; others sort their copies by expert (GPU 1 also sorts GPU 0's)
    blk(B, y0, "balance · split · layout", "accent", "bold", 18)
    blk(B, y1, "sort own + GPU 0 copies")
    blk(B, y2, "sort own copies by expert")
    # multicast of the plan into C on every GPU
    s.path(f"M{B[1] + 2},{y0 + lh / 2} L580,{y0 + lh / 2} L580,{y2 + lh / 2} L{C[0] - 3},{y2 + lh / 2}", ACCENT, 2.5,
           arrow=True)
    s.line(580, y1 + lh / 2, C[0] - 3, y1 + lh / 2, ACCENT, 2.5, arrow=True)
    s.line(580, y0 + lh / 2, C[0] - 3, y0 + lh / 2, ACCENT, 2.5, arrow=True)
    s.text(572, y0 + lh + 15, "multicast", 16, ACCENT, "bold", "end")
    # C and the dispatch that follows
    for y in (y0, y1, y2):
        blk(C, y, "slot per copy", "accent", "bold")
        blk(D, y, "dispatch")
    s.box(A[0], yh, D[1] - A[0], lh, "no per-layer host step", "white", 17, rx=3, dash="6,4", tcolor=MUTED)
    s.save(out("fig_kernel.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 10: GPU 2's fixed buffer (toy: NvS = 20 rows, padding P = 4) and the weight view it feeds
def fig_layout():
    s = SVG(870, 252)
    x0, cw, y, h = 150, 35, 62, 46
    # token rows: E4 2 + pad 2 | E5 2 + pad 2 | E0' 4 | slot 2 empty | tail 8 unused
    kinds = (["E4"] * 2 + ["pad"] * 2 + ["E5"] * 2 + ["pad"] * 2 + ["E0′"] * 4 + ["tail"] * 8)
    s.text(x0 - 14, y + 30, "tokens", 18, INK, "bold", "end")
    s.text(x0 - 14, y + 52, "[NvS, H]", 15, MUTED, "normal", "end", family=MONO)
    for i, k in enumerate(kinds):
        xx = x0 + i * cw
        if k == "pad":
            s.rect(xx, y, cw, h, "url(#hatchGray)", NEUTRAL, 1.5, 0)
        elif k == "tail":
            s.rect(xx, y, cw, h, "#fff", "#A6A6A6", 1.5, 0, dash="4,3")
        elif k == "E0′":
            s.rect(xx, y, cw, h, ACCENT_FILL, ACCENT, 1.5, 0)
        else:
            s.rect(xx, y, cw, h, NEUTRAL_FILL, NEUTRAL, 1.5, 0)
    # group labels under the strip
    groups = [(0, 4, "E4 + pad", INK), (4, 8, "E5 + pad", INK), (8, 12, "E0′", ACCENT), (12, 20, "unused tail", MUTED)]
    for a, b, lab, col in groups:
        s.path(f"M{x0 + a * cw + 2},{y + h + 6} L{x0 + a * cw + 2},{y + h + 14} L{x0 + b * cw - 2},{y + h + 14} "
               f"L{x0 + b * cw - 2},{y + h + 6}", col, 2)
        s.text(x0 + (a + b) * cw / 2, y + h + 36, lab, 18, col, "bold")
    # cu_seqlens markers (end offsets of the 2*E/R groups)
    s.text(x0 - 14, 38, "cu_seqlens", 16, MUTED, "bold", "end", family=MONO)
    for v in (4, 8, 12):
        xx = x0 + v * cw
        s.line(xx, 30, xx, y + h + 2, ACCENT, 2.5)
        s.text(xx, 24, str(v), 18, ACCENT, "bold")
    s.text(x0 + 12 * cw + 22, 24, "(slot 2: 12)", 16, MUTED, "normal", "start")
    # weight compute view [2E/R, H, H'] = local experts + prefetch slots
    wy, wh = 182, 48
    s.text(x0 - 14, wy + 28, "weights", 18, INK, "bold", "end")
    s.text(x0 - 14, wy + 48, "[4, H, H′]", 15, MUTED, "normal", "end", family=MONO)
    wcols = [(0, 4, "E4 local", "neutral"), (4, 8, "E5 local", "neutral"), (8, 12, "E0′ prefetched", "accent"),
             (12, 16, "slot 2 empty", "white")]
    for a, b, lab, kind in wcols:
        s.box(x0 + a * cw + 3, wy, (b - a) * cw - 6, wh, lab, kind, 17, "bold" if kind == "accent" else "normal",
              rx=3, dash="5,4" if kind == "white" else None, tcolor=MUTED if kind == "white" else None)
    s.save(out("fig_layout.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 11: replica weights forward, replica grads backward (E0 between GPU 0 and GPU 2)
def fig_replica_io():
    s = SVG(480, 320)
    # GPU 0 (home) and GPU 2 (replica host)
    s.rect(0, 40, 196, 250, "#fff", NEUTRAL, 2, 6)
    s.text(98, 30, "GPU 0 (home)", 18, INK, "bold")
    s.rect(284, 40, 196, 250, "#fff", NEUTRAL, 2, 6)
    s.text(382, 30, "GPU 2", 18, INK, "bold")
    s.box(14, 64, 168, 56, ["E0 weights", "BF16"], "neutral", 17)
    s.box(298, 64, 168, 56, ["prefetch slot", "E0′"], "accent", 17, "bold")
    s.box(14, 210, 168, 56, ["∇E0 (FP32)", "+= ∇E0′"], "neutral", 17)
    s.box(298, 210, 168, 56, ["reduce slot", "∇E0′ (FP32)"], "accent", 17, "bold")
    # forward push, backward pull
    s.line(184, 92, 294, 92, ACCENT, 2.5, arrow=True)
    s.text(239, 82, "forward", 17, ACCENT, "bold")
    s.line(296, 238, 186, 238, ACCENT, 2.5, dash="8,5", arrow=True)
    s.text(239, 228, "backward", 17, ACCENT, "bold")
    s.text(382, 160, "4 E0 tokens", 17, INK, "normal")
    s.text(382, 182, "computed here", 17, INK, "normal")
    s.text(98, 160, "5 E0 tokens", 17, INK, "normal")
    s.text(98, 182, "computed here", 17, INK, "normal")
    s.save(out("fig_replica_io.svg"))


if __name__ == "__main__":
    fig_counts()
    fig_dilemma()
    fig_overview()
    for st in (1, 2, 3):
        fig_plan(st)
    fig_route()
    fig_kernel()
    fig_layout()
    fig_replica_io()
