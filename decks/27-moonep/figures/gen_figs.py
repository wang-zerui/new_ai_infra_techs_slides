"""Generate the SVG diagrams of deck 27 (MoonEP overview). Run: python3 gen_figs.py (then render.sh each SVG)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import (SVG, ACCENT, ACCENT_FILL, PROBLEM, PROBLEM_FILL, NEUTRAL, NEUTRAL_FILL, MUTED, INK)

HERE = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(HERE, name)


# ----------------------------------------------------------------------------------------------------------
# Slide 2: one MoE layer under expert parallelism (4 GPUs, 2 experts each, all-to-all dispatch)
def fig_ep_layer():
    s = SVG(480, 340)
    lane_y = [46 + i * 74 for i in range(4)]
    lh = 56
    s.text(118, 26, "tokens", 18, MUTED, "bold")
    s.text(227, 26, "all-to-all", 18, MUTED, "bold")
    s.text(383, 26, "E/R experts per GPU", 18, MUTED, "bold")
    tx0, tx1 = 72, 164
    ex0 = 290
    # crossbar first (under boxes)
    for i in range(4):
        for j in range(4):
            y1 = lane_y[i] + lh / 2 + (j - 1.5) * 9
            y2 = lane_y[j] + lh / 2 + (i - 1.5) * 9
            s.line(tx1, y1, ex0, y2, "#A6A6A6", 1.6)
    for i, y in enumerate(lane_y):
        s.text(62, y + lh / 2 + 6, f"GPU {i}", 18, INK, "bold", "end")
        s.box(tx0, y, tx1 - tx0, lh, "S tokens", "white")
        s.rect(ex0, y, 186, lh, "#fff", NEUTRAL, 2)
        s.box(ex0 + 10, y + 8, 78, lh - 16, f"E{2 * i}", "neutral")
        s.box(ex0 + 98, y + 8, 78, lh - 16, f"E{2 * i + 1}", "neutral")
    s.save(out("fig_ep_layer.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slides 4 and 8: per-GPU received tokens in the running example (S*K = 8 copies per GPU), before / after
BASE_Y, PT_PER_TOK = 300, 20
BAR_X = [92, 194, 296, 398]
BAR_W = 72


def _axes(s):
    s.line(80, BASE_Y, 476, BASE_Y, INK, 2)
    for t in (0, 4, 8, 12):
        y = BASE_Y - t * PT_PER_TOK
        s.line(74, y, 80, y, INK, 2)
        s.text(68, y + 6, str(t), 18, INK, "normal", "end")
    s.raw(f'<text x="22" y="{BASE_Y - 6 * PT_PER_TOK}" font-size="18" fill="{INK}" text-anchor="middle" '
          f'transform="rotate(-90 22 {BASE_Y - 6 * PT_PER_TOK})">tokens received</text>')
    for i, x in enumerate(BAR_X):
        s.text(x + BAR_W / 2, BASE_Y + 26, f"GPU {i}", 18, INK, "bold")


def _stack(s, i, segs):
    """segs: list of (label, tokens, kind) from the bottom."""
    y = BASE_Y
    for label, n, kind in segs:
        h = n * PT_PER_TOK
        s.box(BAR_X[i], y - h, BAR_W, h, label, kind, 18, "bold" if kind == "accent" else "normal", rx=2)
        y -= h
    return y


def _sk_line(s):
    y = BASE_Y - 8 * PT_PER_TOK
    s.line(80, y, 476, y, INK, 2, dash="7,5")
    return y


def fig_skew():
    s = SVG(480, 340)
    _axes(s)
    _stack(s, 0, [("E0", 9, "problem"), ("E1", 3, "problem")])
    _stack(s, 1, [("E2", 7, "problem"), ("E3", 5, "problem")])
    for i, (a, b) in ((2, ("E4", "E5")), (3, ("E6", "E7"))):
        top = _stack(s, i, [(a, 2, "neutral"), (b, 2, "neutral")])
        s.rect(BAR_X[i], BASE_Y - 8 * PT_PER_TOK, BAR_W, top - (BASE_Y - 8 * PT_PER_TOK), "#fff", MUTED, 2, 2, dash="6,4")
        s.text(BAR_X[i] + BAR_W / 2, (top + BASE_Y - 8 * PT_PER_TOK) / 2 + 6, "idle", 18, MUTED, "normal", italic=True)
    y8 = _sk_line(s)
    s.text(476, y8 - 8, "S×K = 8", 18, INK, "bold", "end")
    s.text((BAR_X[0] + BAR_X[1] + BAR_W) / 2, 44, "overloaded", 18, PROBLEM, "bold")
    s.save(out("fig_skew.svg"))


def fig_replicas():
    s = SVG(480, 340)
    _axes(s)
    _stack(s, 0, [("E0", 5, "neutral"), ("E1", 3, "neutral")])
    _stack(s, 1, [("E2", 3, "neutral"), ("E3", 5, "neutral")])
    _stack(s, 2, [("E4", 2, "neutral"), ("E5", 2, "neutral"), ("E0′", 4, "accent")])
    _stack(s, 3, [("E6", 2, "neutral"), ("E7", 2, "neutral"), ("E2′", 4, "accent")])
    y8 = BASE_Y - 8 * PT_PER_TOK
    # replica arrows: home GPU -> receiving GPU (weights forward); drawn as elbows above the bars
    c0, c1, c2, c3 = (x + BAR_W / 2 for x in BAR_X)
    s.path(f"M{c0},{y8 - 4} L{c0},{y8 - 44} L{c2},{y8 - 44} L{c2},{y8 - 5}", ACCENT, 2.5, arrow=True)
    # second arrow with a white halo so the crossing reads as a bridge
    d2 = f"M{c1},{y8 - 4} L{c1},{y8 - 84} L{c3},{y8 - 84} L{c3},{y8 - 5}"
    s.path(d2, "#ffffff", 9)
    s.path(d2, ACCENT, 2.5, arrow=True)
    s.text(c0 + 34, y8 - 52, "E0", 18, ACCENT, "bold")
    s.text(c3 - 34, y8 - 92, "E2", 18, ACCENT, "bold")
    _sk_line(s)
    s.text((c2 + c3) / 2, y8 - 8, "S×K = 8", 18, INK, "bold")
    s.save(out("fig_replicas.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 7: layered architecture
def fig_arch():
    W = 870
    s = SVG(W, 300)
    x0, gap = 86, 10
    widths = [124, 148, 152, 180, 140]
    cols, x = [], x0
    for w in widths:
        cols.append((x, x + w)); x += w + gap
    right = cols[-1][1]
    rows = {"caller": (4, 42), "API": (60, 44), "kernels": (118, 62), "buffers": (194, 62), "fabric": (270, 28)}
    for name, (y, h) in rows.items():
        s.text(76, y + h / 2 + 6, name, 18, MUTED, "bold", "end")
    y, h = rows["caller"]
    s.box(x0, y, right - x0, h, "Framework MoE layer: router · grouped expert GEMM", "neutral")
    y, h = rows["API"]
    mono = "Menlo, Courier New, monospace"
    for (c0, c1, name) in ((0, 1, "dispatch()"), (2, 2, "combine()"), (3, 3, "prefetch_weight()"), (4, 4, "reduce_grad()")):
        xa, xb = cols[c0][0], cols[c1][1]
        s.rect(xa, y, xb - xa, h, ACCENT_FILL, ACCENT, 2)
        s.ctext((xa + xb) / 2, y + h / 2, name, 16, ACCENT, "bold", family=mono)
    y, h = rows["kernels"]
    ks = [(["planning", "kernel"], 2), (["dispatch", "fused permute"], 3), (["combine", "fused unpermute"], 3),
          (["weight", "prefetch"], 1), (["grad", "reduce"], 1)]
    for k, (lines, badge) in enumerate(ks):
        xa, xb = cols[k]
        s.rect(xa, y, xb - xa, h, ACCENT_FILL, ACCENT, 2)
        s.ctext((xa + xb) / 2 + 8, y + h / 2 + 3, lines, 17, INK)
        s.badge(xa + 14, y + 14, badge, r=11, size=15)
    y, h = rows["buffers"]
    bufs = [(0, 0, ["plan", "(multicast)"]), (1, 2, ["token buffer [NvS, H]", "fixed S×K slots"]),
            (3, 3, ["weight pool", "[E/R, H, H′]"]), (4, 4, ["FP32 grad", "pool"])]
    for c0, c1, lines in bufs:
        xa, xb = cols[c0][0], cols[c1][1]
        s.rect(xa, y, xb - xa, h, "#fff", ACCENT, 2, dash="7,4")
        s.ctext((xa + xb) / 2, y + h / 2, lines, 17, INK)
    y, h = rows["fabric"]
    s.box(x0, y, right - x0, h, "One NVLink domain: peer-mapped memory + NVSwitch multicast", "neutral", 17)
    s.save(out("fig_arch.svg"))


# ----------------------------------------------------------------------------------------------------------
# Slide 9: host/GPU timelines, DeepEP v2 training mode vs MoonEP (schematic durations)
def fig_gpu_pipeline():
    s = SVG(870, 286)
    lx = 112

    def lane_label(y, h, name):
        s.text(lx - 12, y + h / 2 + 6, name, 18, INK, "bold", "end")

    def blk(x0, x1, y, h, label, kind="white", weight="normal"):
        s.box(x0, y, x1 - x0, h, label, kind, 18, weight, rx=3)

    # (a) DeepEP v2 with CPU sync
    s.text(0, 20, "(a) DeepEP v2 with CPU sync", 18, INK, "bold", "start")
    hy, hh, gy, gh = 34, 32, 80, 40
    lane_label(hy, hh, "host"); lane_label(gy, gh, "GPU")
    blk(lx, 192, gy, gh, "router")
    blk(192, 342, gy, gh, "dispatch")
    s.rect(342, gy, 110, gh, "url(#hatchRed)", PROBLEM, 2, 3)
    s.raw(f'<rect x="368" y="{gy + 10}" width="58" height="22" fill="#fff"/>')
    s.text(397, gy + gh / 2 + 6, "idle", 18, PROBLEM, "bold")
    blk(452, 622, gy, gh, "expert GEMM")
    blk(622, 772, gy, gh, "combine")
    s.box(342, hy, 110, hh, "read counts", "problem", 17, "bold", rx=3)
    s.line(346, gy - 2, 346, hy + hh + 3, PROBLEM, 2, arrow=True)
    s.line(448, hy + hh + 2, 448, gy - 3, PROBLEM, 2, arrow=True)
    # (b) MoonEP
    s.text(0, 160, "(b) MoonEP", 18, ACCENT, "bold", "start")
    hy, gy = 174, 220
    lane_label(hy, hh, "host"); lane_label(gy, gh, "GPU")
    blk(lx, 192, gy, gh, "router")
    blk(192, 248, gy, gh, "plan", "accent", "bold")
    blk(248, 398, gy, gh, "dispatch", "accent", "bold")
    blk(398, 492, gy, gh, "prefetch", "accent", "bold")
    blk(492, 662, gy, gh, "expert GEMM")
    blk(662, 812, gy, gh, "combine")
    s.box(lx, hy, 300, hh, "enqueue kernels ahead", "white", 17, rx=3)
    # time axis
    s.line(lx, 274, 860, 274, INK, 2, arrow=True)
    s.text(852, 266, "time", 18, INK, "normal", "end")
    s.save(out("fig_gpu_pipeline.svg"))


if __name__ == "__main__":
    fig_ep_layer()
    fig_skew()
    fig_replicas()
    fig_arch()
    fig_gpu_pipeline()


# ----------------------------------------------------------------------------------------------------------
# Slide 13: adoption timeline (dates from GitHub API, K3 report, DeepEP README; see notes/sources.md)
def fig_timeline():
    s = SVG(870, 250)
    y = 125
    x0, x1 = 20, 850
    s.line(x0, y, x1, y, NEUTRAL, 3, arrow=True)
    events = [  # (x, date, lines, above, kind)
        (70, "Jul 24", ["repo public", "MIT license"], True, "accent"),
        (210, "Jul 27", ["Kimi K3 report", "2.8T pre-training"], False, "accent"),
        (350, "Aug 6", ["multi-node", "NVLink fabric"], True, "accent"),
        (490, "Aug 13", ["MXFP4 weight", "prefetch"], False, "accent"),
        (640, "Sep 29", ["DeepEP V2.5 adds", "replica primitives"], True, "neutral"),
        (780, "Oct 13", ["1.17k stars", "138 forks"], False, "neutral"),
    ]
    for x, date, lines, above, kind in events:
        col = ACCENT if kind == "accent" else NEUTRAL
        s.raw(f'<circle cx="{x}" cy="{y}" r="9" fill="{ACCENT_FILL if kind == "accent" else NEUTRAL_FILL}" '
              f'stroke="{col}" stroke-width="2.5"/>')
        if above:
            s.line(x, y - 11, x, y - 34, col, 2)
            s.text(x, y - 92, date, 18, col, "bold")
            s.ctext(x, y - 58, lines, 17, INK)
        else:
            s.line(x, y + 11, x, y + 34, col, 2)
            s.text(x, y + 56, date, 18, col, "bold")
            s.ctext(x, y + 90, lines, 17, INK)
    s.text(850, y - 12, "2026", 18, MUTED, "bold", "end")
    s.save(out("fig_timeline.svg"))


if __name__ == "__main__":
    fig_timeline()


# ----------------------------------------------------------------------------------------------------------
# Slide 5: failed suitors as a comparison grid (sources: K3 2.3.3 / 5.2.1, DeepSeek-V3 3.4, EPLB README)
def fig_suitors():
    s = SVG(870, 236)
    colx = [0, 250, 470, 660, 870]
    heads = ["approach", "balances", "updated", "failure mode"]
    for k, h in enumerate(heads):
        s.text((colx[k] + colx[k + 1]) / 2, 22, h, 18, MUTED, "bold")
    rows = [
        (["Router bias", "(decks 55, 56)"], ["batch-average", "expert load"], ["from the", "previous step"],
         ["one micro-batch", "can still skew"]),
        (["Offline replicas", "(EPLB, DeepSeek-V3)"], ["historical", "GPU load"], ["~every", "10 minutes"],
         ["stale when", "routing drifts"]),
        (["Capped cloning", "(ECHO, UltraEP)"], ["current", "micro-batch"], ["every forward"],
         ["stops when no", "plan fits the cap"]),
    ]
    y0, rh = 36, 64
    for r, cells in enumerate(rows):
        y = y0 + r * (rh + 2)
        s.rect(2, y, 866, rh, NEUTRAL_FILL if r % 2 == 0 else "#fff", "#D9D9D9", 1.5, 3)
        for k, lines in enumerate(cells):
            cx = (colx[k] + colx[k + 1]) / 2
            if k == 0:
                s.ctext(cx, y + rh / 2, lines, 18, INK, "bold")
            elif k == 3:
                s.ctext(cx, y + rh / 2, lines, 18, PROBLEM, "bold")
            else:
                s.ctext(cx, y + rh / 2, lines, 18, INK)
    s.save(out("fig_suitors.svg"))


if __name__ == "__main__":
    fig_suitors()


# ----------------------------------------------------------------------------------------------------------
# Slide 6: the lemma behind the insight (K3 App. E): each under-loaded GPU is filled from ONE overloaded GPU
def fig_insight():
    s = SVG(480, 300)
    # overloaded sources
    src = {"A": 22, "B": 172}
    for name, y in src.items():
        s.rect(0, y, 180, 106, PROBLEM_FILL, PROBLEM, 2)
        s.text(90, y + 30, f"GPU {name}", 18, PROBLEM, "bold")
        s.text(90, y + 54, "overloaded", 17, PROBLEM)
        s.box(14, y + 66, 152, 30, "E/R home experts", "white", 16, rx=3)
    # under-loaded receivers
    rcv = {"C": 6, "D": 108, "E": 210}
    for name, y in rcv.items():
        s.rect(296, y, 182, 84, NEUTRAL_FILL, NEUTRAL, 2)
        s.text(387, y + 26, f"GPU {name}", 18, INK, "bold")
        s.rect(308, y + 38, 158, 34, ACCENT_FILL, ACCENT, 2, 3, dash="6,3")
        s.text(387, y + 61, "≤ E/R replicas", 17, ACCENT, "bold")
    # fills: A->C, A->D, B->E (exactly one incoming arrow per receiver)
    s.line(180, src["A"] + 40, 294, rcv["C"] + 42, ACCENT, 2.5, arrow=True)
    s.line(180, src["A"] + 70, 294, rcv["D"] + 42, ACCENT, 2.5, arrow=True)
    s.line(180, src["B"] + 60, 294, rcv["E"] + 42, ACCENT, 2.5, arrow=True)
    s.text(238, 20, "fill to S×K", 17, ACCENT, "bold")
    s.save(out("fig_insight.svg"))


if __name__ == "__main__":
    fig_insight()
